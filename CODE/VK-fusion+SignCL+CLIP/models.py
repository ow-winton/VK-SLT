from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path

class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, attn_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.score = None

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]
            mask = mask[:, None, None, :].float()
            dots -= 10000.0 * (1.0 - mask)
        attn = dots.softmax(dim=-1)
        self.score = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

    def visualize(self):
        return self.score


class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        # Cross attention mechanism
        output, _ = self.attention(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        return output
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text, ground_truth):
        loss_i = self.criterion(logits_per_image, ground_truth.argmax(dim=1))
        loss_t = self.criterion(logits_per_text, ground_truth.argmax(dim=1))
        return (loss_i + loss_t) / 2


def make_resnet(name = 'resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        # print(x.shape)
        '由于输入的图像是n个批次的拼接，所以要按照原来的长度分配生成的特征,然后根据src length batch 分成两批并进行pad成相同长短'
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        # print(x.shape)
        return x




class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        '512'
        self.input_size = input_size
        '1024 '
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            '第一层的时候的输入是512，后续其他层的输入是1024 '
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                '增加一个一维卷积层'
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        
        
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        ' Conv1d 需要的 (batch_size, channels, length) 在这种布局中，卷积操作是沿着最后一个维度（长度）进行的。'

        x = x.permute(0,2,1)

        x = self.temporal_conv(x)
        
        return x.permute(0,2,1)

class V_encoder(nn.Module):
    '简单进行映射，然后归一化和激活 '
    def __init__(self,input,out):
        super(V_encoder, self).__init__()

        '线性层'
        self.src_emb = nn.Linear(input, out)
        modules = []
        modules.append(nn.BatchNorm1d(out))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src: Tensor,
                ):
          
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src
    


class TextCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024):
        super(TextCLIP, self).__init__()
        self.text_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()
        self.projection = ProjectionHead(embed_dim, embed_dim)

    def forward(self, tgt_input):
        text_features = self.text_encoder(
            input_ids=tgt_input['input_ids'].cuda(), 
            attention_mask=tgt_input['attention_mask'].cuda(), 
            return_dict=True
        ).last_hidden_state

        # Use the [EOS] token for sentence-level representation
        eos_indices = tgt_input['input_ids'].argmax(dim=-1)
        eos_features = text_features[torch.arange(text_features.size(0)), eos_indices]
        
        # Project features
        output = self.projection(eos_features)
        return output, text_features


class ImageCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model =  FeatureExtracter(config) 
        
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Cross attention module
        self.cross_attention = CrossAttentionModule(embed_dim=embed_dim)

        # Projection head
        self.projection = ProjectionHead(embed_dim, embed_dim)
        
        
    def forward(self, src_input):
       
        x, images = self.model(src_input['imgs_id'].cuda(), src_input['kps_id'].cuda(),src_input['src_length_batch']) # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
                # Cross attention refinement
        refined_features = self.cross_attention(query=cls_token, key=x, value=x)
        # output = self.lm_head(last_hidden_state[:, 0, :])
        output = self.projection(refined_features[:, 0, :])
        return output, images

class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))

    
    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids,
                    attention_mask = tgt_input['attention_mask'].cuda(),
                    encoder_hidden_states = encoder_hidden_states,
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits
    
        
class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, embed_dim=embed_dim)
        self.model_images = ImageCLIP(config, embed_dim=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        image_features , frames_feature = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth,frames_feature

    def get_images_feature(self, src_input):
        image_features = self.model_images.model(src_input['input_ids'].cuda(),
                                                 src_input['src_length_batch'])  # [b, n, c](src_input)

        return image_features
    
class FeatureExtracter(nn.Module):
    def __init__(self,config, frozen = False,  ):
        super(FeatureExtracter, self).__init__()
        self.ca = SimpleChannelAttention(3)
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

    def forward(self, src_input, src_keypoint, src_length_batch):
        """
        src_input 源输入 n个batch 拼接的一个新视频
        src_length_batch 存储了n个视频长度的信息， 用来把拼接的视频的重新分成n个批次的数据
        """
        src = self.ca(src_input) +self.ca(src_keypoint)
        
        # src = F_vision.resize(src, (224, 224))
        
        images  = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(images )
        
        return src,images


def config_decoder(config, decoder_type):
    from transformers import AutoConfig

    if decoder_type == 'LD':
        
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['visual_encoder'])/'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))

class gloss_free_model(nn.Module):
    def __init__(self, config, args, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(config, frozen = False)

        self.text_model = config_decoder(config, args.decoder_type)
        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(input=1024,out=1024)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

    def share_forward(self, input):
        feature, image_feats = self.backbone(input['imgs_id'].cuda(), input['kps_id'].cuda(),input['src_length_batch'])
        attention_mask = input['attention_mask']
        inputs_embeds = self.sign_emb(feature)
        inputs_embeds = self.embed_scale * inputs_embeds
        return inputs_embeds, attention_mask, image_feats
    def forward(self, input, target):
        inputs_embeds, attention_mask,frames_feature = self.share_forward(input)
        '根据不同的文本模型，这里可能要调整'
        out = self.text_model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask.cuda(),
            labels = target['input_ids'].cuda(),
            decoder_attention_mask = target['attention_mask'].cuda(),
            return_dict = True,
        )
        output = out['encoder_last_hidden_state'][:, 0, :]
        return out['logits'],output, frames_feature
        
    def generate(self,src_input,target, max_new_tokens,num_beams,decoder_start_token_id ):
        inputs_embeds, attention_mask , frames_feature= self.share_forward(src_input)

        out = self.text_model.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out


class SimpleChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(SimpleChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels, bias=False),  # 直接处理通道，不降维
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)