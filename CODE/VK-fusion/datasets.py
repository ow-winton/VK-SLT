from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time
from vidaug import augmentors as va
from augmentation import *

from loguru import logger
from hpman.m import _

# global definition
from definition import *

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip


class S2T_Dataset(Dataset.Dataset):
    def __init__(self, path, phase, args, tokenizer, config, seed=None, training_refurbish=False, aug_rate=0.5):
        # 生成随机种子
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.max_length = config['data']['max_length']
        # self.max_length = 10

        self.img_path = config['data']['img_path']
        self.kps_path = config['data']['keypoint_path']

        self.args = args
        self.aug_rate = aug_rate
        self.phase = phase
        self.config = config
        self.tokenizer = tokenizer
        self.raw_data = utils.load_dataset_file(path)
        self.training_refurbish = training_refurbish
        self.list = [key for key, value in self.raw_data.items()]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        tgt_sample = sample['text']
        length = sample['length']

        name_sample = sample['name']

        img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']], index)
        kp_sample = self.load_imgs([self.kps_path + x for x in sample["kps_path"]], index)

        return name_sample, img_sample, kp_sample, tgt_sample

    def load_imgs(self, paths, index):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        paths = self.length_constraint(paths)
        imgs = torch.zeros(len(paths), 3, self.args.input_size, self.args.input_size)

        batch_image = []
        crop_rect, resize = self.data_augmentation(resize=(self.args.resize, self.args.resize),
                                                   crop_size=self.args.input_size,
                                                   is_train=(self.phase == 'train'), index=index)
        # print('img', crop_rect, resize)
        for i, img_path in enumerate(paths):
            # print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法加载位于 {img_path} 的图像。")
                continue  # 跳过这张图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == 'train':
            seq = self.video_augmentation(index)
            batch_image = seq(batch_image)
            # pass

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i, :, :, :] = img[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

        return imgs

    def length_constraint(self, paths):
        if len(paths) > self.max_length:
            random.seed(self.seed)
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                new_paths.append(paths[i])
            paths = new_paths

        return paths

    def data_augmentation(self, resize=(320, 240), crop_size=224, is_train=True, index=0):
        new_seed = hash((self.seed, index)) % (2 ** 32)
        np.random.seed(new_seed)  # 同步设置 NumPy 的随机种子，如果需要
        if is_train:
            left = np.random.randint(0, resize[0] - crop_size)
            top = np.random.randint(0, resize[1] - crop_size)
        else:
            left = (resize[0] - crop_size) // 2
            top = (resize[1] - crop_size) // 2

        return (left, top, left + crop_size, top + crop_size), resize

    def video_augmentation(self, index=0):
        new_seed = hash((self.seed, index)) % (2 ** 32)
        random.seed(new_seed)
        sometimes = lambda aug: va.Sometimes(self.aug_rate, aug)
        seq = va.Sequential(
            [
                sometimes(va.RandomRotate(30)),
                sometimes(va.RandomResize(0.2)),
                sometimes(va.RandomTranslate(x=10, y=10)),
            ]
        )
        return seq

    def collate_fn(self, batch):
        tgt_batch, img_tmp, src_length_batch, name_batch, kp_tmp = [], [], [], [], []
        for name_sample, img_sample, kp_sample, tgt_sample in batch:
            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            kp_tmp.append(kp_sample)
            tgt_batch.append(tgt_sample)

        max_len = max([len(vid) for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad

        def pad_seq(seq, left_padding, max_length):
            return torch.cat(
                (
                    seq[0][None].expand(left_padding, -1, -1, -1),
                    seq,
                    seq[-1][None].expand(max_length - seq.size(0) - left_padding, -1, -1, -1),
                ), dim=0)

        '填充过的video 和kp，内部存放的batch-size 批次个数据，可以看作多个视频数据输入, 每个数据是一个视频'
        padded_video = [pad_seq(vid, left_pad, max_len) for vid in img_tmp]
        padded_kp = [pad_seq(kp, left_pad, max_len) for kp in kp_tmp]
        '*****************'
        # padded_video = torch.stack(padded_video)
        # print(padded_video.shape)

        video_length = torch.LongTensor([int(np.ceil(len(vid) / 4.0) * 4 + 16) for vid in img_tmp])

        '3d设置'
        # 在填充后，直接使用 max_len 作为每个视频的长度
        # video_length = [max_len] * len(img_tmp)

        img_tmp = [padded_video[i][:video_length[i], :, :, :] for i in range(len(padded_video))]
        kp_tmp = [padded_kp[i][:video_length[i], :, :, :] for i in range(len(padded_kp))]

        '收集batch内数据的长度信息'
        for i in range(len(img_tmp)):
            # print(len(img_tmp[i]))
            src_length_batch.append(len(img_tmp[i]))

        #     '**************************'
        #     print(len(img_tmp[i]))
        # print(video_length)
        # '***************************'
        src_length_batch = torch.tensor(src_length_batch)

        img_batch = torch.cat(img_tmp, 0)
        kp_batch = torch.cat(kp_tmp, 0)

        '3d设置'
        # img_batch = torch.stack(img_tmp)  # 假设每个 img_tmp[i] 都是一个兼容的张量
        # kp_batch = torch.stack(kp_tmp)

        new_src_lengths = (((src_length_batch - 5 + 1) / 2) - 5 + 1) / 2
        '模拟的是一维网络后的长度'
        new_src_lengths = new_src_lengths.long()
        # print(new_src_lengths)
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        # print(mask_gen)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        # print(mask_gen)
        img_padding_mask = (mask_gen != PAD_IDX).long()
        # print(img_padding_mask)

        # with self.tokenizer.as_target_tokenizer():
        # print(tgt_batch)
        tgt_input = self.tokenizer(text_target=tgt_batch, return_tensors="pt", padding=True, truncation=True)
        # print(tgt_input)
        src_input = {}
        src_input['imgs_id'] = img_batch
        src_input['kps_id'] = kp_batch
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths

        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type,
                                              random_shuffle=self.args.random_shuffle, is_train=(self.phase == 'train'))
            # with self.tokenizer.as_target_tokenizer():
            masked_tgt_input = self.tokenizer(text_target=masked_tgt, return_tensors="pt", padding=True,
                                              truncation=True)
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'




