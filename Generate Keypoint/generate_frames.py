import pandas as pd
from utils import *
from loguru import logger
if __name__ == '__main__':
    data_save_path = './'
    downsample_rate = 3
    logger.info(f'start downsample for video, rate{downsample_rate}')
    
    # train_Data = pd.read_csv('./data33_train.csv', sep='\t', index_col='SENTENCE_NAME')
    # val_Data = pd.read_csv('./data33_val.csv', sep='\t', index_col='SENTENCE_NAME')
    # test_Data = pd.read_csv('./data33_test.csv', sep='\t', index_col='SENTENCE_NAME')

    # train_Data = video_filter(train_Data, data_save_path, atype='train')

    # val_Data = video_filter(val_Data, data_save_path, atype='val')

    # test_Data = video_filter(test_Data, data_save_path, atype='test')

#     filter_train_Data= filter_main_df(train_Data)
#     filter_val_Data= filter_main_df(val_Data)
#     filter_test_Data= filter_main_df(test_Data)

#     filter_train_Data.to_csv('filter_train_Data.csv')
#     filter_val_Data.to_csv('filter_val_Data.csv')
#     filter_test_Data.to_csv('filter_test_Data.csv')

    train_data = pd.read_csv('filter_train_Data.csv',index_col= 'SENTENCE_NAME')
    val_data = pd.read_csv('filter_val_Data.csv',index_col= 'SENTENCE_NAME')
    test_data = pd.read_csv('filter_test_Data.csv', index_col='SENTENCE_NAME')

    print(len(train_data),train_data['FRAME_COUNT'].max())
    print(len(val_data),val_data['FRAME_COUNT'].max())
    print(len(test_data), test_data['FRAME_COUNT'].max())

    train_data['SENTENCE'] = train_data['SENTENCE'].str.lower()
    val_data['SENTENCE'] = val_data['SENTENCE'].str.lower()
    test_data['SENTENCE'] = test_data['SENTENCE'].str.lower()

    frame_generate(train_data,downsample_rate,'train')
    frame_generate(val_data,downsample_rate,'val')
    frame_generate(test_data,downsample_rate,'test')
    
    logger.info(f'start downsample for keypoint, rate{downsample_rate}')
    keypoint_generate(train_data, downsample_rate, 'train')
    keypoint_generate(val_data, downsample_rate, 'val')
    keypoint_generate(test_data, downsample_rate, 'test')