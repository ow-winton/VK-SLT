import os
import cv2
import json
import numpy as np


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print(f"Error opening video file: {video_path}")
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def video_filter(df, data_save_path, atype):
    df = df.copy()
    df = df[['START_REALIGNED', 'END_REALIGNED', 'SENTENCE']]
    df['LENGTH'] = df['END_REALIGNED'] - df['START_REALIGNED']
    df = df[['SENTENCE', 'LENGTH']]
    if atype == 'train':
        df['VIDEO_PATH'] = os.path.join(data_save_path, "howtosign/train/raw_videos/") + df.index + '.mp4'
        df['JSON_PATH'] = os.path.join(data_save_path, "howtosign/train/openpose_output/json/") + df.index + '/'
    elif atype == 'val':
        df['VIDEO_PATH'] = os.path.join(data_save_path, "howtosign/val/raw_videos/") + df.index + '.mp4'
        df['JSON_PATH'] = os.path.join(data_save_path, "howtosign/val/openpose_output/json/") + df.index + '/'

    elif atype == 'test':
        df['VIDEO_PATH'] = os.path.join(data_save_path, "howtosign/test/raw_videos/") + df.index + '.mp4'
        df['JSON_PATH'] = os.path.join(data_save_path, "howtosign/test/openpose_output/json/") + df.index + '/'

    df['FRAME_COUNT'] = df['VIDEO_PATH'].apply(get_frame_count)
    return df.dropna()


def filter_main_df(train_data):
    # 计算90%分位数
    threshold = train_data['FRAME_COUNT'].quantile(0.90)
    print("90%分位数阈值:", threshold)

    # 过滤掉超过阈值的数据
    filtered_Data = train_data[train_data['FRAME_COUNT'] <= threshold]

    print("过滤后的数据行数:", filtered_Data.shape[0])
    return filtered_Data


def frame_generate(train_data, downsample_rate, type):
    if type == 'train':
        output_base_folder = './how2sign/frame/train'
    elif type == 'val':
        output_base_folder = './how2sign/frame/val'
    elif type == 'test':
        output_base_folder = './how2sign/frame/test'
    else:
        return ValueError
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    for index, row in train_data.iterrows():

        video_path = row['VIDEO_PATH']  # 访问列值

        output_folder = os.path.join(output_base_folder, index)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # print(f"Error: Cannot open video {video_path}.")
            return
        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 如果没有帧了就退出循环

            # 只保存每第 downsample_rate 帧
            if frame_count % downsample_rate == 0:
                if frame is not None and frame.size > 0:
                    resized_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)

                    frame_file = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")

                    cv2.imwrite(frame_file, resized_frame)
                    # print(f"Attempt to save frame {saved_frame_count}: {success}")
                    saved_frame_count += 1
                else:
                    print(f"Warning: Frame {frame_count} is empty or corrupted.")

            frame_count += 1
        cap.release()

    print(f'生成帧数据集结束，最终数据为{index},{saved_frame_count}')
    return


def draw_keypoints(image, keypoints, confidence_threshold=0.1, original_size=(1280, 720), target_size=(256, 256),
                   radius=2):
    """
    在图像中绘制关键点，并对关键点周围区域进行扩展。

    参数：
    - image: 输入的图像 (height, width, 3)。
    - keypoints: 关键点列表，形状为 (n, 3)。
    - confidence_threshold: 绘制关键点的置信度阈值。
    - original_size: 原始图像尺寸 (width, height)。
    - target_size: 缩放后的目标图像尺寸 (width, height)。
    - radius: 扩展半径。

    返回：
    - image: 绘制了关键点并扩展后的图像。
    """
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i:i + 3]
        if confidence > confidence_threshold:
            # 缩放坐标到目标图像尺寸
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)

            # 颜色表示
            color = (min(max(int(float(confidence) * 255), 0), 255), 0, 0)  # (B, G, R)  # (B, G, R)

            # 确保坐标在目标图像范围内
            if 0 <= scaled_x < target_size[0] and 0 <= scaled_y < target_size[1]:
                # 绘制中心点
                image[scaled_y, scaled_x] = color

                # 对关键点周围进行扩展
                row_start = max(0, scaled_y - radius)
                row_end = min(target_size[1], scaled_y + radius + 1)
                col_start = max(0, scaled_x - radius)
                col_end = min(target_size[0], scaled_x + radius + 1)
                image[row_start:row_end, col_start:col_end] = color

    return image


def process_json_file(json_file, image_size):
    with open(json_file, 'r') as f:
        data = json.load(f)

    image = np.zeros((int(image_size[1]), int(image_size[0]), 3), dtype=np.uint8)

    for person in data['people']:

        if 'hand_left_keypoints_2d' in person:
            draw_keypoints(image, person['hand_left_keypoints_2d'])
        if 'hand_right_keypoints_2d' in person:
            draw_keypoints(image, person['hand_right_keypoints_2d'])

    return image


def keypoint_generate(train_data, downsample_rate, type):
    image_size = (256, 256)
    if type == 'train':
        output_base_folder = './how2sign/frame/key/train'
    elif type == 'val':
        output_base_folder = './how2sign/frame/key/val'
    elif type == 'test':
        output_base_folder = './how2sign/frame/key/test'
    else:
        return ValueError
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    for index, row in train_data.iterrows():
        output_folder = os.path.join(output_base_folder, index)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        dir_path = row['JSON_PATH']  # 访问列值
        json_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
        total_files = len(json_files)
        selected_frames = list(range(0, total_files, downsample_rate))

        for frame_idx, idx in enumerate(selected_frames):
            json_file = os.path.join(dir_path, json_files[idx])
            image = process_json_file(json_file, image_size)
            frame_name = f"frame_{str(frame_idx).zfill(4)}.png"
            output_path = os.path.join(output_folder, frame_name)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)

    print(f'生成关键点数据集结束，最终数据为{index}')
    return
