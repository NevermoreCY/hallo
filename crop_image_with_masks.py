# pylint: disable=W1203,W0718
"""
This module is used to process videos to prepare data for training. It utilizes various libraries and models
to perform tasks such as video frame extraction, audio extraction, face mask generation, and face embedding extraction.
The script takes in command-line arguments to specify the input and output directories, GPU status, level of parallelism,
and rank for distributed processing.

Usage:
    python -m scripts.data_preprocess --input_dir /path/to/video_dir --dataset_name dataset_name --gpu_status --parallelism 4 --rank 0

Example:
    python -m scripts.data_preprocess -i data/videos -o data/output -g -p 4 -r 0
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm
import numpy as np
import ffmpeg
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories(video_path: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = video_path
    dirs = {
        "face_mask": base_dir / "face_mask",
        "sep_pose_mask": base_dir / "sep_pose_mask",
        "sep_face_mask": base_dir / "sep_face_mask",
        "sep_lip_mask": base_dir / "sep_lip_mask",
        "face_emb": base_dir / "face_emb",
        "audio_emb": base_dir / "audio_emb",
        "cropped_img": base_dir / "cropped_img",
        "cropped_video": base_dir / "cropped_video",
        "cropped_no_aud": base_dir / "cropped_no_aud"

    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs



def squre_bbox(height, width,x_min, x_max, y_min, y_max, expand_ratio, shift_up_ratio):
    # note y is for height(row) , x is for width(col)
    x_len = x_max - x_min
    y_len = y_max - y_min



    shift_up_pad = int( y_len * (shift_up_ratio))
    expand_pad = int(max(x_len, y_len) * (expand_ratio-1) / 2 )

    print("expand_pad = ", expand_pad)
    print("before squre : ", x_len, y_len)

    diff = x_len - y_len
    print("diff : ", diff )

    # try to make it a square (x_len = y_len)
    if diff > 0:
        # case when x is larger than y
        pad_both_side = diff // 2
        y_min = y_min - pad_both_side
        y_max = y_max + pad_both_side
        if diff % 2 == 1:
            y_max += 1
    else:
        pad_both_side = (-1*diff) // 2
        x_min = x_min - pad_both_side
        x_max = x_max + pad_both_side
        if diff % 2 == 1:
            x_max += 1

    print("After padding :", x_min, x_max, y_min, y_max)

    # expand
    x_min -= expand_pad
    x_max += expand_pad
    y_min -= expand_pad
    y_max += expand_pad

    print("After expand : ", x_min, x_max, y_min, y_max)

    # shift up
    y_min -= shift_up_pad
    y_max -= shift_up_pad

    print("After shift up ")
    print(x_min, x_max, y_min, y_max)

    print("width" ,width , "height", height)
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    print("1",x_min, x_max, y_min, y_max)
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    print("2",x_min, x_max, y_min, y_max)
    if x_max > width:
        x_min -= (x_max-width)
        x_max = width
    print("3",x_min, x_max, y_min, y_max)
    if y_max > height:
        y_min -= (y_max-height)
        y_max = height

    if y_min < 0:
        y_min = 0
    if x_min< 0:
        x_min = 0
    print("After all ")
    print(x_min, x_max, y_min, y_max)

    x_len = x_max - x_min
    y_len = y_max - y_min

    print("after squre : ", x_len, y_len)

    return int(x_min), int(x_max), int(y_min), int(y_max)


def create_video_from_images(image_folder, audio_file, output_video_file, final_path, fps=25):
    # 获取图片列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 确保图片按顺序排序

    # print(images)

    # 读取第一张图片以确定视频的宽度和高度
    image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(image_path)
    height, width, _ = frame.shape

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 遍历图片并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    video_stream = ffmpeg.input(output_video_file)
    audio_stream = ffmpeg.input(audio_file)
    output_stream = ffmpeg.output(video_stream, audio_stream, final_path, vcodec='copy', acodec='aac')
    output_stream = ffmpeg.run(output_stream,overwrite_output=True)
    # # 使用 ffmpeg 合并音频和视频
    # (
    #     ffmpeg
    #     .input(output_video_file)
    #     .input(audio_file)
    #     .output('output_with_audio.mp4', vcodec='copy', acodec='aac')
    #     .overwrite_output()
    #     .run()
    # )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-i", "--input_dir", type=Path,
                        required=True, help="Directory containing videos")
    parser.add_argument("-o", "--output_dir", type=Path,
                        help="Directory to save results, default is parent dir of input dir")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Specify data processing step 1 or 2, you should run 1 and 2 sequently")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    parser.add_argument("-re", "--repeat", default=1, type=int,
                        help="For each video , how many times do you want to crop")
    parser.add_argument("-ra", "--random", default=False, type=bool,
                        help="whether or not use random expand ratio ")


    # python corp_image_with_masks.py -i E:/bili_data_test_08_22/videos
    # python -m scripts.data_preprocess --input_dir E:/bili_data_test_08_22/videos --step 1

    args = parser.parse_args()
    setup_directories(args.input_dir)

    if args.output_dir is None:
        args.output_dir = args.input_dir.parent

    target_width = 512
    target_height = 512
    expand_ratio = 1.2
    shift_up_ratio = 0.2
    print("\n Random :", args.random)
    print("\n Repeat :", args.repeat)
    images_dir = os.path.join(args.input_dir , "images")
    images_folders = os.listdir(images_dir)
    print(images_dir)
    print(images_folders)

    meta_datas = {}

    for folder in images_folders:
        mask_name = folder+".png"
        mask_path = os.path.join(args.input_dir , "face_mask" , mask_name)
        print(f"mask_path is {mask_path}")
        if os.path.exists(mask_path):
            print("mask exists")
            folder_path = os.path.join(images_dir , folder)
            image_list = os.listdir(folder_path)
            # os.path.join()
            # load mask, get bounding box
            # mask = cv2.imread(str(mask_path))
            # mask_path = "E:\Data\hallo_test_video\face_mask\corner4.png"
            mask = cv2.imread(mask_path)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            try:
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
            except Exception as e:
                print(str(e))

            print(mask.shape)
            row_len = mask.shape[0]
            col_len = mask.shape[1]
            print("***\n\n\n DEBUG \n\n\n ", type(x_min))
            print(f"xmin is {x_min}, ymin is {y_min} , xmax is {x_max} , ymax is {y_max}")
            meta_data = {}
            meta_data["face_bbox"] = [int(x_min),int(x_max),int(y_min),int(y_max), int(col_len), int(row_len)]

            repeat_count = args.repeat
            random = args.random
            x_len = x_max - x_min
            y_len = y_max - y_min

            print(x_len, y_len)

            print("\n\n debug \n\n")
            # 计算max expand ratio
            longer_side = max(x_len,y_len)
            print(x_len,y_len,longer_side)
            max_x_expand_ratio = col_len / longer_side
            max_y_expand_ratio = row_len / longer_side

            max_expand_ratio = min(max_x_expand_ratio,max_y_expand_ratio)
            print(max_x_expand_ratio, max_y_expand_ratio, max_expand_ratio)
            middle_ratio = (max_expand_ratio -1) /2

            crop_bboxs = []
            # now we could repeat crop data by at most 2 times
            for r_id in range(repeat_count):
                print(f"folder {folder} with r_id {r_id}")
                if random:
                    # No repeat, just random corp data once
                    if repeat_count == 1:
                        expand_ratio = np.random.uniform(1.05, 1 + 0.7*(max_expand_ratio-1))
                    elif repeat_count == 2 and r_id ==0:
                        expand_ratio = np.random.uniform(1.05, 1 + 0.3*(max_expand_ratio-1))
                    elif repeat_count == 2 and r_id == 1:
                        expand_ratio = np.random.uniform(1 +  0.4*(max_expand_ratio-1) , 1 + 0.7*(max_expand_ratio-1))
                    print("\n\nDebug expand:", 1, 1 + 0.3*(max_expand_ratio-1), 1 + 0.4*(max_expand_ratio-1), 1 + 0.7*(max_expand_ratio-1), max_x_expand_ratio)
                # if we are repeating

                squared_bbox = squre_bbox(row_len, col_len, x_min, x_max, y_min, y_max, expand_ratio,shift_up_ratio)
                x_min, x_max, y_min, y_max = squared_bbox
                x_len = x_max - x_min
                y_len = y_max - y_min


                print("***\n\n\n DEBUG \n\n\n ", type(x_min) , type(r_id), type(expand_ratio))
                crop_bboxs.append([x_min, x_max, y_min, y_max, x_len /col_len, y_len / row_len ,r_id,expand_ratio, max_expand_ratio, middle_ratio])
                print(crop_bboxs)
                crop_folder_name = folder +"_"+ str(r_id)
                image_folder = os.path.join(args.input_dir, "cropped_img", crop_folder_name)
                os.makedirs(image_folder, exist_ok=True)
                # for each image, corp with bounding box
                for image in image_list:
                    image_path = os.path.join(folder_path , image)
                    # print(f"image_path is {image_path}")
                    img = cv2.imread(image_path)
                    # print(img.shape)
                    img = img[y_min:y_max, x_min:x_max]
                    # print(img.shape)
                    img = cv2.resize(img, (target_width, target_height))
                    # print(img.shape)

                    out_path = os.path.join(args.input_dir , "cropped_img", crop_folder_name , image )
                    # print(out_path)
                    cv2.imwrite(out_path, img)

                # combine image and audio into a video
                audio_name = folder+".wav"
                audio_file = os.path.join(args.input_dir , "audios", audio_name )
                video_name = folder+"_"+ str(r_id) +".mp4"
                # video_with_audio = folder + ".mp4"
                output_video_file = os.path.join(args.input_dir, "cropped_no_aud",video_name)
                print(f"image folder is {image_folder}, audio file is {audio_file}, output video file is {output_video_file}")
                output_video_with_audio_file = os.path.join(args.input_dir, "cropped_video",video_name)

                create_video_from_images(image_folder, audio_file, output_video_file,output_video_with_audio_file )


                # check if first bbox is already large enough, if so there is no need to repeat
                if r_id == 0:
                    if middle_ratio <= 0.225:
                        print(f"For file {folder} , First bbox is already large enough, no need to repeat")
                        break
            meta_data["crop_bboxs"] = crop_bboxs
            meta_datas[folder] = meta_data


    out_path = os.path.join(args.input_dir , "meta_data.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(meta_datas, f)
    print("MetaData is saved at %s" % out_path)















