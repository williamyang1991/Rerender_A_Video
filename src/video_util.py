import os

import cv2
import imageio
import numpy as np


def video_to_frame(video_path: str,
                   frame_dir: str,
                   filename_pattern: str = 'frame%03d.jpg',
                   log: bool = True,
                   frame_edit_func=None):
    os.makedirs(frame_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    if log:
        print('img shape: ', image.shape[0:2])

    count = 0
    while success:
        if frame_edit_func is not None:
            image = frame_edit_func(image)

        cv2.imwrite(os.path.join(frame_dir, filename_pattern % count), image)
        success, image = vidcap.read()
        if log:
            print('Read a new frame: ', success, count)
        count += 1

    vidcap.release()


def frame_to_video(video_path: str, frame_dir: str, fps=30, log=True):

    first_img = True
    writer = imageio.get_writer(video_path, fps=fps)

    file_list = sorted(os.listdir(frame_dir))
    for file_name in file_list:
        if not (file_name.endswith('jpg') or file_name.endswith('png')):
            continue

        fn = os.path.join(frame_dir, file_name)
        curImg = imageio.imread(fn)

        if first_img:
            H, W = curImg.shape[0:2]
            if log:
                print('img shape', (H, W))
            first_img = False

        writer.append_data(curImg)

    writer.close()


def get_fps(video_path: str):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def get_frame_count(video_path: str):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def prepare_frames(input_path: str, output_dir: str, resolution: int, crop):
    l, r, t, b = crop

    def crop_func(frame):
        H, W, C = frame.shape
        left = np.clip(l, 0, W)
        right = np.clip(W - r, left, W)
        top = np.clip(t, 0, H)
        bottom = np.clip(H - b, top, H)
        frame = frame[top:bottom, left:right]
        return resize_image(frame, resolution)

    video_to_frame(input_path, output_dir, '%04d.png', False, crop_func)
