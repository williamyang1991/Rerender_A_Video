import os

import cv2


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


def frame_to_video(video_path: str,
                   frame_dir: str,
                   fps=30,
                   log=True,
                   fourcc='mp4v'):

    first_img = True

    file_list = sorted(os.listdir(frame_dir))
    for file_name in file_list:
        fn = os.path.join(frame_dir, file_name)
        curImg = cv2.imread(fn)

        if first_img:
            H, W = curImg.shape[0:2]
            if log:
                print('img shape', (H, W))
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
            vid_out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            first_img = False

        vid_out.write(curImg)
        if log:
            print(fn)

    vid_out.release()


def get_fps(video_path: str):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps
