import argparse
import json
import os

from src.video_util import get_fps

parser = argparse.ArgumentParser()
parser.add_argument('tid', type=int)
parser.add_argument('-ne', action='store_true')
args = parser.parse_args()

video_cfg = 'videos.json'
video_trans_cfg = 'video_trans.json'
with open(video_cfg, 'r') as fp:
    video_cfg = json.load(fp)
with open(video_trans_cfg, 'r') as fp:
    video_trans_cfg = json.load(fp)
task_config = video_trans_cfg['tasks'][args.tid]

video_list = video_cfg['list']
video_base_dir = video_list[task_config['v_idx']]['dir']
o_video = task_config.get('o_video', '')
fps = get_fps(video_list[task_config['v_idx']]['video'])

end_frame = task_config['frame_count'] - 1
interval = task_config['interval']
key_dir = os.path.split(task_config['o_dir'])[-1]
use_e = '-ne' if args.ne else ''
o_video_cmd = f'--output {o_video}' if len(o_video) > 0 else ''

cmd = (
    f'python video_blend.py {video_base_dir} --beg 1 --end {end_frame} --itv '
    f'{interval} --key {key_dir} {use_e} {o_video_cmd} --fps {fps}')
print(cmd)
os.system(cmd)
