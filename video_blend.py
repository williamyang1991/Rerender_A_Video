import argparse
import os
import platform
import struct
import subprocess
import time
from typing import List

import cv2
import numpy as np
import torch.multiprocessing as mp
from numba import njit

import blender.histogram_blend as histogram_blend
from blender.guide import (BaseGuide, ColorGuide, EdgeGuide, PositionalGuide,
                           TemporalGuide)
from blender.poisson_fusion import poisson_fusion
from blender.video_sequence import VideoSequence
from flow.flow_utils import flow_calc
from src.video_util import frame_to_video

OPEN_EBSYNTH_LOG = False
MAX_PROCESS = 8

os_str = platform.system()

if os_str == 'Windows':
    ebsynth_bin = '.\\deps\\ebsynth\\bin\\ebsynth.exe'
elif os_str == 'Linux':
    ebsynth_bin = './deps/ebsynth/bin/ebsynth'
elif os_str == 'Darwin':
    ebsynth_bin = './deps/ebsynth/bin/ebsynth.app'
else:
    print('Cannot recognize OS. Run Ebsynth failed.')
    exit(0)


@njit
def g_error_mask_loop(H, W, dist1, dist2, output, weight1, weight2):
    for i in range(H):
        for j in range(W):
            if weight1 * dist1[i, j] < weight2 * dist2[i, j]:
                output[i, j] = 0
            else:
                output[i, j] = 1
            if weight1 == 0:
                output[i, j] = 0
            elif weight2 == 0:
                output[i, j] = 1


def g_error_mask(dist1, dist2, weight1=1, weight2=1):
    H, W = dist1.shape
    output = np.empty_like(dist1, dtype=np.byte)
    g_error_mask_loop(H, W, dist1, dist2, output, weight1, weight2)
    return output


def create_sequence(base_dir, beg, end, interval, key_dir):
    sequence = VideoSequence(base_dir, beg, end, interval, 'video', key_dir,
                             'tmp', '%04d.png', '%04d.png')
    return sequence


def process_one_sequence(i, video_sequence: VideoSequence):
    interval = video_sequence.interval
    for is_forward in [True, False]:
        input_seq = video_sequence.get_input_sequence(i, is_forward)
        output_seq = video_sequence.get_output_sequence(i, is_forward)
        flow_seq = video_sequence.get_flow_sequence(i, is_forward)
        key_img_id = i if is_forward else i + 1
        key_img = video_sequence.get_key_img(key_img_id)
        for j in range(interval - 1):
            i1 = cv2.imread(input_seq[j])
            i2 = cv2.imread(input_seq[j + 1])
            flow_calc.get_flow(i1, i2, flow_seq[j])

        guides: List[BaseGuide] = [
            ColorGuide(input_seq),
            EdgeGuide(input_seq,
                      video_sequence.get_edge_sequence(i, is_forward)),
            TemporalGuide(key_img, output_seq, flow_seq,
                          video_sequence.get_temporal_sequence(i, is_forward)),
            PositionalGuide(flow_seq,
                            video_sequence.get_pos_sequence(i, is_forward))
        ]
        weights = [6, 0.5, 0.5, 2]
        for j in range(interval):
            # key frame
            if j == 0:
                img = cv2.imread(key_img)
                cv2.imwrite(output_seq[0], img)
            else:
                cmd = f'{ebsynth_bin} -style {os.path.abspath(key_img)}'
                for g, w in zip(guides, weights):
                    cmd += ' ' + g.get_cmd(j, w)

                cmd += (f' -output {os.path.abspath(output_seq[j])}'
                        ' -searchvoteiters 12 -patchmatchiters 6')
                if OPEN_EBSYNTH_LOG:
                    print(cmd)
                subprocess.run(cmd,
                               shell=True,
                               capture_output=not OPEN_EBSYNTH_LOG)


def process_sequences(i_arr, video_sequence: VideoSequence):
    for i in i_arr:
        process_one_sequence(i, video_sequence)


def run_ebsynth(video_sequence: VideoSequence):

    beg = time.time()

    processes = []
    mp.set_start_method('spawn')

    n_process = min(MAX_PROCESS, video_sequence.n_seq)
    cnt = video_sequence.n_seq // n_process
    remainder = video_sequence.n_seq % n_process

    prev_idx = 0

    for i in range(n_process):
        task_cnt = cnt + 1 if i < remainder else cnt
        i_arr = list(range(prev_idx, prev_idx + task_cnt))
        prev_idx += task_cnt
        p = mp.Process(target=process_sequences, args=(i_arr, video_sequence))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    end = time.time()

    print(f'ebsynth: {end-beg}')


@njit
def assemble_min_error_img_loop(H, W, a, b, error_mask, out):
    for i in range(H):
        for j in range(W):
            if error_mask[i, j] == 0:
                out[i, j] = a[i, j]
            else:
                out[i, j] = b[i, j]


def assemble_min_error_img(a, b, error_mask):
    H, W = a.shape[0:2]
    out = np.empty_like(a)
    assemble_min_error_img_loop(H, W, a, b, error_mask, out)
    return out


def load_error(bin_path, img_shape):
    img_size = img_shape[0] * img_shape[1]
    with open(bin_path, 'rb') as fp:
        bytes = fp.read()

    read_size = struct.unpack('q', bytes[:8])
    assert read_size[0] == img_size
    float_res = struct.unpack('f' * img_size, bytes[8:])
    res = np.array(float_res,
                   dtype=np.float32).reshape(img_shape[0], img_shape[1])
    return res


def process_seq(video_sequence: VideoSequence,
                i,
                blend_histogram=True,
                blend_gradient=True):

    key1_img = cv2.imread(video_sequence.get_key_img(i))
    img_shape = key1_img.shape
    interval = video_sequence.interval
    beg_id = video_sequence.get_sequence_beg_id(i)

    oas = video_sequence.get_output_sequence(i)
    obs = video_sequence.get_output_sequence(i, False)

    binas = [x.replace('jpg', 'bin') for x in oas]
    binbs = [x.replace('jpg', 'bin') for x in obs]

    obs = [obs[0]] + list(reversed(obs[1:]))
    inputs = video_sequence.get_input_sequence(i)
    oas = [cv2.imread(x) for x in oas]
    obs = [cv2.imread(x) for x in obs]
    inputs = [cv2.imread(x) for x in inputs]
    flow_seq = video_sequence.get_flow_sequence(i)

    dist1s = []
    dist2s = []
    for i in range(interval - 1):
        bin_a = binas[i + 1]
        bin_b = binbs[i + 1]
        dist1s.append(load_error(bin_a, img_shape))
        dist2s.append(load_error(bin_b, img_shape))

    lb = 0
    ub = 1
    beg = time.time()
    p_mask = None

    # write key img
    blend_out_path = video_sequence.get_blending_img(beg_id)
    cv2.imwrite(blend_out_path, key1_img)

    for i in range(interval - 1):
        c_id = beg_id + i + 1
        blend_out_path = video_sequence.get_blending_img(c_id)

        dist1 = dist1s[i]
        dist2 = dist2s[i]
        oa = oas[i + 1]
        ob = obs[i + 1]
        weight1 = i / (interval - 1) * (ub - lb) + lb
        weight2 = 1 - weight1
        mask = g_error_mask(dist1, dist2, weight1, weight2)
        if p_mask is not None:
            flow_path = flow_seq[i]
            flow = flow_calc.get_flow(inputs[i], inputs[i + 1], flow_path)
            p_mask = flow_calc.warp(p_mask, flow, 'nearest')
            mask = p_mask | mask
        p_mask = mask

        # Save tmp mask
        # out_mask = np.expand_dims(mask, 2)
        # cv2.imwrite(f'mask/mask_{c_id:04d}.jpg', out_mask * 255)

        min_error_img = assemble_min_error_img(oa, ob, mask)
        if blend_histogram:
            hb_res = histogram_blend.blend(oa, ob, min_error_img,
                                           (1 - weight1), (1 - weight2))

        else:
            # hb_res = min_error_img
            tmpa = oa.astype(np.float32)
            tmpb = ob.astype(np.float32)
            hb_res = (1 - weight1) * tmpa + (1 - weight2) * tmpb

        # cv2.imwrite(blend_out_path, hb_res)

        # gradient blend
        if blend_gradient:
            res = poisson_fusion(hb_res, oa, ob, mask)
        else:
            res = hb_res

        cv2.imwrite(blend_out_path, res)
    end = time.time()
    print('others:', end - beg)


def main(args):
    global MAX_PROCESS
    MAX_PROCESS = args.n_proc

    video_sequence = create_sequence(f'{args.name}', args.beg, args.end,
                                     args.itv, args.key)
    if not args.ne:
        run_ebsynth(video_sequence)
    blend_histogram = True
    blend_gradient = args.ps
    for i in range(video_sequence.n_seq):
        process_seq(video_sequence, i, blend_histogram, blend_gradient)
    if args.output:
        frame_to_video(args.output, video_sequence.blending_dir, args.fps,
                       False)
    if not args.tmp:
        video_sequence.remove_out_and_tmp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Path to input video')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='Path to output video')
    parser.add_argument('--fps',
                        type=float,
                        default=30,
                        help='The FPS of output video')
    parser.add_argument('--beg',
                        type=int,
                        default=1,
                        help='The index of the first frame to be stylized')
    parser.add_argument('--end',
                        type=int,
                        default=101,
                        help='The index of the last frame to be stylized')
    parser.add_argument('--itv',
                        type=int,
                        default=10,
                        help='The interval of key frame')
    parser.add_argument('--key',
                        type=str,
                        default='keys0',
                        help='The subfolder name of stylized key frames')
    parser.add_argument('--n_proc',
                        type=int,
                        default=8,
                        help='The max process count')
    parser.add_argument('-ps',
                        action='store_true',
                        help='Use poisson gradient blending')
    parser.add_argument(
        '-ne',
        action='store_true',
        help='Do not run ebsynth (use previous ebsynth output)')
    parser.add_argument('-tmp',
                        action='store_true',
                        help='Keep temporary output')

    args = parser.parse_args()
    main(args)
