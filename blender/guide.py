import os

import cv2
import numpy as np

from flow.flow_utils import flow_calc, read_flow


class BaseGuide:

    def __init__(self):
        ...

    def get_cmd(self, i, weight) -> str:
        return (f'-guide {os.path.abspath(self.imgs[0])} '
                f'{os.path.abspath(self.imgs[i])} -weight {weight}')


class ColorGuide(BaseGuide):

    def __init__(self, imgs):
        super().__init__()
        self.imgs = imgs


class PositionalGuide(BaseGuide):

    def __init__(self, flows, save_paths):
        super().__init__()
        flows = [read_flow(f) for f in flows]
        # TODO: modify the format of flow to numpy
        H, W = flows[0].shape[2:]
        first_img = PositionalGuide.__generate_first_img(H, W)
        prev_img = first_img
        imgs = [first_img]
        cid = 0
        for flow in flows:
            cur_img = flow_calc.warp(prev_img, flow,
                                     'nearest').astype(np.uint8)

            gray = cur_img[:, :, 1].astype(np.float32)
            gray += cur_img[:, :, 2]
            gray /= 2
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
            mask = mask.astype(np.uint8)
            cur_img = cv2.inpaint(cur_img, mask, 30, cv2.INPAINT_TELEA)
            prev_img = cur_img
            imgs.append(cur_img)
            cid += 1
            cv2.imwrite(f'guide/{cid}.jpg', mask)

        for path, img in zip(save_paths, imgs):
            cv2.imwrite(path, img)
        self.imgs = save_paths

    @staticmethod
    def __generate_first_img(H, W):
        Hs = np.linspace(0, 1, H)
        Ws = np.linspace(0, 1, W)
        i, j = np.meshgrid(Hs, Ws, indexing='ij')
        r = (i * 255).astype(np.uint8)
        g = (j * 255).astype(np.uint8)
        b = np.zeros(r.shape)
        res = np.stack((b, g, r), 2)
        return res


class EdgeGuide(BaseGuide):

    def __init__(self, imgs, save_paths):
        super().__init__()
        edges = [EdgeGuide.__generate_edge(cv2.imread(img)) for img in imgs]
        for path, img in zip(save_paths, edges):
            cv2.imwrite(path, img)
        self.imgs = save_paths

    @staticmethod
    def __generate_edge(img):
        filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        res = cv2.filter2D(img, -1, filter)
        return res


class TemporalGuide(BaseGuide):

    def __init__(self, key_img, stylized_imgs, flows, save_paths):
        super().__init__()
        self.flows = [read_flow(f) for f in flows]
        self.stylized_imgs = stylized_imgs
        self.imgs = save_paths

        first_img = cv2.imread(key_img)
        cv2.imwrite(self.imgs[0], first_img)

    def get_cmd(self, i, weight) -> str:
        if i == 0:
            warped_img = self.stylized_imgs[0]
        else:
            prev_img = cv2.imread(self.stylized_imgs[i - 1])
            warped_img = flow_calc.warp(prev_img, self.flows[i - 1])
            cv2.imwrite(self.imgs[i], warped_img)

        return super().get_cmd(i, weight)
