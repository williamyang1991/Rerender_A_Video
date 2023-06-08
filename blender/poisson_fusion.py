import cv2
import numpy as np
import scipy

As = None
prev_states = None


def construct_A(h, w, grad_weight):
    indgx_x = []
    indgx_y = []
    indgy_x = []
    indgy_y = []
    vdx = []
    vdy = []
    for i in range(h):
        for j in range(w):
            if i < h - 1:
                indgx_x += [i * w + j]
                indgx_y += [i * w + j]
                vdx += [1]
                indgx_x += [i * w + j]
                indgx_y += [(i + 1) * w + j]
                vdx += [-1]
            if j < w - 1:
                indgy_x += [i * w + j]
                indgy_y += [i * w + j]
                vdy += [1]
                indgy_x += [i * w + j]
                indgy_y += [i * w + j + 1]
                vdy += [-1]
    Ix = scipy.sparse.coo_array(
        (np.ones(h * w), (np.arange(h * w), np.arange(h * w))),
        shape=(h * w, h * w)).tocsc()
    Gx = scipy.sparse.coo_array(
        (np.array(vdx), (np.array(indgx_x), np.array(indgx_y))),
        shape=(h * w, h * w)).tocsc()
    Gy = scipy.sparse.coo_array(
        (np.array(vdy), (np.array(indgy_x), np.array(indgy_y))),
        shape=(h * w, h * w)).tocsc()
    As = []
    for i in range(3):
        As += [
            scipy.sparse.vstack([Gx * grad_weight[i], Gy * grad_weight[i], Ix])
        ]
    return As


# blendI, I1, I2, mask should be RGB unit8 type
# return poissson fusion result (RGB unit8 type)
# I1 and I2: propagated results from previous and subsequent key frames
# mask: pixel selection mask
# blendI: contrastive-preserving blending results of I1 and I2
def poisson_fusion(blendI, I1, I2, mask, grad_weight=[2.5, 0.5, 0.5]):
    global As
    global prev_states

    Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(float)
    Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(float)
    Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(float)
    m = (mask > 0).astype(float)[:, :, np.newaxis]
    h, w, c = Iab.shape

    # fuse the gradient of I1 and I2 with mask
    gx = np.zeros_like(Ia)
    gy = np.zeros_like(Ia)
    gx[:-1, :, :] = (Ia[:-1, :, :] - Ia[1:, :, :]) * (1 - m[:-1, :, :]) + (
        Ib[:-1, :, :] - Ib[1:, :, :]) * m[:-1, :, :]
    gy[:, :-1, :] = (Ia[:, :-1, :] - Ia[:, 1:, :]) * (1 - m[:, :-1, :]) + (
        Ib[:, :-1, :] - Ib[:, 1:, :]) * m[:, :-1, :]

    # construct A for solving Ax=b
    crt_states = (h, w, grad_weight)
    if As is None or crt_states != prev_states:
        As = construct_A(*crt_states)
        prev_states = crt_states

    final = []
    for i in range(3):
        weight = grad_weight[i]
        im_dx = np.clip(gx[:, :, i].reshape(h * w, 1), -100, 100)
        im_dy = np.clip(gy[:, :, i].reshape(h * w, 1), -100, 100)
        im = Iab[:, :, i].reshape(h * w, 1)
        im_mean = im.mean()
        im = im - im_mean
        A = As[i]
        b = np.vstack([im_dx * weight, im_dy * weight, im])
        out = scipy.sparse.linalg.lsqr(A, b)
        out_im = (out[0] + im_mean).reshape(h, w, 1)
        final += [out_im]

    final = np.clip(np.concatenate(final, axis=2), 0, 255)
    return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)
