import cv2
import numpy as np


def histogram_transform(img: np.ndarray, means: np.ndarray, stds: np.ndarray,
                        target_means: np.ndarray, target_stds: np.ndarray):
    means = means.reshape((1, 1, 3))
    stds = stds.reshape((1, 1, 3))
    target_means = target_means.reshape((1, 1, 3))
    target_stds = target_stds.reshape((1, 1, 3))
    x = img.astype(np.float32)
    x = (x - means) * target_stds / stds + target_means
    # x = np.round(x)
    # x = np.clip(x, 0, 255)
    # x = x.astype(np.uint8)
    return x


def blend(a: np.ndarray,
          b: np.ndarray,
          min_error: np.ndarray,
          weight1=0.5,
          weight2=0.5):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2Lab)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2Lab)
    min_error = cv2.cvtColor(min_error, cv2.COLOR_BGR2Lab)
    a_mean = np.mean(a, axis=(0, 1))
    a_std = np.std(a, axis=(0, 1))
    b_mean = np.mean(b, axis=(0, 1))
    b_std = np.std(b, axis=(0, 1))
    min_error_mean = np.mean(min_error, axis=(0, 1))
    min_error_std = np.std(min_error, axis=(0, 1))

    t_mean_val = 0.5 * 256
    t_std_val = (1 / 36) * 256
    t_mean = np.ones([3], dtype=np.float32) * t_mean_val
    t_std = np.ones([3], dtype=np.float32) * t_std_val
    a = histogram_transform(a, a_mean, a_std, t_mean, t_std)

    b = histogram_transform(b, b_mean, b_std, t_mean, t_std)
    ab = (a * weight1 + b * weight2 - t_mean_val) / 0.5 + t_mean_val
    ab_mean = np.mean(ab, axis=(0, 1))
    ab_std = np.std(ab, axis=(0, 1))
    ab = histogram_transform(ab, ab_mean, ab_std, min_error_mean,
                             min_error_std)
    ab = np.round(ab)
    ab = np.clip(ab, 0, 255)
    ab = ab.astype(np.uint8)
    ab = cv2.cvtColor(ab, cv2.COLOR_Lab2BGR)
    return ab
