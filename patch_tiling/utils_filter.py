import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.measure import block_reduce


def pooling_img(img: np.ndarray, scale_size: int = 16) -> np.ndarray:
    block_size = (img.shape[0] // scale_size, img.shape[1] // scale_size, 1)
    return block_reduce(image=img, block_size=block_size, func=np.average)


def count_purple_box(pooled: np.ndarray) -> int:
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    c1 = r > g - 10
    c2 = b > g - 10
    c3 = (r + b) / 2 > g + 20
    return pooled[c1 & c2 & c3].size


def count_black_box(pooled: np.ndarray) -> int:
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    c1 = r < 10
    c2 = g < 10
    c3 = b < 10
    return pooled[c1 & c2 & c3].size


def noise_patch(img: np.ndarray, purple_threshold: float = 0.3, black_threshold: float = 0.1) -> bool:
    if (img.shape[0] < 1) or (img.shape[1] < 1):
        return True
    pooled = pooling_img(img)
    purple_count = count_purple_box(pooled=pooled)
    black_count = count_black_box(pooled=pooled)
    total = pooled.size
    # check if black and purple colors are in the range
    c1 = black_count / total < black_threshold
    c2 = purple_count / total >= purple_threshold
    return not (c1 & c2)


def good_tile(tile):
    tile = np.array(tile)
    h, w, _ = tile.shape
    total_pixels = h * w

    # If the tile is not square
    if (h != w) or (h == 0) or (w == 0):
        return False

    # If the tile is white (or, all the pixels are same)
    if (tile == tile[0, 0, 0]).all():
        return False

    # If the green channel is more salient than the red or blue channel (normal tile is purple,
    # so the red and blue are more salient)
    red = tile[:, :, 0]
    green = tile[:, :, 1]
    blue = tile[:, :, 2]

    t1 = ((red > green) * 1).sum() / total_pixels
    t2 = ((blue > green) * 1).sum() / total_pixels

    if t1 < 0.1 and t2 < 0.1:
        if tile.std(axis=2).mean() < 5:
            return False

    # If the tile has black frames

    right_black = tile[:, -1, :].mean() < 10
    left_black = tile[:, 0, :].mean() < 10
    top_black = tile[0, :, :].mean() < 10
    bottom_black = tile[-1, :, :].mean() < 10

    black_frame = right_black or left_black or top_black or bottom_black

    if black_frame:
        return False

    # Otherwise, the tile is good
    return True






