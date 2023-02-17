from typing import List, Tuple, TypeAlias

import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray

from src.tools import constructPower2Shapes


# assert("pyproject.toml" in os.listdir(os.getcwd()))


Mipmap: TypeAlias = List[ndarray | Mat]


def constructMipmapPower2(img: ndarray, min_shape: int | Tuple[int, ...] = 256, divisions: int | None = None) -> List[ndarray]:
	h, w, ch = img.shape[0], img.shape[1], 0
	if img.ndim == 3:
		ch = img.shape[2]
	mipmap_shapes = constructPower2Shapes((h, w), min_shape, divisions)
	max_h, max_w = mipmap_shapes[0]
	pad_widths = [(0, max_h - h), (0, max_w - w)]
	if ch > 0:
		pad_widths += [(0, 0),]
	img = np.pad(img, pad_widths, mode="constant", constant_values=0)
	mipmaps = [img,]
	for h, w in mipmap_shapes[1:]:
		mipmaps.append(cv2.pyrDown(mipmaps[-1], dstsize=(w, h)))
	return mipmaps


def constructMipmap(img: ndarray, min_shape: int | Tuple[int, ...] = 256, divisions: int | None = None) -> List[ndarray]:
	h, w = img.shape[0], img.shape[1]
	if isinstance(min_shape, (int, float)):
		min_shape = int(min_shape)
		min_shape = (min_shape, min_shape)
	if divisions is None:
		divisions = 100000
	mipmaps = [img,]
	while True:
		prev_img = mipmaps[-1]
		if not (prev_img.shape[0] > min_shape[0] and prev_img.shape[1] > min_shape[1] and divisions > 0):
			break
		mipmaps.append(cv2.pyrDown(prev_img))
		divisions -= 1
	return mipmaps
