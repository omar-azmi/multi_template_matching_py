import math
from typing import List, Optional, Tuple

import cv2
from cv2 import Mat
import numpy as np
from numpy import ndarray
from skimage.feature.peak import peak_local_max

from src.mipmap import Mipmap


# assert("pyproject.toml" in os.listdir(os.getcwd()))


def matchTemplate(image: ndarray | Mat, template: ndarray | Mat, method: int = cv2.TM_CCORR_NORMED, mask: Optional[ndarray | Mat] = None, rect: Optional[Tuple[int, int, int | None, int | None]] = [0, 0, None, None], min_correlation=0.95) -> List[Tuple[int, int, float]]:
	if rect is None:
		rect = [0, 0, None, None]
	h, w = template.shape[0], template.shape[1]
	correlation = cv2.matchTemplate(image[rect[0]:rect[2], rect[1]:rect[3]], template, method, mask=mask)
	peaks_yx = peak_local_max(correlation, min_distance=min(h // 2, w // 2), threshold_abs=min_correlation, exclude_border=True)
	# peaks_yx is sorted in descending order by default
	peaks_yx[:,0] += rect[0]
	peaks_yx[:,1] += rect[1]
	peaks_yxv = []
	for y, x in peaks_yx:
		val = correlation[y, x]
		peaks_yxv += [(y, x, val),]
	return peaks_yxv


def matchTemplateOfMipmaps(image_mipmaps: Mipmap, template_mipmaps: Mipmap, method: int = cv2.TM_CCORR_NORMED, mask_mipmaps: Optional[Mipmap] = None, min_correlation=0.95) -> Tuple[int, int]:
	m = len(image_mipmaps)
	if mask_mipmaps is None:
		mask_mipmaps = [None] * m
	assert (len(template_mipmaps) == m and len(mask_mipmaps) == m)
	rect = [0, 0, None, None]
	y, x = 0, 0
	for i in reversed(range(0, m)):
		img, templ, mask = image_mipmaps[i], template_mipmaps[i], mask_mipmaps[i]
		correlation = cv2.matchTemplate(img[rect[0]:rect[2], rect[1]:rect[3]], templ, method, mask=mask)
		y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
		val = correlation[y, x]
		y += rect[0]
		x += rect[1]
		if i == 0:
			break
		if val < min_correlation:
			# we will discard the current match result (y, x) and not narrow down our `rect` at this moment
			rect = [0, 0, None, None]
			y, x = 0, 0
			continue
		next_img = image_mipmaps[i - 1]
		th, tw = templ.shape[0], templ.shape[1]
		sy, sx = math.ceil(next_img.shape[0] / img.shape[0]), math.ceil(next_img.shape[1] / img.shape[1])
		y0 = max((y - 1) * sy, 0)
		x0 = max((x - 1) * sx, 0)
		y1 = min((y + th + 1) * sy, next_img.shape[0])
		x1 = min((x + tw + 1) * sx, next_img.shape[1])
		rect = [y0, x0, y1, x1]
	return int(y), int(x)
