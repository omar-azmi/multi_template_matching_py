import math
from typing import Iterable, List, Tuple, TypeAlias, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
from cv2 import Mat
import time
from src.main import maskOutBackgroundColor, phaseCorrelation

Mipmap: TypeAlias = List[ndarray | Mat]


def constructPower2Shapes(shape: Tuple[int, ...], min_shape: int | Tuple[int, ...] = 1, divisions: int | None = None) -> List[Tuple[int, int]]:
	dims = len(shape)
	if isinstance(min_shape, (int, float)):
		min_shape = int(min_shape)
		min_shape = tuple(min_shape for d in range(dims))
	max_shape = list(min_shape)
	if divisions is None:
		divisions = 100000
	for d in range(dims):
		while max_shape[d] < shape[d]: max_shape[d] *= 2
	shapes_power2 = []
	while all([max_shape[d] >= min_shape[d] for d in range(dims)]) and divisions > 0:
		shapes_power2 += [tuple(max_shape),]
		max_shape = [s // 2 for s in max_shape]
		divisions -= 1
	return shapes_power2


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


def matchTemplateOfMipmaps(image_mipmaps: Mipmap, template_mipmaps: Mipmap, method: int = cv2.TM_CCORR_NORMED, mask_mipmaps: Optional[Mipmap] = None, min_correlation=0.95) -> Tuple[int, int]:
	m = len(image_mipmaps)
	if mask_mipmaps is None:
		mask_mipmaps = [None] * m
	assert len(template_mipmaps) == m and len(mask_mipmaps) == m
	rect = [0, 0, None, None]
	y, x = 0, 0
	for i in reversed(range(0, m)):
		img, templ, mask = image_mipmaps[i], template_mipmaps[i], mask_mipmaps[i]
		correlation = cv2.matchTemplate(img[rect[0]:rect[2], rect[1]:rect[3]], templ, method, mask=mask)
		y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
		val = correlation[y, x]
		y += rect[0]
		x += rect[1]
		if val < min_correlation or i == 0:
			# we will discard the current match result (y, x) and not narrow down our `rect` at this moment
			continue
		next_img = image_mipmaps[i - 1]
		th, tw = templ.shape[0], templ.shape[1]
		sy, sx = math.ceil(next_img.shape[0] / img.shape[0]), math.ceil(next_img.shape[1] / img.shape[1])
		y0 = max((y - 1) * sy, 0)
		x0 = max((x - 1) * sx, 0)
		y1 = min((y + th + 1) * sy, next_img.shape[0])
		x1 = min((x + tw + 1) * sx, next_img.shape[1])
		rect = [y0, x0, y1, x1]
	return (y, x)


scan_img_path = "./benchmark/base.jpg"
template_path = "./benchmark/template.jpg"

scan_img = np.asarray(cv2.imread(scan_img_path, cv2.IMREAD_GRAYSCALE))
template_img = np.asarray(cv2.imread(template_path, cv2.IMREAD_GRAYSCALE))
template_img_mask = (maskOutBackgroundColor(template_img, [255], [30]) * 255).astype(np.uint8)

scan_mipmaps = constructMipmap(scan_img, 256)
template_mipmaps = constructMipmap(template_img, min_shape=1, divisions=len(scan_mipmaps) - 1)
template_mask_mipmaps = constructMipmap(template_img_mask, min_shape=1, divisions=len(scan_mipmaps) - 1)

if False:
	for i, img in enumerate(scan_mipmaps):
		cv2.imshow(f"mipmap {i}", img)
	cv2.waitKey()
	cv2.destroyAllWindows()

	for i, img in enumerate(template_mipmaps):
		cv2.imshow(f"mipmap {i}", img)
	cv2.waitKey()
	cv2.destroyAllWindows()

t0 = time.time()
y, x = matchTemplateOfMipmaps(scan_mipmaps, template_mipmaps, mask_mipmaps=template_mask_mipmaps)
t1 = time.time()
print(f"time={t1-t0} sec\t(y,x)=({y},{x})")

# for scan, template, mask in zip(scan_mipmaps, template_mipmaps, template_mask_mipmaps):
for i in reversed(range(len(scan_mipmaps))):
	scan, template, mask = scan_mipmaps[i], template_mipmaps[i], template_mask_mipmaps[i]
	t0 = time.time()
	correlation = cv2.matchTemplate(scan, template, cv2.TM_CCORR_NORMED, mask=mask)
	t1 = time.time()
	y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
	h, w = template.shape[0], template.shape[1]
	print(f"time={t1-t0} sec\t(y,x)=({y},{x})\tcorrelation={np.max(correlation)}")
	cv2.imshow("match", scan[y:y + h, x:x + w])
	cv2.waitKey()
	cv2.destroyAllWindows()
