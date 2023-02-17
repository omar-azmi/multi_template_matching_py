import math
from typing import Iterable, List, Tuple, TypeAlias, Optional
from pathlib import Path
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
from cv2 import Mat
import time
from src.correlation import phaseCorrelation
from src.tools import maskOutBackgroundColor, constructPower2Shapes
from src.mipmap import constructMipmap, constructMipmapPower2, Mipmap

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


scan_img_path = "./assets/base.jpg"
template_path = "./assets/templates/45.jpg"

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
