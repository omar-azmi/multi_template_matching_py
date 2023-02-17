from typing import List, Tuple
from pathlib import Path
import time
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
from src.correlation import phaseCorrelationWithFT
from src.tools import maskOutBackgroundColor

# assert("pyproject.toml" in os.listdir(os.getcwd()))


def constructTemplateWeight(img_arr: ndarray) -> ndarray:
	"""get the weights of an image template such that near-pure white pixels get a zero weight, while others get 1.0

	Parameters
	----------
	img_arr : ndarray of shape (H, W, 3)

	Returns
	-------
	ndarray of shape (H, W) contains the alpha weights of each pixel. 0.0 implies no weight, 1.0 implies max weight
	"""
	max_whiteness = (255 - 15) * 3
	weights = np.sum(img_arr, axis=2) < max_whiteness
	return weights


def findTemplateIn(scan_img_path: Path, template_img_path: Path, scan_rect: Tuple[int, int, int, int] | List[int] | None = None) -> Tuple[int, int, int, int]:
	# function parameters
	min_correlation_threshhold = 0.95
	match_crop_pad = [5, 5, 5, 5]

	scan_img = cv2.imread(scan_img_path, cv2.IMREAD_COLOR)
	template_img = cv2.imread(template_img_path, cv2.IMREAD_COLOR)
	template_img_mask = (ndimage.binary_erosion(constructTemplateWeight(np.asarray(template_img)), np.ones((5, 5))) * 255.0).astype(np.uint8)
	perf_time0 = time.time()
	# do masked template matching and get the correlation image
	correlation_img = cv2.matchTemplate(scan_img, template_img, cv2.TM_CCORR_NORMED, mask=template_img_mask)
	perf_time1 = time.time()
	print(f"cv2 correlation image computation time:\n\t{perf_time1 - perf_time0} seconds")
	max_val = 1
	match_rects = []
	h, w = template_img.shape[0], template_img.shape[1]
	while max_val > min_correlation_threshhold:
		# find max value of correlation image
		max_yx = np.unravel_index(correlation_img.argmax(), correlation_img.shape)
		max_val = correlation_img[max_yx]
		if max_val > min_correlation_threshhold:
			y, x = max_yx
			match_rects += [(y, x, h, w),]
			# mask out this match to remove redundant almost-close matches
			y0 = max(y - match_crop_pad[0], 0)
			x0 = max(x - match_crop_pad[1], 0)
			y1 = min(y + h + match_crop_pad[0], correlation_img.shape[0])
			x1 = min(x + w + match_crop_pad[1], correlation_img.shape[1])
			correlation_img[y0:y1, x0:x1] = 0
	return match_rects


scan_img_path = "./assets/base_small.jpg"
template_img_path = "./assets/templates/45.jpg"

match_rects = findTemplateIn(scan_img_path, template_img_path)
print(f"matching rectangle coordinates:\n\t{match_rects}")
for y, x, h, w in match_rects:
	match_img = cv2.imread(scan_img_path, cv2.IMREAD_UNCHANGED)[y:y + h, x:x + w]
	cv2.imshow("cv2 match", match_img)
	cv2.waitKey()

scan_img = np.asarray(cv2.imread(scan_img_path, cv2.IMREAD_COLOR))
template_img = np.asarray(cv2.imread(template_img_path, cv2.IMREAD_COLOR))
template_img_mask = maskOutBackgroundColor(template_img, [255, 255, 255], [30, 30, 30])
scan_img_ft = np.fft.fft2(scan_img, axes=(0, 1))
perf_time0 = time.time()
correlation_img = phaseCorrelationWithFT(scan_img_ft, template_img, template_img_mask)
perf_time1 = time.time()
if correlation_img.ndim == 3:
	correlation_img = np.sum(correlation_img, axis=2) / correlation_img.shape[2]
y, x = np.unravel_index(correlation_img.argmax(), correlation_img.shape)
h, w = template_img.shape[0], template_img.shape[1]
print(f"phaseCorrelationWithFT computation time:\n\t{perf_time1 - perf_time0} seconds")
print(f"matching rectangle coordinates:\n\t{[(y, x, h, w)]}")
cv2.imshow("phaseCorrelationWithFT match", scan_img[y:y + h, x:x + w])
cv2.waitKey()
