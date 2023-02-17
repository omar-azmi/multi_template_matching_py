from typing import List, Tuple, TypeAlias
from pathlib import Path
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
import time


def constructTemplateWeight(img_arr: ndarray) -> ndarray:
	"""construct the weights of an RGB image array template such that near-pure white pixels get a zero weight, while others get 1.0

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


def convolveRows(row_normalized_base: ndarray, template_row_flipped: ndarray, template_height: int, template_y: int) -> ndarray:
	h = row_normalized_base.shape[0] - template_height + 1
	w = row_normalized_base.shape[1] - template_row_flipped.shape[0] + 1
	convolved_rows = np.zeros((h, w), dtype=float)
	for y in range(0, h):
		convolved_rows[y, :] = np.convolve(row_normalized_base[template_y + y, :], template_row_flipped, mode="valid")
	return convolved_rows


def findTemplateInGray(base_img: ndarray[int | float], template_img: ndarray[int | float], template_weight: ndarray[float]):
	weighted_template_img = template_img * template_weight / (np.sum(template_weight) * np.sum(template_img))
	# `template_img` is flipped horizontally (x -> width - x), so that convolution with the flipped row strips would correspond to cross-correlation with the unflipped row
	flipped_template_img = np.flip(weighted_template_img, axis=1)
	#row_normalizing_coefficients = 1 / np.sum(base_img, axis=1)
	row_normalized_base_img = np.copy(base_img).astype(float)
	# for y in range(0, row_normalized_base_img.shape[0]):
	#	row_normalized_base_img[y] *= row_normalizing_coefficients[y]
	template_height, template_width = flipped_template_img.shape[0], flipped_template_img.shape[1]
	perf_time0 = time.time()
	y = 0
	convolved_rows = convolveRows(row_normalized_base_img, flipped_template_img[y], template_height, y)
	for y in range(1, template_height):
		print(y)
		convolved_rows += convolveRows(row_normalized_base_img, flipped_template_img[y], template_height, y)
	for y in range(0, convolved_rows.shape[0]):
		convolved_rows[y, :] /= np.convolve(row_normalized_base_img[y, :], np.ones((flipped_template_img.shape[1],), dtype=float), mode="valid")
	perf_time1 = time.time()
	print(f"execution time:\n\t{perf_time1 - perf_time0} seconds")
	return convolved_rows


def findTemplateInGray2(base_img: ndarray[int | float], template_img: ndarray[int | float], template_weight: ndarray[float]):
	template_height, template_width = template_img.shape[0], template_img.shape[1]
	base_height, base_width = base_img.shape[0], base_img.shape[1]
	weighted_template_img = template_img * template_weight / (np.sum(template_weight) * np.sum(template_img))
	normalized_base_img = np.zeros((base_height - template_height + 1, base_width - template_width + 1,), dtype=float)
	for y in range(0, base_height - template_height + 1):
		normalized_base_img[y, :] = np.convolve(base_img[y, :], np.ones((template_width,), dtype=float), mode="valid")
	# normalized_base_img *= 1 / template_width
	correlation_arr = np.zeros((base_height - template_height + 1, base_width - template_width + 1,), dtype=float)
	for top in range(0, template_height):
		template_row = weighted_template_img[top, :]
		for y in range(0, base_height - template_height + 1):
			correlation_arr[y, :] += np.correlate(base_img[top + y, :], template_row, mode="valid") / normalized_base_img[y, :]
	return correlation_arr


def computePhaseCorrelation(base_img: ndarray[int | float], template_img: ndarray[int | float]):
	base_ft = np.fft.fft2(base_img)
	perf_time0 = time.time()
	template_ft = np.fft.fft2(template_img)
	base_ft_conj = np.ma.conjugate(template_ft)
	correlation_ft = base_ft * base_ft_conj
	correlation_ft /= np.absolute(correlation_ft)
	correlation = np.fft.ifft2(correlation_ft).real
	perf_time1 = time.time()
	print(f"execution time:\n\t{perf_time1 - perf_time0} seconds")
	return correlation


scan_img_path = "./assets/base.jpg"
template_img_path = "./assets/templates/45.jpg"

scan_img = np.asarray(cv2.imread(scan_img_path, cv2.IMREAD_GRAYSCALE))
template_img = np.asarray(cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE))
template_img_mask = ndimage.binary_erosion(constructTemplateWeight(np.asarray(cv2.imread(template_img_path, cv2.IMREAD_COLOR))), np.ones((5, 5))).astype(float)

template_img_padded = np.pad(template_img * template_img_mask, [
	(0, scan_img.shape[0] - template_img.shape[0]),
	(0, scan_img.shape[1] - template_img.shape[1]),
])
img = computePhaseCorrelation(scan_img, template_img_padded)
h, w = template_img.shape[0], template_img.shape[1]
for px in np.argpartition(img, -10, axis=None)[-10:]:
	y, x = np.unravel_index(px, img.shape)
	print(y, x, img[y, x])
	cv2.imshow("match_img", scan_img[y:y + h, x:x + w])
	cv2.waitKey()

cv2.imshow("phase_correlation_img", (img * 255 / np.max(img)).astype(np.uint8))
cv2.waitKey()

input()
input()

# cv2.imshow("template_img", template_img)
# cv2.waitKey()
# cv2.imshow("template_img_mask", (template_img_mask * 255).astype(np.uint8))
# cv2.waitKey()

c = findTemplateInGray2(scan_img, template_img, template_img_mask)
c *= 255 / np.max(c)
y, x = np.unravel_index(np.argmax(c), c.shape)
h, w = template_img_mask.shape[0], template_img_mask.shape[1]
print(y, x, np.max(c))

cv2.imshow("scan_img", scan_img[y:y + h, x:x + w].astype(np.uint8))
cv2.imshow("correlation_img", c.astype(np.uint8))
cv2.waitKey()
