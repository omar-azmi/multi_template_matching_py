from typing import List, Tuple, TypeAlias
from pathlib import Path
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
import time

# assert("pyproject.toml" in os.listdir(os.getcwd()))

IntervalBound: TypeAlias = List[Tuple[float, float]]
ResolvableBound: TypeAlias = None | int | float | List[int] | List[float] | IntervalBound


def resolveLowerUpperBound(
	bounds: ResolvableBound,
	bounds_length: None | int = None
) -> IntervalBound:
	"""performs the following conversions to your input `bounds`, to get a consistient output: \n
	- `None, bounds_length` -> `[(0, 0)] * bounds_length`
	- `[x, y, z, ...]` -> `[(x / 2, x / 2), (y / 2, y / 2), (z / 2, z / 2), ...]`
	- `[(x_l, x_u), (y_l, y_u), ...]` -> `[(x_l, x_u), (y_l, y_u), ...]`

	:param bounds: sequence of bounds
	:param bounds_length: length of bounds to generate if `bounds` is `None`
	:return: list of 2-tuples describing (lower, upper) bounds
	"""
	if bounds is None:
		bounds = 0
	if bounds_length is None:
		bounds_length = 1
	if isinstance(bounds, (int, float)):
		bounds = [bounds] * bounds_length
	if isinstance(bounds[0], (int, float)):
		bounds = [(x / 2, x / 2) for x in bounds]
	assert (isinstance(bounds[0], (list, tuple)) and isinstance(bounds[0][0], (int, float)))
	return bounds


def maskOutBackgroundColor(
	img: ndarray,
	bg_color: int | float | list[int] | list[float] | ndarray[int | float],
	tolerance: int | float | list[int] | list[float] | ndarray[int | float] | None = None,
) -> ndarray[float]:
	"""construct a mask that masks out the background color of the provided image array

	:param img: single channel image array of shape (H, W), \
		or a multichannel image array of shape (H, W, Ch)
	:param bg_color: specify the background color of your image \
		- if `img.shape == (H, W)`, then `bg_color` must of single numeric type \
		- if `img.shape == (H, W, Ch)`, then `bg_color` must of numeric tuple type of length Ch. example: `[255, 255, 255]`
	:param tolerance: specify either a symmetric tolerance range of each channel, \
		or a 2-tuple (lower_tolerance, upper_tolerance) of each channel. \
		two examples for setting equivalent tolerance of a channel RGB image: \
			```py
			tolerance_symmetric = [25, 25, 10]
			tolerance_lower_upper = [(12.5, 12.5), (12.5, 12.5), (5, 5)]
			```
	:return: mask of shape (H, W) containg the weights of each pixel. 0.0 implies no weight (background colored pixel), 1.0 implies max weight (non-background pixel)
	"""
	if img.ndim == 2:
		img = img[:, :, np.newaxis]
	h, w, channels = img.shape
	if isinstance(bg_color, (int, float)):
		bg_color = [bg_color]
	tolerance = resolveLowerUpperBound(tolerance, channels)
	assert (len(bg_color) == channels and len(tolerance) == channels)
	mask_out = np.ones((h, w), dtype=float)
	for c in range(0, channels):
		channel_color = bg_color[c]
		lower_tolerance, upper_tolerance = tolerance[c]
		mask_out *= np.logical_and(
			img[:, :, c] > channel_color - lower_tolerance,
			img[:, :, c] < channel_color + upper_tolerance,
		)
	return np.logical_not(mask_out)


def phaseCorrelationWithFT(scan_img_ft: ndarray[np.complex128], template_img: ndarray[int | float], template_img_mask: None | ndarray[int | float] = None):
	sh, sw, channels = scan_img_ft.shape[0], scan_img_ft.shape[1], 0
	if scan_img_ft.ndim == 3:
		channels = scan_img_ft.shape[2]
	if template_img_mask is not None:
		template_img = template_img.copy()
		if channels == 0:
			template_img *= template_img_mask
		else:
			for c in range(0, channels):
				template_img[:, :, c] *= template_img_mask
	template_img_ft = np.fft.fft2(template_img, (sh, sw), axes=(0, 1))
	correlation_ft = scan_img_ft * np.ma.conjugate(template_img_ft)
	if channels == 0:
		correlation_ft /= np.abs(correlation_ft)
	else:
		for c in range(0, channels):
			correlation_ft[:, :, c] /= np.abs(correlation_ft[:, :, c])
	correlation = np.fft.ifft2(correlation_ft, axes=(0, 1)).real
	return correlation


def phaseCorrelation(scan_img: ndarray[int | float], template_img: ndarray[int | float], template_img_mask: None | ndarray[int | float] = None):
	scan_img_ft = np.fft.fft2(scan_img, axes=(0, 1))
	return phaseCorrelationWithFT(scan_img_ft, template_img, template_img_mask)
