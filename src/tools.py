from typing import Iterator, List, Literal, Optional, Tuple, TypeAlias

import numpy as np
from cv2 import Mat, boundingRect, connectedComponents
from numpy import ndarray

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


def uniqueElementsInOrder(arr: ndarray[int] | Mat, axis: int = 0, order: Literal[1, -1] = 1, ignore_values: List[int] = []) -> List[int]:
	# make `axis` the first axis of `arr` by circulating/rolling axis as needed
	arr = np.rollaxis(arr, axis, 0)
	uniques_along_axis = np.concatenate([np.unique(row) for row in arr])
	if order == -1:
		# we wish to traverse from end of axis to its begining (think left-to-right, or bottom-to-top traversal)
		uniques_along_axis = np.flip(uniques_along_axis)
	ordered_uniques_dict = {v: True for v in uniques_along_axis}
	for val in ignore_values:
		if val in ordered_uniques_dict:
			del ordered_uniques_dict[val]
	ordered_uniques = list(ordered_uniques_dict.keys())
	return ordered_uniques


def imageComponents(binary_image: ndarray[bool | np.uint8] | Mat, order: Optional[Literal["y+", "y-", "x+", "x-"]] = None) -> Iterator[Tuple[int, Tuple[int, int, int, int], ndarray[np.uint8]]]:
	labeled_image: ndarray[int]
	num_labels, labeled_image = connectedComponents(binary_image)
	labels = list(range(1, num_labels))
	if order is not None:
		axis = 0 if order[0] == "y" else 1
		direction = 1 if order[1] == "+" else -1
		labels = uniqueElementsInOrder(labeled_image, axis=axis, order=direction, ignore_values=[0,])
	for i in labels:
		label_i_image = (labeled_image == i).astype(np.uint8, copy=False)
		x, y, w, h = boundingRect(label_i_image)  # as `Tuple[int, int, int, int]`
		# we yield a copy of the sliced `label_i_image`, because otherwise, the entirity of the bigger
		# `label_i_image` will remain in-memory without its unnecessary parts being garbage collected
		yield (i, (y, x, h, w), label_i_image[y:y + h, x:x + w].copy())
