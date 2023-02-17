import os
from typing import Generator, Iterator, List, Tuple
import numpy as np
from numpy import ndarray
from scipy import ndimage
from PIL import Image, ImageDraw, ImageOps, ImageFont

# "pyproject.toml" in os.listdir(os.getcwd())


def getTemplateWeight(img_arr: ndarray) -> ndarray:
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


def pick_random_index_generator(arr: ndarray, arr_weights: ndarray | None) -> Iterator[Tuple[int, int]]:
	"""pick a (y, x) index at random based on the provided optional `arr_weights`

	Parameters
	----------
	arr : ndarray of shape (H, W) or (H, W, 3) or etc...
	arr_weights : ndarray of shape (H, W)
	"""
	if arr_weights is None:
		arr_weights = np.ones((arr.shape[0], arr.shape[1],), dtype=float)
	idys, idxs = np.where(arr_weights > 0)
	ids = np.arange(len(idys))
	np.random.shuffle(ids)
	# for y, x in [(112, 62), (55, 55), (44, 94), (130, 80), (14, 62)]:
	# colors:
	# [(77, 77, 79), (87, 63, 53), (36, 36, 38), (62, 53, 58), (11, 92, 75)]
	# yield (y, x,)
	for i in ids:
		yield (idys[i], idxs[i])


def findTemplateIn(scan_img_arr: ndarray, template_img_arr: ndarray, template_weight_arr: ndarray | None = None, scan_rect: Tuple[int, int, int, int] | List[int] | None = None) -> Tuple[int, int, int, int]:
	# default parameters
	color_diff_uncertainty = 150  # color of the pixel found in the `scan_img_arr` should be within \pm `color_diff_uncertainty / 2`
	max_scan_iterations = 10  # max number of times we will test for consistency in the pixel layout
	max_index_matches = 5000  # mask_index matches must be less than this value in order to move to individual index matching
	low_sigmoid_certainty_cutoff = 0.5  # pixel colors with sigmoid_certainty less than this value will be considered non-matches

	if scan_rect is None:
		scan_rect = (0, 0, scan_img_arr.shape[0], scan_img_arr.shape[1],)
	# `bx` = [y0, x0, y1, x1] rect-coords of the scan region
	bx = (scan_rect[0], scan_rect[1], scan_rect[0] + scan_rect[2], scan_rect[1] + scan_rect[3],)
	bx = (
		max(bx[0], 0),
		max(bx[1], 0),
		min(bx[2], scan_img_arr.shape[0] - template_img_arr.shape[0]),
		min(bx[3], scan_img_arr.shape[1] - template_img_arr.shape[1]),
	)
	scan_img_arr = np.pad(scan_img_arr[bx[0]:bx[2], bx[1]:bx[3]], [(0, template_img_arr.shape[0],), (0, template_img_arr.shape[1],), (0, 0,)])

	def sigmoid_certainty_func(v: float | ndarray[float]) -> float | ndarray[float]:
		# this sigmoid function return almost 1.0 when `v` is less than `color_diff_uncertainty / 2`, otherwise its output is almost zero when `v` is greater
		# paste the following expression in desmos to see a graph of this function: "\frac{1}{1+\exp\left(x-\frac{\sigma}{2}\right)}"
		return 1 / (1 + np.exp(v - color_diff_uncertainty / 2))

	match_mask: ndarray[float] = np.ones((scan_img_arr.shape[0] - template_img_arr.shape[0], scan_img_arr.shape[1] - template_img_arr.shape[1],), dtype=float)

	def mask_matching_pixels_colors(y: int, x: int) -> ndarray[float]:
		template_px_color = template_img_arr[y, x]
		# subtract the color of the random pixel from all pixels of `scan_img_arr` and then compare which pixels have a matching (+ uncertainity) color
		color_diff_arr = np.sum(np.abs(scan_img_arr - template_px_color), axis=2)
		rect = [y, x, y + match_mask.shape[0], x + match_mask.shape[1]]
		np.multiply(
			match_mask,
			sigmoid_certainty_func(color_diff_arr[rect[0]:rect[2], rect[1]:rect[3]]),
			match_mask,
		)
		return match_mask

	def find_matching_indexes(y: int, x: int, prev_matches_ys: ndarray[int], prev_matches_xs: ndarray[int]) -> Tuple[ndarray[int], ndarray[int]]:
		template_px_color = template_img_arr[y, x]
		new_matches_ys = []
		new_matches_xs = []
		matches_len = len(prev_matches_ys)
		for i in range(matches_len):
			scan_px_y, scan_px_x = prev_matches_ys[i] + y, prev_matches_xs[i] + x
			color_diff = sum(abs(scan_img_arr[scan_px_y, scan_px_x] - template_px_color))
			if sigmoid_certainty_func(color_diff) > low_sigmoid_certainty_cutoff:
				new_matches_ys += [scan_px_y]
				new_matches_xs += [scan_px_x]
		matches_ys = np.array(new_matches_ys, dtype=int) - y
		matches_xs = np.array(new_matches_xs, dtype=int) - x
		return (matches_ys, matches_xs,)

	matches_ys: ndarray[int] = None
	matches_xs: ndarray[int] = None
	matches_len = match_mask.shape[0] * match_mask.shape[1]
	pick_yx = pick_random_index_generator(template_img_arr, template_weight_arr)
	for y, x in pick_yx:
		if matches_len > max_index_matches:
			# perform a mask match
			mask = mask_matching_pixels_colors(y, x) > low_sigmoid_certainty_cutoff
			matches_len = np.sum(mask)
			if matches_len <= max_index_matches:
				matches_ys, matches_xs = np.where(mask)
		else:
			matches_ys, matches_xs = find_matching_indexes(y, x, matches_ys, matches_xs)
			matches_len = len(matches_ys)
			if matches_len == 1:
				return (matches_ys[0] + bx[0], matches_xs[0] + bx[1], template_img_arr.shape[0], template_img_arr.shape[1],)
			if len(matches_ys) == 0:
				return Exception("zero matches found")
		print(y, x, matches_len)
		max_scan_iterations -= 1

		"""
		template_px_color = template_img_arr[y, x]
		if first_time:
			# subtract the color of the random pixel from all pixels of `scan_img_arr` (within the search bounds and with the correct offset) and then compare which pixels have a matching (+ uncertainity) color
			color_diff_arr = np.sum(np.abs(scan_img_arr - template_px_color), axis=2)
			matches_ys, matches_xs = np.where(color_diff_arr < color_diff_uncertainty)
			if len(matches_ys) > max_index_matches:
				return Exception("too many matches found")
			first_time = False
		else:
			new_matches_ys, new_matches_xs = [], []
			for i in range(len(matches_ys)):
				scan_px_y, scan_px_x = matches_ys[i] + y, matches_xs[i] + x
				color_diff = sum(abs(scan_img_arr[scan_px_y, scan_px_x] - template_px_color))
				if color_diff < color_diff_uncertainty:
					new_matches_ys += [scan_px_y]
					new_matches_xs += [scan_px_x]
			matches_ys = np.array(new_matches_ys, dtype=int)
			matches_xs = np.array(new_matches_xs, dtype=int)
		# set the coordinates of the matches so that they reference the top-left pixel of their own template's location rect
		matches_ys -= y
		matches_xs -= x
		max_scan_iterations -= 1
		print(y, x, template_px_color, len(matches_ys))
		if len(matches_ys) == 1:
			return (matches_ys[0] + bx[0], matches_xs[0] + bx[1], template_img_arr.shape[0], template_img_arr.shape[1],)
		if len(matches_ys) == 0:
			return Exception("zero matches found")
		if max_scan_iterations <= 0:
			return Exception("max scan iterations reached")
		"""


template_img_path = "./assets/cutouts/0020_0010/1.jpg"
scan_img_path = "./assets/pages/0020_0010/242.jpg"

#template_img_path = "./temp/45.jpg"
#scan_img_path = "./assets/pages/0020_0010/242.jpg"

template_img_arr = np.asarray(Image.open(template_img_path).convert("RGB"))
scan_img_arr = np.asarray(Image.open(scan_img_path).convert("RGB"))

template_rect = findTemplateIn(scan_img_arr, template_img_arr, ndimage.binary_erosion(getTemplateWeight(template_img_arr), np.ones((9, 9))))
print(template_rect)
if isinstance(template_rect, list, tuple):
	y, x, w, h = template_rect
	cropped_img = Image.fromarray(scan_img_arr[y:y + h, x:x + w])
	cropped_img.show()

"""
a = Image.fromarray(getTemplateWeight(template_img_arr))
b = Image.fromarray(ndimage.binary_erosion(getTemplateWeight(template_img_arr), np.ones((9, 9))))
# a.show()
# b.show()
c = Image.new("RGB", (a.width, a.height), "black")
c.paste(
	Image.open(template_img_path).convert("RGB"),
	mask=b
)
c.show()
"""
