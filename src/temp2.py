from typing import List, Tuple, TypeAlias
from pathlib import Path
import numpy as np
from numpy import ndarray
from scipy import ndimage
import cv2
import time
from src.main import maskOutBackgroundColor, phaseCorrelation

# assert("pyproject.toml" in os.listdir(os.getcwd()))
# 1d fft for template row matching
# downsample fft pyramid then progressive match tree/routines

scan_img_path = "./assets/base.jpg"
img_path = "./assets/templates/45.jpg"
resize = 1

scan_img = np.asarray(cv2.imread(scan_img_path, cv2.IMREAD_GRAYSCALE))
img = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

if False:
	img_mask = maskOutBackgroundColor(img, [255], [30])
	img_ft = np.fft.fft2(img * img_mask, axes=(0, 1))
	h, w = img.shape[0], img.shape[1]
	new_h, new_w = h, w  # int((w * resize) // 2)
	print(w // 2 - new_w, w // 2 + new_w)
	#img_ft = np.fft.ifftshift(np.fft.fftshift(img_ft, axes=(1,))[:, w // 2 - new_w:w // 2 + new_w], axes=(1,))
	# img_ft = np.concatenate((img_ft[:, 0:new_w], img_ft[:, -new_w:None]), axis=1)
	# img_downsampled = cv2.resize(np.fft.ifft2(img_ft, axes=(0, 1)).real.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
	img_downsampled = np.fft.ifft2(img_ft, axes=(0, 1)).real

	scan_img_ft = np.fft.fft2(scan_img, axes=(0, 1))
	sh, sw, channels = scan_img_ft.shape[0], scan_img_ft.shape[1], 0
	new_h, new_w = sh, sw  # int((sw * resize) // 2)
	#scan_img_ft = np.fft.ifftshift(np.fft.fftshift(scan_img_ft, axes=(1,))[:, sw // 2 - new_w:sw // 2 + new_w], axes=(1,))
	if scan_img_ft.ndim == 3:
		channels = scan_img_ft.shape[2]
	img_downsampled_ft = np.fft.fft2(img_downsampled, (new_h, new_w), axes=(0, 1))
	correlation_ft = scan_img_ft * np.ma.conjugate(img_downsampled_ft)
	if channels == 0:
		correlation_ft /= np.abs(correlation_ft)
	else:
		for c in range(0, channels):
			correlation_ft[:, :, c] /= np.abs(correlation_ft[:, :, c])
	correlation = np.fft.ifft2(correlation_ft, axes=(0, 1)).real
	correlation = cv2.resize((correlation * 255 / np.max(correlation)).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)


downscale = 6
scan_img = cv2.resize(scan_img, (scan_img.shape[1] // downscale, scan_img.shape[0] // downscale), interpolation=cv2.INTER_NEAREST)
img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale), interpolation=cv2.INTER_NEAREST)
img_mask = maskOutBackgroundColor(img, [255], [30])
scan_img = cv2.medianBlur(scan_img, 5)
img = cv2.medianBlur(img, 5)

cv2.imshow("scan_img", scan_img)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

#correlation = phaseCorrelation(scan_img, img, img_mask)
correlation = cv2.matchTemplate(scan_img, img, cv2.TM_CCORR_NORMED, mask=(img_mask * 255).astype(np.uint8))
y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
y *= downscale
x *= downscale
h, w = img.shape[0] * downscale, img.shape[1] * downscale
print(y, x, np.max(correlation))
scan_img = np.asarray(cv2.imread(scan_img_path, cv2.IMREAD_UNCHANGED))
#correlation[y:y + h, x:x + w] = np.max(correlation)

cv2.imshow("match", scan_img[y:y + h, x:x + w])
cv2.waitKey()
cv2.destroyAllWindows()
