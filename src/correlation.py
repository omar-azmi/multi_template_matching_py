from typing import Optional

import numpy as np
from numpy import ndarray

# assert("pyproject.toml" in os.listdir(os.getcwd()))


def phaseCorrelationWithFT(scan_img_ft: ndarray[np.complex128], template_img: ndarray[int | float], template_img_mask: Optional[ndarray[int | float]] = None):
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


def phaseCorrelation(scan_img: ndarray[int | float], template_img: ndarray[int | float], template_img_mask: Optional[ndarray[int | float]] = None):
	scan_img_ft = np.fft.fft2(scan_img, axes=(0, 1))
	return phaseCorrelationWithFT(scan_img_ft, template_img, template_img_mask)
