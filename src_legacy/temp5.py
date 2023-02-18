import json
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
from src.tools import maskOutBackgroundColor, constructPower2Shapes, imageComponents
from src.mipmap import constructMipmap, constructMipmapPower2, Mipmap


img_path = "./assets/templates/61.jpg"
img = np.asarray(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
#mask = (ndimage.binary_erosion(maskOutBackgroundColor(img, [255, 255, 255], 30), np.ones((5, 5))) * 255).astype(np.uint8)
mask = (maskOutBackgroundColor(img, [255, 255, 255], 30) * 255).astype(np.uint8)
for i, (label, rect, component_mask) in enumerate(imageComponents(mask, "x+")):
	y, x, h, w = rect
	cv2.imshow(f"component: {i}, label: {label}", component_mask * 255)
cv2.waitKey()
cv2.destroyAllWindows()
