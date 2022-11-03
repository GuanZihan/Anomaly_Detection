import numpy as np
import cv2

mask = cv2.imread("kitty.png")
mask = cv2.resize(mask, (32,32))
np.save("kitty.npy", mask)