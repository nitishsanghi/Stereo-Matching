import cv2
import numpy as np
import matplotlib.pyplot as plt


left_image = cv2.imread('/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_2/000000_10.png')

g1 = cv2.GaussianBlur(left_image,(5,5),1)
g2 = cv2.GaussianBlur(left_image,(5,5),10)

image = g1 - g2
plt.imshow(image)
plt.show()