import cv2
import os
import torch.nn as nn
import torch
import numpy as np
from PIL import Image


"""
转置卷积：nn.ConvTranspose2d
    一般用于上采样
"""
img = Image.open('../assets/10.jpg')
img = np.array(img)
img = cv2.resize(img, (224, 224))

img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
print(max(img_tensor))

img_tensor = img_tensor / 255.
print(img_tensor.shape)

conv = nn.Conv2d(3, 32, 16, 16)
out_tensor = conv(img_tensor)
print(out_tensor.shape)

unconv = nn.ConvTranspose2d(32, 3, 16, 16)
restore_tensor = unconv(out_tensor)
print(restore_tensor.shape)

restore_image = restore_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
# cv2.imwrite("./restore_image.jpg", restore_image*255)
restore_image = restore_image * 255
restore_image = restore_image.astype(np.uint8)
# restore_image.save("./restore_image.jpg")

image = Image.fromarray(restore_image)
image.save("./restore_image.jpg")
