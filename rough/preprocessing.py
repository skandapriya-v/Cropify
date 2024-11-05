from PIL import Image
import torchvision.transforms as TF
import sys
sys.path.append('.')
import CNN
import torch
import numpy as np

new_img = Image.open('leaf3.jpg').convert('RGB').resize((224, 224))
img = TF.ToTensor()(new_img)
img = img.unsqueeze(0)
img = img / 255.0

img_numpy = img.squeeze(0).permute(1, 2, 0).numpy()
img_pil = Image.fromarray((img_numpy * 255).astype('uint8'))
img_pil.save('preprocessed_image.jpg')

