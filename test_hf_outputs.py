import urllib.request
from PIL import Image
from transformers import pipeline
import sys

urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Clouds_over_the_Atlantic_Ocean.jpg/800px-Clouds_over_the_Atlantic_Ocean.jpg', 'landscape.jpg')
img = Image.open('landscape.jpg')

pipe = pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-v2-Model")
res = pipe(img)
print("prithivMLmods:", res)

pipe2 = pipeline('image-classification', model="dima806/deepfake_vs_real_image_detection")
res2 = pipe2(img)
print("dima806:", res2)
