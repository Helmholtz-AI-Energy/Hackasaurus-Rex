import torch
from PIL import Image
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

from hackasaurus_rex.data import DroneImages

drone_images = DroneImages("/hkfs/work/workspace/scratch/ih5525-energy-train-data")
x, y = drone_images[0]

output = model(x.unsqueeze(0))
