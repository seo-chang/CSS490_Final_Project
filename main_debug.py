import os
import torchvision as vision

# Use pretrained model for comparison
model = vision.models.wide_resnet50_2(pretrained=True)

