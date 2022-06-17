
import cv2
import torch
import urllib.request
from torchvision.transforms import ToTensor
import numpy as np
from django.conf import settings
from PIL import Image
import os


def super_resolve(input_url, output_url):
    # Training settings
    """model_path = settings.BASE_DIR +'/model.pth'
    print(model_path)
    the_model = torch.load(model_path)"""
    
    #img = cv2.imread(settings.BASE_DIR)
    img = cv2.imread("./"+input_url)
    #print(os.)
    print(settings.BASE_DIR)
    print(input_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    print(output_url)

    the_model = torch.hub.load("intel-isl/MiDaS", model_type)
    print("modelok!")
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    input_batch = transform(img).to("cpu")
    with torch.no_grad():
        prediction = the_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    mini=output.min()
    maxi=output.max()
    out = ((output-mini)/(maxi-mini) * 255//1)
    
    cv2.imwrite(output_url,out)
    return out