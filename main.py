import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from stn.resnet import ResNet18, ResNet34
from stn.stn_network import SpatialTransformerNetwork
from stn.stn_ocr import StnOcr

def stnOcrModel():
    num_steps = 1
    detection_filter = [32, 48, 48]
    recognition_filter = [32, 64, 128]
    
    # Instantiate ResNet-based models
    resnet_model = ResNet18(num_classes=10)
    
    # Instantiate STN-OCR model
    stn_detection = StnOcr(input_shape=(1, 600, 150), nb_classes=10,
                            detection_filter=detection_filter,
                            recognition_filter=recognition_filter)
    
    # Initialize the flag for the detection phase
    flag = 'detection'
    
    # Generate theta using the detection phase
    theta = stn_detection.resnetDetRec(flag)
    inp = stn_detection.input
    
    # Instantiate Spatial Transformer Network
    stn_obj = SpatialTransformerNetwork(input=inp, theta=theta,
                                         num_steps=num_steps)
    
    # Generate sampled image from the grid generator
    sampled_image = stn_obj.image_sampling()
    
    # Switch the flag to recognition mode
    flag = 'recognition'
    
    # Obtain the final output of the recognition model
    out = stn_detection.resnetDetRec(sampled_image, flag)
    
    # Combine all the components into a single model
    stn_model = nn.Sequential(resnet_model, stn_obj, out)
    print(stn_model)

if __name__ == '__main__':
    stnOcrModel()
