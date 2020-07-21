#!/usr/bin/env python3

'''
Author: Julio C. Olaya
Data: April 17, 2020
'''

# imports
import matplotlib.pyplot as plt
import argparse
import torch
import time
import json
import random
import os
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image
from get_input_args import get_input_args_predict
import utility as u

# Predict a class
def predict_helper(image_path, model, topk=5):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.cuda()
    model.eval()
    image = u.process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(dev)

    logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    p = top_p[:topk,:].tolist()[0]
    c = top_class[:topk,:].tolist()[0]

    return p, c

def predict(in_args):
    img_path = in_args.input
    checkpoint = in_args.checkpoint
    cat_to_name_file = in_args.category_names
    top_k = in_args.top_k
   
    # Load model
    model = u.load_checkpoint(checkpoint)

    # Predict classes
    prob, classes = predict_helper(img_path, model, top_k)

    # Display results
    u.display_results(model.class_to_idx, prob, classes, cat_to_name_file, img_path, top_k)
    
# Main functions
def main():
    # read initial inputs
    in_args = get_input_args_predict()
    
    # train neural network
    predict(in_args)

# Call to main function to run the program
if __name__ == "__main__":
     main()