#!/usr/bin/env python3

'''
Author: Julio C. Olaya
Data: April 15, 2020
'''
# imports
import argparse
import matplotlib.pyplot as plt
import torch
import json
import random
import os
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

def load_data(dir_path):
    data_dir = dir_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=test_transforms)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=64)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64)

    dataloaders = [trainloader, validloader, testloader]
    image_datasets = [image_datasets_train, image_datasets_valid, image_datasets_test]
    
    return dataloaders, image_datasets

# Map labels
def map_labels(cat_to_name_file):
    with open(cat_to_name_file, 'r') as f:
            cat_to_name = json.load(f)

    return cat_to_name

# Load model
def get_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg19(pretrained=True)


    for param in model.parameters():
         param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, 512),
                           nn.ReLU(),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

    model.to(dev);

    return model, optimizer

# Load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model, optimizer = get_model()
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model

# Process image
def process_image(image):
    ima = Image.open(image)
    ima = ima.resize((255,255))
    tmp = (255-224) / 2.0
    ima = ima.crop((tmp,tmp, 255-tmp, 255-tmp))
    ima = np.array(ima)/254

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ima = (ima - mean) / std

    return ima.transpose(2,0,1)

# Return image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ima = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ima = std * ima + mean
    ima = np.clip(ima, 0, 1)
    ax.imshow(ima)

    return ax

# Show image
def show_image(class_to_idx, prob, classes, cat_to_name_file, img_path, top_k):
    cat_to_name = map_labels(cat_to_name_file)
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    labels = [cat_to_name[idx_to_class[x]] for x in classes]

    max_prob = np.argmax(prob)
    label = labels[max_prob]

    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
    image = Image.open(img_path)
    ax1.set_title(label)
    ax1.imshow(image)
    ax1.axis('off')
    ax2.set_aspect(0.3)
    ax2.barh(np.arange(top_k), prob)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(labels)
    ax2.set_title('Probabilities')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
def display_results(class_to_idx, prob, classes, cat_to_name_file, img_path, top_k):
    cat_to_name = map_labels(cat_to_name_file)
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    labels = [cat_to_name[idx_to_class[x]] for x in classes]

    max_prob = np.argmax(prob)
    label = labels[max_prob]
    
    print("\nResults:")
    for i in range(len(classes)):
        print(labels[i] + ': ' + str(prob[i]))
   

    
