#!/usr/bin/env python3

'''
Author: Julio C. Olaya
Data: April 15, 2020
'''
# imports
import argparse
import torch
import time
import json
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

from get_input_args import get_input_args_train
from utility import load_data, map_labels

from workspace_utils import active_session

# Train neural network
def train_nn(in_arg):
    # Load data
    dataloaders, data_images = load_data(in_arg.data_dir)

    # Build network
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg19(pretrained=True)


    for param in model.parameters():
        param.requires_grad = False

    hidden_units = in_arg.hidden_units
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           #nn.Dropout(p=0.5),
                           nn.ReLU(),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(dev);
    
     # Train network
    epochs = in_arg.epochs
    with active_session():
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in dataloaders[0]:
                images, labels = images.to(dev), labels.to(dev)

                optimizer.zero_grad()

                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in dataloaders[1]:
                        images, labels = images.to(dev), labels.to(dev)

                        log_ps = model.forward(images)
                        test_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print("\nEpoch: {}/{} " .format(epoch+1, epochs), "\nTraining loss: {:.4f} ".format(running_loss/len(dataloaders[0])))
                print("Test loss: {:.4f} ".format(test_loss/len(dataloaders[1])),"\nAccuracy: {:.4f} ".format(accuracy/len(dataloaders[1])))

                running_loss = 0
                model.train()
            if (accuracy/len(dataloaders[1])) > 0.90:
                break
            
             # Save checkpoints
    model.class_to_idx = data_images[0].class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers':in_arg.hidden_units,
              'arch': in_arg.arch,
              'learning_rate': in_arg.learning_rate,
              'batch_size': 64,
              'classifier' : classifier,
              'epochs': in_arg.epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, in_arg.dir)


# Main functions
def main():
    # read initial inputs
    in_arg = get_input_args_train()

    # train neural network
    train_nn(in_arg)



# Call to main function to run the program
if __name__ == "__main__":
    main()

