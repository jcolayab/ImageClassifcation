#!/usr/bin/env python3

'''
Author: Julio C. olaya
Date: April 15, 2020
'''

import argparse
def get_input_args_train():

        # Create Parserusing argumentParser
        parser = argparse.ArgumentParser(description='Training flower classifier')

        # Create command line arguments as mentioned above using add_argument() from ArguementParser method
        parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'data directory')
        parser.add_argument('--dir', type = str, default = 'checkpoint.pth', help = 'checkpoint')
        parser.add_argument('--arch', type = str, default = 'vgg19', help = 'name of the model architecture')
        parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning rate')
        parser.add_argument('--hidden_units', type = int, default = 512, help = 'hidden units')
        parser.add_argument('--epochs', type = int, default = 20, help = 'epochs')
        parser.add_argument('--gpu', type = str, default = 'gpu', help = 'gpu')


        in_args = parser.parse_args()

        print("\nData directory path: ", in_args.data_dir)
        print("Checkpoint: ", in_args.dir)
        print("Arch: ", in_args.arch)
        print("Learnig rate: ", in_args.learning_rate)
        print("No. of hidden units: ", in_args.hidden_units)
        print("No. of epochs: ", in_args.epochs)
        print("GPU: ", in_args.gpu)

        return in_args
    
def get_input_args_predict():

    # Create Parserusing argumentParser
    parser = argparse.ArgumentParser(description='Predicting flower classifier')

        # Create command line arguments as mentioned above using add_argument()
    parser.add_argument('--input', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'save_directory')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'checkpoint')
    parser.add_argument('--top_k', type = int, default = 3, help = 'top k most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'category name')
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'gpu')


    in_args = parser.parse_args()

    print("\nImage path: ", in_args.input)
    print("Checkpoint: ", in_args.checkpoint)
    print("top k: ", in_args.top_k)
    print("category_names: ", in_args.category_names)
    print("GPU: ", in_args.gpu)

    return in_args