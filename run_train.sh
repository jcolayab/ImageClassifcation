#!/bin/sh
#!/usr/bin/env python3
python train.py  --data_dir flowers --dir checkpoint.pth --arch vgg19 --learning_rate 0.01 --hidden_units 512 --epochs 40 --gpu gpu