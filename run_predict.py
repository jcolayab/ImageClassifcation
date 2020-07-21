#!/bin/sh
#!/usr/bin/env python3

python predict.py  --input flowers/test/1/image_06743.jpg --checkpoint save_dir/checkpoint.pth --top_k 3  --category_names 'cat_to_name.json' --gpu gpu