#!/bin/bash
python gen_train.py 300 3000
python LR_train.py
python gen_train.py 300 3000
python LR_test.py
