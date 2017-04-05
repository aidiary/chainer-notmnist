#!/bin/bash

# for max_train in 50 100 1000 5000 20000; do
#     python sklearn_logreg.py $max_train
# done

python train_notmnist.py --model LogReg
python train_notmnist.py --model MLP --unit 1024
