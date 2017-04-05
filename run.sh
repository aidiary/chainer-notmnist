#!/bin/bash

# scikit-learnのロジスティック回帰
# for max_train in 50 100 1000 5000 20000; do
#     python sklearn_logreg.py $max_train
# done

python train_notmnist.py --model LogReg　--out result_logreg
python train_notmnist.py --model MLP --unit 1024 --out result_mlp

# L2正則化を導入
python train_notmnist.py --model LogReg --reg L2 result_logreg_l2
python train_notmnist.py --model MLP --unit 1024 --reg L2 result_mlp_l2
