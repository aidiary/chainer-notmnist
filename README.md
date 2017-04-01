# chainer_notmnist

## Overview

- Chainerで[notMNIST](http://yaroslavvb.blogspot.jp/2011/09/notmnist-dataset.html)の画像認識

## Requirements

- Python 3.5
- Chainer 1.22.0
- scikit-learn
- matplotlib

## Logistic regression (scikit-learn)

ロジスティック回帰（scikit-learn）でのベースライン

```
$ python sklearn_logreg.py [max_train]
```

|訓練データ数|テスト精度|
|---------|--------|
|50       |0.566   |
|100      |0.764   |
|1000     |0.830   |
|5000     |0.851   |
|200000   |0.891   |


## Logistic regression (Chainer)

- Chainerでの実装
- `test_dataset`は検証用データとして使用

```
$ python train_notmnist.py
GPU: -1
# unit: 1000
# Minibatch-size: 100
# epoch: 20
model type: LogReg

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.680755    0.409343              0.819455       0.893972                  2.97568
2           0.638873    0.40069               0.830325       0.897855                  6.50036
3           0.632791    0.401015              0.832135       0.897695                  9.9341
4           0.629504    0.399971              0.83261        0.897004                  13.3296
5           0.627661    0.397034              0.83289        0.89945                   16.7266
6           0.625969    0.403611              0.83323        0.895993                  20.2158
7           0.62468     0.395492              0.834135       0.897801                  23.5893
8           0.623673    0.40067               0.834225       0.895195                  27.3385
9           0.6234      0.396476              0.83411        0.896631                  30.7311
10          0.622067    0.395322              0.83471        0.897642                  34.1545
11          0.621883    0.396458              0.834235       0.897429                  37.7959
12          0.621211    0.397842              0.83477        0.897642                  41.258
13          0.62056     0.393788              0.83483        0.898067                  44.6625
14          0.62004     0.39635               0.834955       0.896844                  48.0537
15          0.619855    0.3958                0.83506        0.895514                  51.6186
16          0.619573    0.397582              0.83494        0.897323                  55.1184
17          0.619424    0.394369              0.83476        0.898865                  58.6306
18          0.619172    0.396384              0.8353         0.898759                  62.0721
19          0.618694    0.399222              0.83535        0.897057                  65.4735
20          0.618446    0.397801              0.835435       0.895195                  68.9314
```

## Multi-layer perceptron


## Convolutional neural networks


## Reference

- [Udacity: Deep Learning](https://classroom.udacity.com/courses/ud730/)
- [tensorflow/examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)

## License

MIT License
