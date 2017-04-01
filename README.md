# chainer_notmnist

## Overview

- Chainerで[notMNIST](http://yaroslavvb.blogspot.jp/2011/09/notmnist-dataset.html)の画像認識

## Requirements

- Python 3.5
- Chainer 1.21.0
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

## Multi-layer perceptron


## Convolutional neural networks


## Reference

- [Udacity: Deep Learning](https://classroom.udacity.com/courses/ud730/)
- [tensorflow/examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)

## License

MIT License
