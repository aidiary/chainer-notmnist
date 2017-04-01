import os
import sys
import pickle
from sklearn.linear_model import LogisticRegression


def main():
    if len(sys.argv) != 2:
        print('usage: python sklearn_logreg.py [max_train]')
        sys.exit(1)

    # 訓練データの最大数
    max_train = int(sys.argv[1])
    print('max_train:', max_train)

    pickle_file = os.path.join('data', 'notMNIST.pickle')

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    test_labels = data['test_labels']

    num_train = len(train_dataset)
    num_test = len(test_dataset)

    # print('num_train:', num_train)
    # print('num_test:', num_test)

    train_dataset = train_dataset.reshape(num_train, -1)
    test_dataset = test_dataset.reshape(num_test, -1)

    # print(train_dataset.shape)  # (200000, 784)
    # print(test_dataset.shape)   # (18724, 784)

    # logistic regression
    logreg = LogisticRegression()
    logreg.fit(train_dataset[:max_train], train_labels[:max_train])
    print('accuracy:', logreg.score(test_dataset, test_labels))


if __name__ == '__main__':
    main()
