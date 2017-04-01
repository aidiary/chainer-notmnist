import argparse
import os
import pickle

import chainer
import chainer.links as L
from chainer.datasets import TupleDataset
from chainer import training
from chainer.training import extensions

import numpy as np
from net import *


def main():
    parser = argparse.ArgumentParser(description='Chainer example: notMNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--model', '-m',
                        choices=('LogReg', 'MLP', 'CNN'),
                        default='LogReg', help='model type')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('model type: {}'.format(args.model))
    print('')

    # Set up a neural network to train
    if args.model == 'LogReg':
        model = LogReg(n_out=10)
    elif args.model == 'MLP':
        model = MLP(n_out=10)
    elif args.model == 'CNN':
        model = CNN(n_out=10)

    model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    # Load the notMNIST dataset
    pickle_file = os.path.join('data', 'notMNIST.pickle')
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # test_datasetを訓練中の検証用データとして使う
    # 本来は検証にvalid_datasetを使って訓練後の最後の評価でtest_datasetを使うべき
    # TODO: 訓練後にtest_datasetでの評価は自分で書く必要あり？
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    test_labels = data['test_labels']

    num_train = len(train_dataset)
    num_test = len(test_dataset)

    if args.model == 'LogReg' or args.model == 'MLP':
        train_dataset = train_dataset.reshape(num_train, -1)
        test_dataset = test_dataset.reshape(num_test, -1)

    train = TupleDataset(train_dataset, train_labels)
    test = TupleDataset(test_dataset, test_labels)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
