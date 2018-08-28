#!/usr/bin/env python
import argparse
import json

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
from datetime import datetime
from tb_chainer import SummaryWriter, name_scope, within_name_scope, utils

from clusterone import get_data_path, get_logs_path

# TensorBoard Extension
class TensorBoardReport(chainer.training.Extension):
    def __init__(self, out_dir):
        self.writer = SummaryWriter(out_dir)

    def __call__(self, trainer):
        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            if isinstance(v, chainer.Variable):
                value = v.data
            # elif isinstance(v, chainer.cuda.cupy.ndarray):
            elif isinstance(v, chainer.backends.cuda.ndarray):
                value = chainer.cuda.to_cpu(v)
            else:
                value = v

            self.writer.add_scalar(n, value, n_iter)

        # Optimizer
        link = trainer.updater.get_optimizer('main').target
        for name, param in link.namedparams():
            self.writer.add_histogram(name, chainer.cuda.to_cpu(param.data), n_iter)

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    @within_name_scope('MLP')
    def forward(self, x):
        with name_scope('linear1', self.l1.params()):
            h1 = F.relu(self.l1(x))
        with name_scope('linear2', self.l2.params()):
            h2 = F.relu(self.l2(h1))
        with name_scope('linear3', self.l3.params()):
            o = self.l3(h2)
        return o

def main():
    # This script is almost identical to train_mnist.py. The only difference is
    # that this script uses data-parallel computation on two GPUs.
    # See train_mnist.py for more details.
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    args.out = get_logs_path(root=args.out)

    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    chainer.backends.cuda.get_device_from_id(0).use()

    model = L.Classifier(MLP(args.unit, 10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # ParallelUpdater implements the data-parallel gradient computation on
    # multiple GPUs. It accepts "devices" argument that specifies which GPU to
    # use.
    try:
        config = os.environ['TF_CONFIG']
        config = json.loads(config)
        n_gpus = len(config['cluster']['worker'])
        devices = {str(i) for i in range(1,n_gpus)}
        devices['main'] = 0
    except:
        devices = {'main': 0}

    updater = training.updaters.ParallelUpdater(
        train_iter,
        optimizer,
        # The device of the name 'main' is used as a "master", while others are
        # used as slaves. Names other than 'main' are arbitrary.
        devices=devices,
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(TensorBoardReport(args.out), trigger=(100, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
