import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import dagen
import dagen.image
from dagen.image.image import get_ds_simple

from ..trainer import train

params = {}
params["batch_size"] = 8

X_train, Y_train = get_ds_simple(cnt_samples=1000)
X_train = np.expand_dims(X_train, axis=1).astype(np.float32) / 255
Y_train = Y_train[:, np.newaxis]
print(X_train.shape)
print(Y_train.shape)


class Net(chainer.Chain):

    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.Convolution2D(1, 32, 2),
            conv2=L.Convolution2D(None, 32, 2),
            l1=L.Linear(None, 10),
            l2=L.Linear(None, 1)
        )
        self.train = train

    def __call__(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.sigmoid_cross_entropy(y, t)
        accuracy = F.binary_accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


def main():
    model = Classifier(Net())
    ds_train = chainer.datasets.tuple_dataset.TupleDataset(X_train, Y_train)
    train(model, ds_train)


if __name__ == "__main__":
    main()
