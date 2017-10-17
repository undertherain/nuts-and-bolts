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
            conv1=L.Convolution2D(None, 16, 3, pad=1),
            conv2=L.Convolution2D(None, 16, 3, pad=1),
            conv3=L.Convolution2D(None, 16, 3, pad=1),
            conv4=L.Convolution2D(None, 16, 3, pad=1),
            conv5=L.Convolution2D(None, 1, 3, pad=1),
        )
        self.train = train

    def encode(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        return h

    def decode(self, x):
        h = x
        h = F.relu(self.conv3(h))
        h = F.unpooling_2d(h, 2, outsize=(32, 32))
        h = F.relu(self.conv4(h))
        h = F.unpooling_2d(h, 2, outsize=(64, 64))
        h = F.sigmoid(self.conv5(h))
        return h

    def __call__(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        #accuracy = F.binary_accuracy(y, t)
        chainer.report({'loss': loss}, self)
        return loss


def main():
    net = Net()
    model = Classifier(net)
    #res = net(X_train[0:1])
    #print(res.shape)
    ds_train = chainer.datasets.tuple_dataset.TupleDataset(X_train, X_train)
    train(model, ds_train)


if __name__ == "__main__":
    main()
