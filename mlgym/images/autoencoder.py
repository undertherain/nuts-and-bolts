import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import dagen
import dagen.image
from dagen.image.image import get_ds_simple
from dagen.image.image import merge_samples

from ..trainer import train

params = {"nb_epoch" : 20 }
params["batch_size"] = 8

X_train, Y_train = get_ds_simple(cnt_samples=1000)
X_train = np.expand_dims(X_train, axis=1).astype(np.float32) / 255
Y_train = Y_train[:, np.newaxis]
print(X_train.shape)
print(Y_train.shape)


class Net(chainer.Chain):

    def __init__(self, train=True):
        super(Net, self).__init__(
            conv_e_1=L.Convolution2D(None, 16, 3, pad=1),
            conv_e_2=L.Convolution2D(None, 8, 3, pad=1),
            conv_e_3=L.Convolution2D(None, 8, 3, pad=1),
            conv_d_1=L.Convolution2D(None, 8, 3, pad=1),
            conv_d_2=L.Convolution2D(None, 8, 3, pad=1),
            conv_d_3=L.Convolution2D(None, 16, 3, pad=1),
            conv_d_4=L.Convolution2D(None, 1, 3, pad=1),
        )
        self.train = train

    def encode(self, x):
        h = x
        h = F.relu(self.conv_e_1(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv_e_2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv_e_3(h))
        h = F.max_pooling_2d(h, 2)
        return h

    def decode(self, x):
        h = x
        h = F.relu(self.conv_d_1(h))
        h = F.unpooling_2d(h, 2, outsize=(16, 16))
        h = F.relu(self.conv_d_2(h))
        h = F.unpooling_2d(h, 2, outsize=(32, 32))
        h = F.relu(self.conv_d_3(h))
        h = F.unpooling_2d(h, 2, outsize=(64, 64))
        h = F.sigmoid(self.conv_d_4(h))
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
        # accuracy = F.binary_accuracy(y, t)
        chainer.report({'loss': loss}, self)
        return loss


def make_noise(X):
    for idx, _ in enumerate(X):
        noise = np.random.rand(64, 64)
        X[idx, :, :, :] *= noise
        X[idx, :, :, :] += noise

    return X


def main():
    net = Net()
    model = Classifier(net)
    # res = net(X_train[0:1])
    # print(res.shape)
    # generate examples before training
    print("image size : ", X_train[:1].size)
    im = merge_samples(X_train[:10], Y_train[:10])
    im.save("/tmp/ae_original.png")

    noisy_X = make_noise(X_train)
    im = merge_samples(noisy_X, Y_train[:10])
    im.save("/tmp/noisy.png")

    encoded = net.encode(X_train[:1])
    print("encoded size", encoded.size)

    generated = net(X_train[:10])
    im = merge_samples(generated.data, Y_train)
    im.save("/tmp/ae_untrained.png")

    ds_train = chainer.datasets.tuple_dataset.TupleDataset(noisy_X, X_train)
    train(model, ds_train)

    generated = net(X_train[:10])
    im = merge_samples(generated.data, Y_train)
    im.save("/tmp/ae_trained.png")

    denoised = net(noisy_X)
    im = merge_samples(denoised.data, Y_train[:10])
    im.save("/tmp/denoised.png")


if __name__ == "__main__":
    main()
