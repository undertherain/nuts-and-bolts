import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import dagen
import dagen.image
from dagen.image.image import get_ds_counting
import PIL

from mlgym.trainer import train

dim_image=64

params = {}
params["batch_size"] = 10
params["nb_epoch"] = 100

class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 2, 4, pad=3),
            # Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None, deterministic=False)
            # conv1=L.Convolution2D(1, 2, 4, pad=3, initialW=w, initial_bias=np.array([-4,-2], dtype=np.float32)) ,
            # conv2=L.Convolution2D(None, 2, 3, pad=2),
            # conv3=L.Convolution2D(None, 2, 3, pad=2),
            # l1=L.Linear(None, 2, initialW=np.array([[0,0.26],[1,0]],dtype=np.float32)),
            l1=L.Linear(None, 2),
        )
        self.train = train

    def get_features(self, x):
        h = x
        # h = F.relu(self.conv1(h))
        h = F.leaky_relu(self.conv1(h))
        # h = F.leaky_relu(self.conv2(h))
        # h = F.max_pooling_2d(h, 2)
        # h = F.relu(self.conv3(h))
        return h

    def __call__(self, x):
        h = self.get_features(x)
        h = F.sum(h, axis=(2, 3))
        h = self.l1(h)
        return h


class Model(chainer.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        #print("y_shape:", y.shape)
        #print("t_shape:", t.shape)
        #loss = F.softmax_cross_entropy(y, t)
        loss = F.mean_absolute_error(y, t.astype(np.float32))
        chainer.report({'loss': loss}, self)
        return loss


def main():
    X_train, Y_train = get_ds_counting(cnt_samples=1000)
    X_test, Y_test = get_ds_counting(cnt_samples=100)
    X_train = np.expand_dims(X_train, axis=1).astype(np.float32) / 255
    X_test = np.expand_dims(X_test, axis=1).astype(np.float32) / 255
    print(X_train.shape)
    print(X_test.shape)
    
    net = CNN()
    model = Model(net)

    ds_train = chainer.datasets.tuple_dataset.TupleDataset(X_train, Y_train)
    ds_test = chainer.datasets.tuple_dataset.TupleDataset(X_test, Y_test)
    train(model, ds_train, ds_test, params)

if __name__ == "__main__":
    main()
