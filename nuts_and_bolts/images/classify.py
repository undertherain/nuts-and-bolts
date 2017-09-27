import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import dagen
import dagen.image
from dagen.image.image import get_ds_simple


params = {}
params["batch_size"] = 8
X_train, Y_train = get_ds_simple(cnt_samples=1000)
X_train = X_train.astype(np.float32)[:,np.newaxis]
Y_train = Y_train[:,np.newaxis]
print(X_train.shape)
print(Y_train.shape)


class Net(chainer.Chain):

    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.Convolution2D(1, 32, 2), #Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None, deterministic=False)
            conv2=L.Convolution2D(None, 32, 2),
            l1=L.Linear(None, 1)
        )
        self.train = train

    def __call__(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        return self.l1(h)

class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.sigmoid_cross_entropy(y, t)
        accuracy = F.binary_accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

model = Classifier(Net())
nb_epoch = 10
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)
train = chainer.datasets.tuple_dataset.TupleDataset(X_train, Y_train)
train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (nb_epoch, 'epoch'), out='/tmp/result')
# trainer.extend(extensions.Evaluator(test_iter, model, device=id_device))
# trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
# trainer.extend(extensions.ProgressBar())
trainer.run()
