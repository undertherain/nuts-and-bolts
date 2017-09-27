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
X_train = X_train.astype(np.float32)
print(X_train.dtype)
print(Y_train.dtype)

class Net(chainer.Chain):

    def __init__(self):
        super(Net, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, 10)  # n_in -> n_units
            self.l2 = L.Linear(None, 10)  # n_units -> n_units
            self.l3 = L.Linear(None, 2)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = L.Classifier(Net())
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
