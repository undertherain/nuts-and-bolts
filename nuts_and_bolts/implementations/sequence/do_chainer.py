import numpy as np
import begin
import chainer
import chainer.links as L
import chainer.functions as F
import sys

sys.path.append("../../data/sequence")


from sequence import get_data


init = chainer.initializers.HeUniform(scale=1.0, dtype=None)


class RNN(chainer.Chain):
    def __init__(self):
        super(RNN, self).__init__(
            l1=L.LSTM(1, 16),  # the first LSTM layer
            l2=L.LSTM(None, 16),  # the first LSTM layer
            out=L.Linear(None, 1),  # the feed-forward output layer
        )
        # for param in self.params():
            # param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = True

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        # Given the current word ID, predict the next word.
        h = x
        h = self.l1(h)
        h = self.l2(h)
        h = self.out(h)
        return h


@begin.start
def main():
    X, Y = get_data()
    rnn = RNN()
    print ("hi")
#    optimizer = chainer.optimizers.SGD(lr=0.1)
    optimizer = chainer.optimizers.NesterovAG(lr=0.1, momentum=0.95)
    optimizer.setup(rnn)
    nb_epoch = 20
    for i in range(nb_epoch):
        rnn.reset_state()
        rnn.cleargrads()
        for col in range(X.shape[1]):
            prediction = rnn(X[:, col, :])
        loss = F.mean_squared_error(prediction[:, 0], Y)
        loss.backward()
        optimizer.update()
        if i % 2 == 0:
            print (loss.data)
