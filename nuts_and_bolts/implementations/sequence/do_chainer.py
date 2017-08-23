import numpy as np
import begin
import chainer
import chainer.links as L
import chainer.functions as F
import sys
from matplotlib import pyplot as plt

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


def train_seq(net, data):
    net.reset_state()
    net.cleargrads()
    for col in range(data.shape[1]):
        prediction = net(data[:, col, :])
    return prediction


@begin.start
def main():
    X, Y = get_data()
    rnn = RNN()
    print ("x shape:", X.shape)
    print ("y shape:", Y.shape)
#    optimizer = chainer.optimizers.SGD(lr=0.1)
    optimizer = chainer.optimizers.NesterovAG(lr=0.01, momentum=0.95)
    optimizer.setup(rnn)
    nb_epoch = 16
    for i in range(nb_epoch):
        for j in range(X.shape[0]):
            prediction = train_seq(rnn, X[j: j + 1])
            loss = F.mean_squared_error(prediction[:, 0], Y[j:j+1   ])
            loss.backward()
            optimizer.update()
        print (loss.data)

    rnn.reset_state()
    rnn.cleargrads()
    generated = []
    prefix = X[0:1]
    for i in range(100):
        pred = train_seq(rnn, prefix)
        generated.append(pred.data[0, 0])
        prefix = np.vstack([prefix[0], pred.data])[np.newaxis, 1:, :]
    plt.plot(np.arange(len(generated)), generated)
    plt.savefig("result_chainer.png")
