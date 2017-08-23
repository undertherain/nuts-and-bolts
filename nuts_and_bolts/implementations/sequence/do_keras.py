import numpy as np
import keras
from keras.layers import Dense, LSTM, Activation, Convolution1D, Flatten, Dropout, MaxPooling1D
from keras.optimizers import RMSprop,SGD,Adam
import begin
from matplotlib import pyplot as plt
import sys


sys.path.append("../../data/sequence")

from sequence import get_data, size_seq


@begin.start
def main():
    X, Y = get_data()
    batch_size = 16
    # init = "he_uniform"
    # init = "he_normal"
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    model = keras.models.Sequential()
    model.add(LSTM(16, input_shape=(size_seq, 1), kernel_initializer=init,
        batch_input_shape=(batch_size, size_seq, 1), return_sequences=True, stateful=False))
    model.add(LSTM(16, input_shape=(size_seq, 1), kernel_initializer=init,
        return_sequences=False, stateful=False))
    model.add(Dense(1, kernel_initializer=init))
    # model.add(Activation(''))
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=0.0, nesterov=True)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(X, Y, epochs=16, validation_data=None, batch_size=batch_size)
    prefix = X[0:1]
    generated = []
    for i in range(100):
        pred = model.predict(prefix)
        generated.append(pred[0, 0])
        prefix = np.vstack([prefix[0], pred])[np.newaxis, 1:, :]
    plt.plot(np.arange(len(generated)) + prefix.shape[1], generated, label="predicted")

    ground_truth = list(X[0])
    for i in Y:
        ground_truth.append(i)
    plt.plot(np.arange(len(ground_truth)), ground_truth, label="ground truth", alpha=0.7)

    plt.legend()
    plt.savefig("result_keras.png")
    # plt.show()
