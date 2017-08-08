import numpy as np
import begin
from matplotlib import pyplot as plt


size_whole = 201
size_seq = 16
naturals = np.arange(size_whole, dtype=np.float32)
values = np.sin(naturals / 4) + 1.3 * np.cos(naturals / 2)


def get_data():
    X = []
    Y = []
    for i in range(size_whole - size_seq - 1):
        X.append(values[i: i + size_seq, np.newaxis])
        Y.append(values[i + size_seq])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


@begin.start
def main():
    print("testing signal")
    X, Y = get_data()
    print("train shape:", X.shape)
    plt.plot(naturals, values)
    plt.savefig("ground_truth.png")
