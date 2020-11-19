#!/usr/bin/env python

import torch
from torch.nn import LSTM, Linear
from torch.nn.functional import cross_entropy
from torch.optim import SGD

input_seq = [0, 0, 0, 1, 1, 1] * 1000
input_seq[:10], input_seq[-10:]

seq_len = 8


class BinaryPredictor(torch.nn.Module):
    def __init__(
        self,
        batch=1,
        input_size=1,
        hidden_size=20,
        num_layers=1,
        num_directions=1,
    ):
        super().__init__()
        self.rnn = LSTM(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=(num_directions == 2),
        )
        self.linear = Linear(hidden_size, 2)
        self.c0 = torch.zeros((num_layers * num_directions, batch, hidden_size))
        self.h0 = torch.zeros((num_layers * num_directions, batch, hidden_size))

    def forward(self, x):
        self.h0.detach_()
        self.c0.detach_()
        output = self.rnn(window, (self.h0, self.c0))
        # print(output)
        output, (self.h0, self.c0) = output
        yhat = self.linear(output[-1])
        # print(yhat)
        return yhat


bin_predictor = BinaryPredictor()
opt = SGD(bin_predictor.parameters(), lr=0.001, momentum=0.9)


for start in range(len(input_seq) - seq_len):
    end = start + seq_len
    window = torch.Tensor(input_seq[start:end]) - 0.5
    window.unsqueeze_(1)
    window.unsqueeze_(1)
    y = torch.LongTensor([input_seq[end]])
    yhat = bin_predictor(window)  # or any other model :)
    loss = cross_entropy(yhat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss, yhat, y)
