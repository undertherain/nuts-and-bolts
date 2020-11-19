#!/usr/bin/env python

import torch
from torch.nn import LSTM, Embedding, Linear
from torch.nn.functional import cross_entropy  # , one_hot
from torch.optim import SGD

input_seq = [0, 0, 0, 1, 1, 1] * 1000
input_seq[:10], input_seq[-10:]

seq_len = 8
num_tokens = 2


class BinaryPredictor(torch.nn.Module):
    def __init__(
        self,
        batch=1,
        num_tokens=2,
        emb_size=6,
        hidden_size=10,
        num_layers=1,
        num_directions=1,
    ):
        super().__init__()
        self.emb = Embedding(num_tokens, emb_size)
        self.rnn = LSTM(
            emb_size,
            hidden_size,
            num_layers,
            bidirectional=(num_directions == 2),
        )
        self.linear = Linear(hidden_size, num_tokens)
        self.c0 = torch.zeros((num_layers * num_directions, batch, hidden_size))
        self.h0 = torch.zeros((num_layers * num_directions, batch, hidden_size))

    def forward(self, x):
        self.h0.detach_()
        self.c0.detach_()
        emb = self.emb(x)
        output = self.rnn(emb, (self.h0, self.c0))
        # print(output)
        output, (self.h0, self.c0) = output
        yhat = self.linear(output[-1])
        # print(yhat)
        return yhat


bin_predictor = BinaryPredictor()
opt = SGD(bin_predictor.parameters(), lr=0.001, momentum=0.9)


for start in range(len(input_seq) - seq_len):
    end = start + seq_len
    window = input_seq[start:end]
    window = torch.LongTensor(window)
    # window = one_hot(window, num_classes=num_tokens)
    # window = window.float()
    window.unsqueeze_(1)
    y = torch.LongTensor([input_seq[end]])
    yhat = bin_predictor(window)  # or any other model :)
    loss = cross_entropy(yhat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss, yhat, y)
