import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = self.__sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

    @staticmethod
    def __sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
