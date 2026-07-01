from __future__ import division
import numpy as np


class BrierLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        self.softmax = softmax
        loss = np.sum(np.square(softmax - target), axis = 1)
        return np.mean(loss)
        # TODO END

    def backward(self, input, target):
		# TODO START
        B, C = input.shape
        p = self.softmax
        y = target
        dL_dp = 2 * (p - y) / B 
        grad_input = p * (dL_dp - np.sum(dL_dp * p, axis=1, keepdims=True))
        return grad_input
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp = np.exp(input)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        self.softmax = softmax
        Entropyloss = - np.sum(target * np.log(softmax),axis=1)
        return np.mean(Entropyloss)
        # TODO END

    def backward(self, input, target):
        # TODO START
        B = input.shape[0]
        grad_input = (self.softmax - target) / B
        return grad_input
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        B, C = input.shape
        # exp = np.exp(input)
        # softmax = exp / sum(exp, axis=1, keepdims = True)
        self.margin = 5
        target_index = np.argmax(target, axis=1)
        x_t = input[np.arange(B), target_index].reshape(-1, 1)
        h_k = self.margin - x_t + input
        h_k[np.arange(B), target_index] = 0
        loss = np.maximum(0, h_k).sum(axis = 1)
        self.cache = (h_k, target_index)
        return np.mean(loss)
        # TODO END

    def backward(self, input, target):
        # TODO START
        B, C = input.shape
        margins, t = self.cache

        grad = np.zeros_like(input)
        grad[margins > 0] = 1
        counts = np.sum(margins > 0, axis=1)
        grad[np.arange(B), t] = -counts

        grad /= B
        return grad
        # TODO END


# Bonus
class LabelSmoothingCrossEntropyLoss(object):
    def __init__(self, name, epsilon=0.1):
        self.name = name
        self.epsilon = float(epsilon)

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END