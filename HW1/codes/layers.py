import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        lambda_ = 1.0570
        alpha_ = 1.67326
        out = np.where(input > 0,
                       lambda_ * input,
                       lambda_ * alpha_ * (np.exp(input) - 1) 
                       )
        return out
        # TODO END

    def backward(self, grad_output):
        # TODO START
        lambda_ = 1.0570
        alpha_ = 1.67326
        x = self._saved_tensor
        grad_input = np.where(x > 0,
                              lambda_,
                              lambda_ * alpha_ * np.exp(x)
                              )
        return grad_output * grad_input
        # TODO END

class HardSwish(Layer):
    def __init__(self, name):
        super(HardSwish, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)   
        output = np.where(input <= -3,
                          0,
                          np.where(input >= 3, input, input*(input+3)/6)
                         )     
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        x = self._saved_tensor
        grad_input = np.where(x <= -3,
                              0,
                              np.where(x > 3, 1, (2*x+3)/6)
                              )
        return grad_output * grad_input
        # TODO END

class Mish(Layer):
    def __init__(self, name):
        super(Mish, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        output = input*np.tanh(np.log(1+np.exp(input)))
        return output
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        x = self._saved_tensor
        g1 = np.tanh(np.log(1+np.exp(x)))
        g2 = 4*x/(np.square(np.exp(x) + 1 + 1/(np.exp(x)+1))*(1+np.exp(-x)))
        return (g1 + g2) * grad_output
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        output = input @ self.W + self.b
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        x = self._saved_tensor
        grad_x = grad_output @ self.W.T
        self.grad_W = x.T @ grad_output
        self.grad_b = grad_output.sum(axis = 0)
        return grad_x
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
