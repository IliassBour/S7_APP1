import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        self.y = x @ self.w.T + self.b

        return self.y

    def backward(self, output_grad, cache):
        #dL/dX
        self.input_grad = output_grad @ self.w
        #dL/dW
        self.w_grad = output_grad.T @ cache # x
        #dL/dB
        self.b_grad = np.sum(output_grad, axis=0)

        return self.input_grad, self.w_grad, self.b_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {"y": self.y, "y_grad": self.y_grad}

    def get_buffers(self):
        raise self.get_parameters()

    def forward(self, x):
        self.y = 1/(1+np.exp(-x))

        return self.y, {"y": self.y}

    def backward(self, output_grad, cache):
        self.y_grad = (1 - self.y) * self.y
        self.y_grad *= output_grad

        return self.y_grad


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        self.y = np.array(x)
        self.y[self.y < 0] = 0

        return self.y, x

    def backward(self, output_grad, cache):
        self.y_grad = np.array(cache) # x
        with np.nditer(self.y_grad, op_flags=['readwrite']) as it:
            for element in it:
                if element[...] < 0:
                    element[...] = 0
                else:
                    element[...] = 1
        self.y_grad *= output_grad

        return self.y_grad
