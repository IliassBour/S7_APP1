import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.w = np.zeros((output_count, input_count))
        self.b = np.zeros(output_count)

    def get_parameters(self):
        r_dict = {
            "w": self.w,
            "b": self.b
        }
        return r_dict

    def get_buffers(self):
        return self.get_parameters()

    def forward(self, x):
        self.y = x @ self.w.T + self.b
        r_dict = {"X": x}

        return self.y, r_dict

    def backward(self, output_grad, cache):
        #dL/dX
        self.input_grad = output_grad @ self.w
        #dL/dW
        self.w_grad = output_grad.T @ cache['X'][:]
        #dL/dB
        self.b_grad = np.sum(output_grad, axis=0)

        r_dict = {
            "w": self.w_grad,
            "b": self.b_grad
        }

        return self.input_grad, r_dict


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        self.input_count = input_count
        self.alpha = alpha
        self.epsilon = 1e-42 ### MAY NEED TO PUT A SMALLER NUMBER
        self.gamma = np.zeros(input_count)
        self.beta = np.zeros(input_count)
        self.global_mean = []
        self.global_variance = []

    def get_parameters(self):
        r_dic = {
            "global_mean": self.global_mean,
            "global_variance": self.global_variance,
            "gamma": self.gamma,
            "beta": self.beta,
            "epsilon": self.epsilon
        }
        return r_dic

    def get_buffers(self):
        return self.get_parameters()

    def forward(self, x):
        mean = np.mean(x)
        variance = np.var(x)

        if not self.global_mean:
            self.global_mean = mean
        else:
            self.global_mean = (1 - self.alpha)*self.global_mean + self.alpha*mean

        if not self.global_variance:
            self.global_variance = variance
        else:
            self.global_variance = (1 - self.alpha)*self.global_variance + self.alpha*variance

        x_calc = np.zeros_like(x)
        with np.nditer(x_calc, op_flags=['readwrite']) as it:
            for x in np.nditer(x, op_flags=['readwrite']):
                it = (x-self.global_mean)/(self.global_variance+self.epsilon) ###division-wise
                it.iternext()

        y = self.gamma * x_calc + self.beta ###multiplication-wise

        return y, {}

    def _forward_training(self, x):
        return 0

    def _forward_evaluation(self, x):
        return 0

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {"y": self.y}

    def get_buffers(self):
        raise self.get_parameters()

    def forward(self, x):
        self.y = 1/(1+np.exp(-x))

        return self.y, {"y": self.y}

    def backward(self, output_grad, cache):
        self.y_grad = (1 - self.y) * self.y
        self.y_grad *= output_grad

        r_dict = {
            "y": self.y_grad
        }

        return self.y_grad, r_dict


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {"y": self.y}

    def get_buffers(self):
        return self.get_parameters()

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

        return self.y_grad, {"y_grad": self.y_grad}
