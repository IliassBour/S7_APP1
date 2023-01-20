import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        seed = np.random.default_rng(420)
        self.w = seed.normal(0, 2/(input_count+output_count), size=(output_count, input_count))
        self.b = seed.normal(0, 2/output_count, size=output_count)
        super().__init__()

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

    def __init__(self, input_count, alpha=0.10):
        self.input_count = input_count
        self.alpha = alpha
        self.epsilon = 1e-42 ### MAY NEED TO PUT A SMALLER NUMBER
        self.gamma = np.zeros(input_count)
        self.beta = np.ones(input_count)
        self.global_mean = np.zeros(input_count)
        self.global_variance = np.zeros(input_count)
        super().__init__()

    def get_parameters(self):
        r_dic = {
            "global_mean": self.global_mean,
            "global_variance": self.global_variance,
            "gamma": self.gamma,
            "beta": self.beta,
            "epsilon": np.array([self.epsilon])
        }
        return r_dic

    def get_buffers(self):
        return self.get_parameters()

    def forward(self, x):
        if self.is_training():
            self.y, rdic = self._forward_training(x)
        else:
            self.y, rdic = self._forward_evaluation(x)

        return self.y, rdic

    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)

        if np.all(self.global_mean == 0):
            self.global_mean = mean
        else:
            self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * mean

        if np.all(self.global_variance == 0):
            self.global_variance = variance
        else:
            self.global_variance = (1.0 - self.alpha) * np.float_(self.global_variance) + self.alpha * variance

        x_norm = (x - self.global_mean)
        square = np.sqrt(self.global_variance + self.epsilon)
        x_calc = x_norm / square
        self.y = self.gamma * x_calc + self.beta  ###multiplication-wise

        rdic = {
            "y": self.y,
            "x_calc": x_calc,
            "x": x,
            "x_norm": x_norm,
            "square": square
        }

        return self.y, rdic

    def _forward_evaluation(self, x):
        x_norm = (x - self.global_mean)
        square = np.sqrt(self.global_variance + self.epsilon)
        x_calc = x_norm / square
        self.y = self.gamma * x_calc + self.beta  ###multiplication-wise

        rdic = {
            "y": self.y,
            "x_calc": x_calc,
            "x": x,
            "x_norm": x_norm,
            "square": square
        }

        return self.y, rdic

    def backward(self, output_grad, cache):
        x_calc = cache["x_calc"]
        x = cache['x']
        x_norm = cache["x_norm"]
        square = cache["square"]
        M = np.shape(output_grad)[0] * np.shape(output_grad)[1]

        gamma_grad = np.sum((output_grad * x_calc), axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        x_calc_grad = output_grad * self.gamma
        x_norm_grad = x_calc_grad / square
        mean_grad = -1 * np.sum(x_norm_grad, axis=0)
        square_grad = np.sum(x_calc_grad * x_norm * -square**(-2), axis=0)
        var_grad = square_grad / 2 / square
        x_grad = x_calc_grad/square + 1/M * (2 * var_grad * x_norm + mean_grad)

        rdic = {
            "gamma": gamma_grad,
            "beta": beta_grad,
            "global_mean": self.global_mean,
            "global_variance": self.global_variance,
            "epsilon": self.epsilon
        }

        return x_grad, rdic

class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}
        #return {"y": np.zeros(1)}

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
        return {}
        #return {"y": np.zeros(1)}

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

        return self.y_grad, {"y": self.y_grad}
