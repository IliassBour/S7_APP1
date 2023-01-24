import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        y_calc = softmax(x)

        target1hot = np.zeros(np.shape(x))
        for i in range(len(target)):
            target1hot[i,target[i]] = 1

        self.loss = -1 * np.sum(target1hot * np.log(y_calc))/len(target)


        #Calcul gradient
        y = target1hot.argmax(axis=1)
        m = y.shape[0]
        grad = softmax(x)
        grad[range(m), y] -= 1
        grad = grad / m
        self.input_grad = grad

        return self.loss, self.input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    x_temp = np.copy(x)
    nb_rows, _ = np.shape(x)
    for i in range(nb_rows):
        x_temp[i,:] = np.exp(x_temp[i,:])/np.sum(np.exp(x_temp[i,:]))

    return x_temp


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = (np.sum((x - target) ** 2)) / target.size

        input_grad = (2 * (x - target)) / target.size

        return loss, input_grad
