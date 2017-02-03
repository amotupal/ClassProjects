# sample_submission.py
import numpy as np
from sklearn.linear_model.perceptron import Perceptron
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


class regressor(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """
    alpha = 0.1
    error_buffer = []
    error_valid = []
    l = 0.001

    def __init__(self, data):
        self.x, self.y = data
        # Here is where your training and all the other magic should happen.
        # Once trained you should have these parameters with ready.
        self.w = np.random.rand(self.x.shape[1], 1)
        self.b = np.random.rand(1)
        input_shape = self.x.shape
        X = np.ones((input_shape[0], input_shape[1] + 1))
        X[:, :-1] = self.x
        self.x = X
        self.w = np.concatenate((self.w, np.reshape(self.b, (1, 1))), axis=0)
        self.error_buffer.append(get_error(self.x, self.y, self.w, self.l))
        n = len(self.x) * 0.9
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y, test_size=0.2, random_state=40)
        self.X_train = self.x
        self.y_train = self.y
        self.train()
        plt.plot(self.error_buffer)
        plt.plot(self.error_valid)
        plt.show()

    def train(self):
        i = 0
        while(True):
            # + (self.l * np.hstack(self.w))
            delta = sum([(y - x.dot(self.w)) * x for x,
                         y in zip(self.X_train, self.y_train)])
            # delta = ((self.y_train -
            # self.X_train.dot(self.w)).T.dot(self.X_train)) /
            # len(self.X_train)
            reg = 2 * self.l * self.w
            delta /= len(self.X_train)
            delta = np.reshape(delta, (len(delta), 1)) + reg
            print(delta.shape)
            self.w += self.alpha * np.vstack(delta)
            self.error_buffer.append(
                get_error(self.X_train, self.y_train, self.w, self.l))
            self.error_valid.append(
                get_error(self.X_val, self.y_val, self.w, self.l))
            print("Error at epoch " + str(i) + " :: " +
                  str(self.error_buffer[-1]) + "Error Difference ::" + str((self.error_buffer[-2] - self.error_buffer[-1])))
            i += 1
            if(self.error_buffer[-2] - self.error_buffer[-1] < 0.00001):
                break

    def get_params(self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return (self.w[:-1], self.w[-1])

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # Here is where you write a code to evaluate the data and produce
        # predictions.
        x = np.concatenate((x, np.vstack(np.ones(len(x)))), axis=1)
        return(x.dot(self.w))


def l2_norm(W):
    return sum([x * x for x in W])


def get_error(X, y, w, l):
    regularizer = l2_norm(w)
    # print(w.shape)
    error = y - X.dot(w)
    return np.mean(error * error + l * 2 * regularizer)
    # return sum([(y - x.T.dot(w)) ** 2 for x, y in zip(X_train, y_train)])

if __name__ == '__main__':
    pass
