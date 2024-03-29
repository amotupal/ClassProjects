# sample_submission.py
import numpy as np
# import matplotlib.pyplot as plt


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
    """float: This is the Learning Rate of for the training algorithm"""
    error_buffer = []
    """list: This is the list of errors for every epoch in the training"""
    l = 0.0000001
    """float: lambda value for the regularization"""

    def __init__(self, data, **kwargs):
        self.x, self.y = data
        self.w = np.random.rand(self.x.shape[1], 1)
        self.b = np.random.rand(1)
        input_shape = self.x.shape
        X = np.ones((input_shape[0], input_shape[1] + 1))
        X[:, :-1] = self.x
        self.x = X
        self.w = np.concatenate((self.w, np.reshape(self.b, (1, 1))), axis=0)
        self.X_train = []
        self.y_train = []

        if(len(self.x) > 1000):
            self.num_batches = 1000
            """Number of Minibatches"""
        elif(len(self.x > 10)):
            self.num_batches = len(self.x) / 10
        else:
            self.num_batches = 1
        self.batch_size = int(len(self.x) / self.num_batches)
        self.batchnum = 0
        self.error_buffer.append(self.get_error())
        self.next_batch()
        self.train()

    def next_batch(self):
        """
        This method stores the next minibatch data in self.X_train and labels in self.y_train
        """
        i = 0
        x_batch = []
        y_batch = []
        while((i + self.batchnum) < len(self.x)):
            x_batch.append(self.x[i + self.batchnum].tolist())
            y_batch.append(self.y[i + self.batchnum].tolist())
            i += self.batch_size
        self.X_train = np.array(x_batch)
        self.y_train = np.array(y_batch)
        self.batchnum = (self.batchnum + 1) % self.num_batches

    def train(self):
        """
        Method to train the Linear regressor
        """
        i = 1
        while(True):
            # + (self.l * np.hstack(self.w))
            delta = sum([(y - x.dot(self.w)) * x for x,
                         y in zip(self.X_train, self.y_train)])

            reg = 2 * self.l * self.w
            delta /= len(self.X_train)
            delta = np.reshape(delta, (len(delta), 1)) + reg
            # delta = (delta - np.mean(delta)) / np.std(delta)
            self.w += self.alpha * np.vstack(delta)
            self.error_buffer.append(self.get_error())
            if(self.batchnum == self.num_batches - 1):
                print("Training Error at epoch " + str(i) +
                      " :: " + str(self.error_buffer[-1]))
                i += 1

            if((np.abs(self.error_buffer[-2] - self.error_buffer[-1]) < 0.000001) or (i == 10)):
                break
            else:
                self.next_batch()

    def get_error(self):
        """
        This method calculates and returns the mean squared error at current iteration

        Returns:
            numpy.float64: A float of the error at current iteration
        """
        regularizer = l2_norm(self.w)
        # print(w.shape)
        error = self.y - self.x.dot(self.w)
        return np.mean(error * error + self.l * regularizer)
        # return sum([(y - x.T.dot(w)) ** 2 for x, y in zip(X_train, y_train)])

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


def l2_norm(w):
    """
    This method calculates the L2 norm for the given vector.
    Args:
        W: A vector whose L2 norm need to be calculated

    Returns:
        numpy.ndarray: with one element i.,e the l2_norm of the given vector.
    """
    return sum([x * x for x in w])


if __name__ == '__main__':
    pass
