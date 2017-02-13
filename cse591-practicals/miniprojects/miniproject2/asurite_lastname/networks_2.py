# sample_submission.py
import numpy as np


class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """

    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.alpha = 0.01
        self.loss_buffer = []
        u_labels = np.unique(labels)
        self.y = np.vstack(np.array(labels))
        label_index = [self.y == u_label for u_label in u_labels]
        self.y = np.asarray(np.hstack(label_index), dtype=np.float32)

        self.activations = []
        self.params = []  # [(w,b),(w,b)]
        self.ip_dim = data.shape[1]
        if(len(self.y.shape) > 1):
            self.op_dim = self.y.shape[1]
        else:
            self.op_dim = 1
        self.hdim = [3]
        self.numhid = len(self.hdim)
        self.params.append(np.random.rand(self.ip_dim, self.hdim[0]))
        if(self.hdim > 1):
            for i in range(1, len(self.hdim)):
                self.params.append(np.random.rand(
                    self.hdim[i - 1], self.hdim[i]))
        self.params.append(np.random.rand(self.hdim[-1], self.op_dim))
        self.backprop()

    def get_params(self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        temp_params = []
        for param in self.params:
            temp_params.append((param[:, :-1], param[:, -1]))
        return temp_params

    def backprop(self):
        """
        Method that is used to Backpropogate the gradients

        Returns:
        """
        for i in range(500):
            y = self.forwardpass(self.x)
            t = self.y
            common = y - t
            delta_wl = self.activations[0].T.dot(common)
            self.params[-1] -= self.alpha * (delta_wl)
            # print(delta_wl)
            dhl_da1 = self.params[-1].T
            a1 = self.activations[0]
            da1_dh1 = (1 - np.power(a1, 2))
            # dh1_dw1 = self.add_ones_column(self.x)
            dh1_dw1 = self.x.T
            delta_w1 = common.dot(dhl_da1) * da1_dh1
            delta_w1 = dh1_dw1.dot(delta_w1)
            # print(delta_w1.shape)
            # print(self.params[0].shape)
            self.params[0] -= self.alpha * delta_w1
        # print(delta_w1)

    def forwardpass(self, x):
        # if(x.shape[1] != self.x.shape[1]):
        #     x = self.add_ones_column(x)
        # for i in range(len(self.hdim)):
        self.activations = []
        h1 = x.dot(self.params[0])
        a1 = np.tanh(-1 * h1)
        # a1 = 1 / (1 + a1)
        self.activations.append(a1)
        # a1 = self.add_ones_column(a1)

        h_l = a1.dot(self.params[-1])
        a2 = np.exp(-1 * h_l)
        a2 = 1 / (1 + a2)
        self.activations.append(a2)
        return a2

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

        y = np.argmax(self.forwardpass(x), axis=1)
        # y[y > 0.5] = 1
        # y[y < 0.5] = 0
        print(y)
        return y

    def add_ones_column(self, x):
        return np.concatenate((x, np.ones((len(x), 1))), axis=1)


class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """

    def __init__(self, data, labels):
        super(mlnn, self).__init__(data, labels)


if __name__ == '__main__':
    pass
