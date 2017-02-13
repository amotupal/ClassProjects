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
        print(self.x)
        self.activation_type = 'tanh'
        if(self.x.shape[1] > 2):
            self.activation_type = 'ReLU'
        self.mean = self.x.mean()
        self.std = self.x.std()
        self.x = (self.x - self.mean) / self.std
        self.alpha = 0.01
        self.lamda = 0.001
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
        self.params.append(
            (np.random.rand(self.ip_dim, self.hdim[0]) / np.sqrt(self.ip_dim), np.zeros((1, 3))))
        if(self.hdim > 1):
            for i in range(1, len(self.hdim)):
                self.params.append(np.random.rand(
                    self.hdim[i - 1], self.hdim[i]) / np.sqrt(self.hdim[-1]))
        self.params.append(
            (np.random.rand(self.hdim[-1], self.op_dim) /
             np.sqrt(self.hdim[-1]), np.zeros((1, 2))))
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

        return self.params

    def backprop(self):
        """
        Method that is used to Backpropogate the gradients

        Returns:
        """
        W = []
        b = []
        for param in self.params:
            W.append(param[0])
            b.append(param[1])
        num_examples = len(self.x)
        # delta_weight = []
        # delta_bias = []
        for i in range(500):
            # print(W1.shape)
            # print(b[0])
            activations = []
            if(self.activation_type == 'tanh'):
                activations.append(np.tanh(self.x.dot(W[0]) + b[0]))
            elif (self.activation_type == 'ReLU'):
                activations.append(np.maximum(0, (self.x.dot(W[0]) + b[0])))
            exp_scores = np.exp(activations[0].dot(W[1]) + b[1])
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            activations.append(probs)
        # Backpropagation
            delta_last = probs
            delta_last -= self.y

            dWL = (activations[0].T).dot(delta_last)
            dbL = np.sum(delta_last, axis=0, keepdims=True)
            W[1] += -self.alpha * dWL + self.lamda * W[1]
            b[1] += -self.alpha * dbL
            # delta2 = delta3.dot(W2.T) * (1 - np.power(activations[0], 2))
            if(self.activation_type == 'tanh'):
                delta_active = 1 - np.power(activations[0], 2)
            elif(self.activation_type == 'ReLU'):
                delta_active = activations[0]
                delta_active[delta_active <= 0] = 0
                delta_active[delta_active > 0] = 1
            backprop_delta = delta_last.dot(
                W[1].T) * delta_active
            delta_weight = np.dot(self.x.T, backprop_delta)
            delta_bias = np.sum(backprop_delta, axis=0)

            # break

            delta_weight += self.lamda * W[0]

            W[0] += -self.alpha * delta_weight
            b[0] += -self.alpha * delta_bias

        self.params[0] = (W[0], b[0])
        self.params[1] = (W[1], b[1])

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
        x = (x - self.mean) / self.std
        W1, b1, W2, b2 = self.params[0][0], self.params[
            0][1], self.params[1][0], self.params[1][1]
        # Forward propagation
        z1 = x.dot(W1) + b1
        if(self.activation_type == 'tanh'):
            a1 = np.tanh(z1)
        elif (self.activation_type == 'ReLU'):
            a1 = np.maximum(0, z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        print(np.argmax(probs, axis=1))
        return np.argmax(probs, axis=1)

    def add_ones_column(self, x):
        return np.concatenate((x, np.ones((len(x), 1))), axis=1)

# def get_loss():


class mlnn(xor_net):
    """
    At the moment just inheriting the network above.
    """

    def __init__(self, data, labels):
        super(mlnn, self).__init__(data, labels)


if __name__ == '__main__':
    pass
