# sample_submission.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import warnings
warnings.filterwarnings("error")


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
        # print(labels)
        self.labels = labels
        self.norm_input = False
        self.norm_type = 'None'

        """This the block to set the Parameters for the network"""
        if(self.x.shape[1] > 2):
            self.activation_type = 'ReLU'
            self.hdim = [10, 5]
            # print(self.y)
            self.epochs = 50000
            self.alpha = 0.001
            self.lamda = 0.0001
            self.loss_diff = 0.0000001
            self.min_epochs = 1000
            self.normalize_input('stand')

        else:
            self.activation_type = 'tanh'
            self.hdim = [5]
            self.epochs = 20000
            self.alpha = 0.001
            self.lamda = 0.0001
            self.loss_diff = 0.00000001
            self.min_epochs = 500
            self.normalize_input('norm')

        # self.alpha = 0.01
        # self.lamda = 0.001
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
        self.numhid = len(self.hdim)

        self.params.append(
            (np.random.rand(self.ip_dim, self.hdim[0]) / np.sqrt(self.ip_dim), np.zeros((1, self.hdim[0]))))
        if(self.hdim > 1):
            for i in range(1, len(self.hdim)):
                self.params.append((np.random.rand(
                    self.hdim[i - 1], self.hdim[i]) / np.sqrt(self.hdim[i - 1]), np.zeros((1, self.hdim[i]))))
        self.params.append(
            (np.random.rand(self.hdim[-1], self.op_dim) /
             np.sqrt(self.hdim[-1]), np.zeros((1, self.op_dim))))

        # print(self.params)
        # print(self.params[0][0].shape)
        self.backprop()
        # plt.plot(self.loss_buffer)
        # plt.show()

        # print(self.params)
        # predictions = self.get_predictions(self.x)
        # acc = (np.sum(np.asarray(predictions == labels, dtype='int'),
        #               axis=0) / float(labels.shape[0])) * 100
        # print("Training Set Accuracy ::" + str(acc))

    def normalize_input(self, norm_type, **kwargs):
        """
        This method normalizes the given array, It normalizes the training data if there is no argument test 

        Args:
            test: This argument is used to tell the function that we are not normalizing traning data.
                   This function only cares wether you used this argument or not, value of this argument doesn't matter
            x: This contains the data to be normalized, typiclaly it has the same number of columns as the training data.
               To Normalize the data in this variable test paramenter should be passes failing to do that will make this data invisible.
        """
        if('test' in kwargs):
            if(self.norm_type == 'norm'):
                return (kwargs['x'] - self.min_input) / (self.max_input - self.min_input)
            elif(self.norm_type == 'stand'):
                return (kwargs['x'] - self.input_mean) / self.input_std
            else:
                return kwargs['x']

        if(norm_type == 'norm'):
            self.norm_input = True
            self.norm_type = 'norm'
            self.max_input = np.max(self.x)
            self.min_input = np.min(self.x)
            self.x = (self.x - self.min_input) / \
                (self.max_input - self.min_input)
        elif (norm_type == 'stand'):
            self.norm_input = True
            self.norm_type = 'stand'
            self.input_mean = self.x.mean()
            self.input_std = self.x.std()
            self.x = (self.x - self.input_mean) / self.input_std

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

        """
        W = []
        b = []
        W_prev = []
        b_prev = []
        loss_ideal = sys.float_info.max
        for param in self.params:
            W.append(param[0])
            b.append(param[1])
        num_examples = len(self.x)
        # delta_weight = []
        # delta_bias = []
        prev_loss = 0
        for i in range(self.epochs):

            # print(W[0].shape)
            # print(b[0])
            activations = []
            # activations.append(self.x)
            h = (self.x.dot(W[0]) + b[0])
            for i in range(1, len(self.hdim) + 1):
                activations.append(
                    self.get_activation_value(h, self.activation_type))
                # if(self.activation_type == 'tanh'):
                #     activations.append(np.tanh(h))
                # elif (self.activation_type == 'ReLU'):
                #     activations.append(np.maximum(0, h))
                h = activations[i - 1].dot(W[i]) + b[i]
            try:
                exp_scores = np.exp(h)
            except RuntimeWarning:
                W = W_prev
                b = b_prev
                print('in Catch')
                self.params = zip(W, b)
                return
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            predictions = np.argmax(probs)

            loss = self.get_loss(W, b, predictions=predictions)

            # print("Loss ::" + str(loss))
            # print("Loss Difference ::" + str((prev_loss - loss)))
            if((np.abs(prev_loss - loss) < self.loss_diff) and i > self.min_epochs):
                break
            prev_loss = loss
            # activations.append(probs)
            self.loss_buffer.append(np.abs(loss))

            W_prev = copy.deepcopy(W)
            b_prev = copy.deepcopy(b)

            # Backpropagation
            delta = probs - self.y
            n_hid = len(self.hdim)
            for i in range(len(self.hdim)):
                cur_layer = n_hid - i
                delta_weight = (activations[cur_layer - 1].T).dot(delta)
                try:
                    # delta_weight = (
                    # delta_weight - delta_weight.mean()) / delta_weight.std()
                    delta_bias = np.sum(delta, axis=0, keepdims=True)
                    # delta_bias = (delta_bias - delta_bias.mean()) / \
                    #     delta_bias.std()
                except RuntimeWarning:
                    W = W_prev
                    b = b_prev
                    print('in Catch')
                    self.params = zip(W, b)
                    return
                delta_weight += self.lamda * W[cur_layer]
                W[cur_layer] = W[cur_layer] - self.alpha * delta_weight
                b[cur_layer] = b[cur_layer] - self.alpha * delta_bias
                delta_active = self.get_activation_gradient(
                    activations[cur_layer - 1], self.activation_type)
                # if(self.activation_type == 'tanh'):
                #     delta_active = 1 - np.power(activations[cur_layer - 1], 2)
                # elif(self.activation_type == 'ReLU'):
                #     delta_active = activations[cur_layer - 1]
                #     # print(delta_active)
                #     delta_active[delta_active <= 0] = 0
                #     delta_active[delta_active > 0] = 1
                delta = delta.dot(
                    W[cur_layer].T) * delta_active

            try:
                delta_weight = np.dot(self.x.T, delta)
                # delta_weight = (delta_weight - delta_weight.mean()
                #                 ) / delta_weight.std()

                delta_bias = np.sum(delta, axis=0)
                # delta_bias = (delta_bias - delta_bias.mean()) / \
                # delta_bias.std()
            except RuntimeWarning:
                W = W_prev
                b = b_prev
                print('in Catch')
                self.params = zip(W, b)
                return
            # break
            delta_weight += self.lamda * W[0]
            W[0] = W[0] - self.alpha * delta_weight
            b[0] = b[0] - self.alpha * delta_bias
        loss = self.get_loss(W, b)

        # if(loss > loss_ideal):
        #     W = W_ideal
        #     b = b_ideal

        self.params = zip(W, b)

    def get_activation_gradient(self, h, activation_type):
        """
        This Method calculates the gradiant of the activcations for the nodes actuvation and activation type for the layer

        Args:
            h: h is the activation of the current layer calculated in the forward pass.
            activation_type: it is the type of the activation function used in the layer.
                            Available activation types are tanh, ReLU and sigmoid

        Returns:
            numpy.ndarray: the value after applying the gradiant to the activation
        """
        if(activation_type == 'tanh'):
            return 1 - np.power(h, 2)
        elif(activation_type == 'ReLU'):
            h[h > 0] = 1
            h[h < 0] = 0
            h[h == 0] = 0
            return h
        elif(activation_type == 'sigmoid'):
            return h * (1 - h)

    def get_activation_value(self, h, activation_type):
        """
        This Method calculates the activcations for the nodes given the inputs and activation type for the activations

        Args:
            h: h is the input for the activation, it is the value of inputs to layer after multiplying  weights and adding bias
            activation_type: it is the type of the activation function used in the layer. 
                            Available activation types are tanh, ReLU and sigmoid 

        Returns:
            numpy.ndarray: the value after applying the activation 
        """

        if(activation_type == 'tanh'):
            return np.tanh(h)
        elif(activation_type == 'ReLU'):
            return np.maximum(0, h)
        elif(activation_type == 'sigmoid'):
            n = 1 / (1 + np.exp(-1 * h))
            return h

    def get_loss(self, W, b, ** kwargs):
        """
        This method calculates the Training error

        Args:
            W: Weights to use for the prediction
            b: Biases to use for prediction
            predictions: Pedictions to be used to calculate the loss. 
                            If no predictions are provided this function uses the provided Weights W and biases b and the training set self.x 
                            to calculate the predictions

        Returns:
            int: loss is an int value of the sum of cross etropy and l2 regularization loss calculated using the predictions of the model
        """
        predictions = []
        if('predictions' in kwargs):
            predictions = kwargs['predictions']
        else:
            predictions = self.predict(self.x, weights=W, biases=b)

        loss = np.sum(self.labels * predictions)
        reg = np.sum([np.sum(np.square(weight))
                      for weight in W]) * self.lamda / 2
        loss += reg
        return loss / len(self.x)

    def predict(self, x, **kwargs):
        """
        Method predicts the values for model given weights and biases

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
            weights: Weights to use for the prediction, If this parameter is not provided then the default weights are used
            biases: Biases to use for prediction, If this parameter is not provided then the default biases are used.

        Returns:
            list: This method returns an array of predictions for the given data. Length of y equals to the number of samples in the input x
        """
        if(self.norm_type == 'stand'):
            x = (x - self.input_mean) / self.input_std
        elif(self.norm_type == 'norm'):
            x = (x - self.min_input) / (self.max_input - self.min_input)
        W = []
        b = []
        if('weights' in kwargs):
            W = kwargs['weights']
        else:
            W = []
            for param in self.params:
                W.append(param[0])
        if('biases' in kwargs):
            b = kwargs['biases']
        else:
            b = []
            for param in self.params:
                b.append(param[1])

        # Forward propagation
        h = x.dot(W[0]) + b[0]
        for i in range(1, len(self.hdim) + 1):
            act = self.get_activation_value(h, self.activation_type)
            # if(self.activation_type == 'tanh'):
            #     act = np.tanh(h)
            # elif (self.activation_type == 'ReLU'):
            #     act = np.maximum(0, h)
            h = act.dot(W[i]) + b[i]
        try:
            np.exp(h)
        except RuntimeWarning:
            h /= len(h)
        act = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True)
        # print(np.argmax(act, axis=1))
        return np.argmax(act, axis=1)

    def get_predictions(self, x, **kwargs):
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
        y = self.predict(x)
        print(y)
        return y

    def add_ones_column(self, x):
        """
        This method adds a column of one at the end of a given matrix

        Args:
            X: 2D Matrix with integers

        Returns:
            numpy.ndarray: returns a matrix with a column of 1's added at the end.        """
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
