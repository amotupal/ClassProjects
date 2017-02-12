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
        self.params = []  # [(w,b),(w,b)]
        self.ip_dim = data.shape[1]
        if(len(labels.shape) > 1):
            self.op_dim = labels.shape[1]
        else:
            self.op_dim = 1
        self.hdim = [3]
        self.numhid = len(self.hdim)
        self.x = np.concatenate(self.x, np.ones((len(self.x), 1)))
        self.params.append(np.random.rand(self.hdim[0] + 1, self.ip_dim))
        if(np.hdim > 1):
            for i in range(1, self.hdim):
                self.params.append(np.random.rand(
                    self.hdim[i] + 1, self.ip_dim))

        print(self.params)

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

        return np.random.randint(low=0, high=2, size=x.shape[0])


class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """

    def __init__(self, data, labels):
        super(mlnn, self).__init__(data, labels)


if __name__ == '__main__':
    pass
