import numpy as np

# helper functions
def sigmoid(x, c1=1, c2=0):
    x = np.float64(x)

    # approximate to the value since float can't handle
    if c1*(x-c2) > 100:
        return 1
    elif c1*(x-c2) < -100:
        return 0
    return 1 / (1 + np.exp(-c1*(x-c2)))

def to2D(arr):
    return arr.reshape(-1, 1) if arr.ndim < 2 else arr

# Defining ELM
class ELMclf:
    """
    Random initialisation of input weights and biases are removed, but can be inserted manually.
    Defaults sigmoid activation function
    FIXME: currently it seems like y must be 2D, consider whether to include in doc or change code
    :param n: hidden layer neurons size
    :var weights_i: array of input weights for each neuron, in range [0,1]
    :var weights_o: array of output weights for each output node
    :var biases: array of biases (n biases) for neuron, no bias for output nodes
    """
    def __init__(self, n, random_state=None):
        self.n = n  # number of hidden nodes, N_tilde
        self.weights_i = None  # input weights, W
        self.weights_o = None  # hidden layer weight, beta
        self.biases = None  # biases for input_weights, b
        self.activation = sigmoid
        # Currently default using sigmoid function
        self.hidden_mat = None

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y, weights_i=None, biases=None):
        if weights_i is not None:
            self.weights_i = weights_i
        else:
            self.weights_i = self._init_input_weights(X.shape[1])
        if biases is not None:
            self.biases = biases
        else:
            self.biases = self._init_biases()

        H = self._obtain_H(X)
        self.hidden_mat = H

        beta = self._obtain_beta(y, H)
        self.weights_o = beta
        return self

    def predict(self, X):
        out_net = np.matmul(self._obtain_H(X),self.weights_o)
        out_threshold = out_net

        # this assumes binary label of 0 and 1
        out_threshold = np.vectorize(self.activation)(out_threshold, 1, 0.5) # for sigmoid
        out_threshold[out_threshold > 0.5] = 1
        out_threshold[out_threshold <= 0.5] = 0
        # out_threshold[out_threshold == -0] = 0 # convert all zeros to +0
        return out_threshold

    def predict_proba(self, X):
        # only one output node for binary
        out_net = np.matmul(self._obtain_H(X),self.weights_o)
        out_threshold = out_net
        out_threshold[out_net <= 0.5] = 1 - out_threshold[out_net <= 0.5]
        return out_threshold

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == to2D(y))


    def _init_input_weights(self, x_size):
        """
        :param x_size: input nodes size (uses feature size)
        :returns: input_weights
        """
        # rng = np.random.default_rng(self.random_state)
        input_weights = self.rng.random((self.n, x_size))
        return input_weights

    def _init_biases(self):
        """
        :returns: biases
        """
        # rng = np.random.default_rng(self.random_state)
        bias_weights = self.rng.random(self.n)
        return bias_weights

        # random.seed()  # reset seed after operation

    def _obtain_H(self, x_train):
        """
        calculate the hidden layer output matrix, g(w.x+b) for size N x N_tilde
        :param x_train: instances to be trained, expects x_train of 'x_size' number of features
        :return: hidden layer output matrix, H
        """
        n_samples = x_train.shape[0]  # N
        n_features = x_train.shape[1] # x_size of init_weights
        expected_n_features = self.weights_i.shape[1] # previously declared
        if n_features != expected_n_features:
            raise Exception(f'Number of features does not match previously declared value: {expected_n_features}')

        # First substep of H: w.x
        f = lambda i,j: np.dot(self.weights_i[int(j)], x_train[int(i)])
        H = np.fromfunction(np.vectorize(f), (n_samples, self.n), dtype=self.weights_i.dtype)
        # print(H.shape)

        # Second substep: w.x + b
        H += self.biases

        # Finally, apply activation function, g(w.x + b)
        H = np.vectorize(self.activation)(H)
        # self.hidden_mat = H # set at fit
        return H

    @staticmethod
    def _obtain_beta(y_train, H):
        """
        Getting output weights (which is beta), using Singular Value Decomposition
        :param y_train: or targets to match each x_train, expects n_sample of 0 and 1 for binary classification
        :param H: the hidden layer output matrix
        :return: output weights, which is also stored in object
        """
        # Error handling (>2D array is not handled)
        n_samples = y_train.shape[0]
        expected_n_samples = H.shape[0]
        if n_samples != expected_n_samples:
            raise Exception(f'Number of samples does not match previously given x_train: {expected_n_samples}')

        # Update y_train if it is 1D to fit the formula
        y_train_dup = to2D(y_train)

        # Obtain Moore-Penrose generalised inverse
        H_plus = np.linalg.pinv(H)

        # Calculate output weights
        output_weights = np.matmul(H_plus, y_train_dup)
        # self.weights_o = output_weights # set at fit
        return output_weights
