import numpy as np

# helper functions
# def sigmoid(x:float, c1:float=1, c2:float=0) -> int:
#     x = np.float64(x)
#
#     # approximate to the value since float can't handle
#     if c1*(x-c2) > 100:
#         return 1
#     elif c1*(x-c2) < -100:
#         return 0
#     return 1 / (1 + np.exp(-c1*(x-c2)))

def sigmoid(x, c1=1, c2=0):
    return np.divide(1, np.add(1, np.exp(np.multiply(-c1, np.subtract(x, c2)))))
def to2D(arr:np.ndarray):
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
    def __init__(self, n:int, random_state:int=None):
        self.n = n  # number of hidden nodes, N_tilde
        self.weights_i = None  # input weights, W
        self.weights_o = None  # hidden layer weight, beta
        self.biases = None  # biases for input_weights, b
        self.l2_param = None
        self.activation = lambda x: sigmoid(x, 1, 0)
        # Currently default using sigmoid function
        self.hidden_mat = None

        self.random_state = random_state
        self.rng = None # update on fit

    def fit(self, X:np.ndarray, y:np.ndarray, l2_param:float=0, weights_i:np.ndarray=None, biases:np.ndarray=None):
        # ensure that rng is reset for every new fit
        weights_changed = weights_i is not None and np.any(weights_i != self.weights_i)
        biases_changed = biases is not None and np.any(biases != self.biases)

        # IF nothing important changed (if no external modification), why refit
        # without this, something weird happens in the training, and I do not know why
        if self.weights_i is not None and self.biases is not None and not weights_changed and not biases_changed:
            return self
        self.rng = np.random.default_rng(self.random_state)

        if weights_changed: self.weights_i = weights_i
        else              : self.weights_i = self._init_input_weights(X.shape[1])
        if biases_changed: self.biases = biases
        else             : self.biases = self._init_biases()
        self.l2_param = l2_param

        H = self._obtain_H(X)
        self.hidden_mat = H

        beta = ELMclf._obtain_beta(y, H, l2_param)
        self.weights_o = beta
        return self

    def predict(self, X:np.ndarray):
        out_net = np.matmul(self._obtain_H(X),self.weights_o)
        out_threshold = out_net
        # print(out_net)

        # this assumes binary label of 0 and 1
        # out_threshold = np.vectorize(self.activation)(out_threshold) # for sigmoid
        out_threshold[out_threshold > 0.5] = 1
        out_threshold[out_threshold <= 0.5] = 0
        # out_threshold[out_threshold == -0] = 0 # convert all zeros to +0
        return out_threshold

    def predict_proba(self, X:np.ndarray):
        # only one output node for binary
        out_net = np.matmul(self._obtain_H(X),self.weights_o)
        out_threshold = out_net
        print(np.min(out_net),np.max(out_net))

        out_threshold = np.vectorize(self.activation)(out_threshold)  # for sigmoid
        # print(np.round(out_threshold, decimals=2))
        # print(np.max(np.round(out_threshold, decimals=2)))
        # print(np.min(np.round(out_threshold, decimals=2)))
        out_threshold[out_threshold <= 0.5] = 1 - out_threshold[out_threshold <= 0.5]
        return out_threshold

    def score(self, X:np.ndarray, y:np.ndarray):
        predictions = self.predict(X)
        return np.mean(predictions == to2D(y))


    def _init_input_weights(self, x_size:int):
        """
        :param x_size: input nodes size (uses feature size)
        :returns: input_weights
        """
        # rng = np.random.default_rng(self.random_state)
        # input_weights = self.rng.random((self.n, x_size))
        input_weights = self.rng.uniform(-1,1,size=(self.n,x_size)) # Originally using 0-1
        return input_weights

    def _init_biases(self):
        """
        :returns: biases
        """
        # rng = np.random.default_rng(self.random_state)
        # bias_weights = self.rng.random(self.n)
        bias_weights = self.rng.uniform(-1,1,size=self.n)
        return bias_weights

        # random.seed()  # reset seed after operation

    def _obtain_H(self, x_train:np.ndarray):
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

        # First substep of H: w.x + b
        H = x_train @ self.weights_i.T + self.biases

        # Finally, apply activation function, g(w.x + b)
        H = self.activation(H)
        # self.hidden_mat = H # set at fit
        return H

    def _update_H(self, x_train:np.ndarray, prev_input_weights:np.ndarray, prev_biases:np.ndarray):
        """
        ONLY WORKS IF X IS SAME (which can't be checked)
        :param x_train:
        :param prev_input_weights:
        :param prev_biases:
        :return:
        """
        n_samples = x_train.shape[0]  # N
        n_features = x_train.shape[1]  # x_size of init_weights
        expected_n_features = self.weights_i.shape[1]  # previously declared
        if n_features != expected_n_features:
            raise Exception(f'Number of features does not match previously declared value: {expected_n_features}')
        # First substep of H: w.x
        f = lambda i, j: np.dot(self.weights_i[int(j)], x_train[int(i)])
        H = np.fromfunction(np.vectorize(f), (n_samples, self.n), dtype=self.weights_i.dtype)

    @staticmethod
    def _obtain_beta(y_train:np.ndarray, H:np.ndarray, l2_param:float=0):
        """
        Getting output weights (which is beta), using Singular Value Decomposition
        If l2_regularisation parameter used (!= 0):, then only apply that
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

        # Either (H^+)T or ((lambda.I + (H^T)H)^+)(H^T)T
        if l2_param == 0:
            # Obtain Moore-Penrose generalised inverse
            H_plus = np.linalg.pinv(H)

            # Calculate output weights
            output_weights = np.matmul(H_plus, y_train_dup)
            # self.weights_o = output_weights # set at fit
        else:
            H_transpose = H.T
            # use the method that makes algo faster
            # if shorter width (H^T)(H) is larger than (H)(H^T),
            # then use (H^T)((lambda.I + H(H^T))^+)T
            if H.shape[0] < H.shape[1]:
                ridge_mat = l2_param * np.identity(H.shape[0]) + H @ H_transpose
                inv_ridge_mat = np.linalg.pinv(ridge_mat)
                output_weights = H_transpose @ inv_ridge_mat @ y_train_dup
            else:
                ridge_mat = l2_param * np.identity(H.shape[1]) + H_transpose @ H
                inv_ridge_mat = np.linalg.pinv(ridge_mat)
                output_weights = inv_ridge_mat @ H_transpose @ y_train_dup
        return output_weights
