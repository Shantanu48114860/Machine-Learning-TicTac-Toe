import numpy as np


class Linear_regressor:
    def initialize(self):
        theta = np.random.randn(9) * 0.01
        b = np.zeros((9, 1))

        parameters = {
            "theta": theta,
            "b": b
        }
        return parameters

    def forward_prop(self, X_train, parameters):
        theta = parameters["theta"]
        b = parameters["b"]

        A0 = X_train
        Y_hat = theta @ A0 + b
        return Y_hat

    def back_prop(self, Y_hat, parameters, X, Y):
        theta = parameters["theta"]
        b = parameters["b"]

        m = X.shape[1]
        diff_Y = Y_hat - Y

        d_theta = (1 / m) * (diff_Y @ X.T)
        d_b = (1 / m) * (np.sum(diff_Y, axis=1, keepdims=True))

        grads = {
            "d_theta": d_theta,
            "d_b": d_b
        }
        return grads

    def calculate_cost(self, Y_hat, Y):
        m = Y.shape[1]
        cost = ((1) / m) * np.sum((Y_hat - Y) ** 2)

        return cost

    def update_parameters(self, grads, learning_rate, parameters):
        d_theta = grads["d_theta"]
        d_b = grads["d_b"]

        theta = parameters["theta"]
        b = parameters["b"]

        theta = theta - (learning_rate * d_theta)
        b = b - (learning_rate * d_b)

        parameters = {
            "theta": theta,
            "b": b
        }
        return parameters

    def model(self, X_train, Y_train, num_iterations=100, learning_rate=0.01, print_cost=False):
        # layer sizes
        n_x = X_train.shape[0]
        n_y = Y_train.shape[0]

        # initialize parameters
        parameters = self.initialize()
        costs = []
        for i in range(0, num_iterations):
            # forward prop
            Y_hat = self.forward_prop(X_train, parameters)

            # cost function
            cost = self.calculate_cost(Y_hat, Y_train)
            costs.append(cost)

            # back prop
            grads = self.back_prop(Y_hat, parameters, X_train, Y_train)

            # update param
            parameters = self.update_parameters(grads, learning_rate, parameters)

            if print_cost and i % 50 == 0:
                print("Cost after iteration {0}: {1}".format(i, cost))

        return parameters, costs

    def predict(self, X, parameters):
        Y_hat = self.forward_prop(X, parameters)
        return Y_hat

    def do_lin_reg(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        np_X_train = np_X_train.T
        np_Y_train = np_Y_train.T

        np_X_test = np_X_test.T
        np_Y_test = np_Y_test.T

        learning_rate = 0.001
        parameters, costs = self.model(np_X_train, np_Y_train, num_iterations=500,
                                       learning_rate=learning_rate,
                                       print_cost=True)

        print(parameters["theta"].shape)
        y_preds_test = self.predict(np_X_test, parameters)
        print("Y_true shape: {0}".format(np_Y_test.shape))
        print("Y_pred shape: {0}".format(y_preds_test.shape))

        print(np_Y_test[0])
        print(y_preds_test[0])
