import numpy as np
import matplotlib.pyplot as plt

# Hint: You can use this function to get gaussian distribution.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, mixture_weight, mean_params, cov_params):
        self.mixture_weight = mixture_weight
        self.mean_params = mean_params
        self.cov_params = cov_params
        self.n_iter = 0

    def estimation_step(self, data):
        num_samples = data.shape[0]
        num_components = self.mixture_weight.shape[0]

        # Compute the responsibility matrix
        w = np.zeros((num_samples, num_components))
        for j in range(num_components):
            w[:, j] = self.mixture_weight[j] * multivariate_normal.pdf(data, mean=self.mean_params[j], cov=self.cov_params[j])
        w /= np.sum(w, axis=1, keepdims=True)

        return w


    def maximization_step(self, data, w):
        num_components = self.mixture_weight.shape[0]

        # Update mixture weights
        self.mixture_weight = np.mean(w, axis=0)

        # Update means
        for j in range(num_components):
            self.mean_params[j] = np.sum(w[:, j].reshape(-1, 1) * data, axis=0) / np.sum(w[:, j])

        # Update covariances
        for j in range(num_components):
            diff = data - self.mean_params[j]
            cov_param = np.dot((w[:, j].reshape(-1, 1) * diff).T, diff) / np.sum(w[:, j])
            cov_param = np.diag(np.diag(cov_param))
            self.cov_params[j] = cov_param

    def get_log_likelihood(self, data):
        log_prob = 0
        num_samples = data.shape[0]
        num_components = self.mixture_weight.shape[0]

        for i in range(num_samples):
            prob_i = 0
            for j in range(num_components):
                prob_i += self.mixture_weight[j] * multivariate_normal.pdf(data[i], mean=self.mean_params[j], cov=self.cov_params[j])
            log_prob += np.log(prob_i)

        return log_prob

    def print_iteration(self):
        print("m :\n", self.mixture_weight)
        print("mu :\n", self.mean_params)
        print("covariance matrix :\n", self.cov_params)
        print("-------------------------------------------------------------")

    def perform_em_iterations(self, data, num_iterations, display=True):
        log_prob_list = []

        # Display initialization.
        if display:
            print("Initialization")
            self.print_iteration()

        for n_iter in range(1,num_iterations+1):

            # Estimation step
            w = self.estimation_step(data)

            # Maximization step
            self.maximization_step(data, w)

            # Calculate log likelihood
            log_prob = self.get_log_likelihood(data)
            log_prob_list.append(log_prob)

            # Display each iteration.
            if display:
                print(f"Iteration: {n_iter}")
                self.print_iteration()

        return log_prob_list
    
    def predict(self, data):
        # Compute the responsibility matrix
        w = self.estimation_step(data)

        # Assign each data point to the component with the highest responsibility
        predictions = np.argmax(w, axis=1)

        return predictions

num_iterations = 5
num_mixture = 3
mixture_weight = np.array([1] * num_mixture) # m
mean_params = np.array([[3,3], [2,2], [-3,-3]], dtype = float)
cov_params = np.array([np.eye(2)] * num_mixture)
data=np.array([[1,2],[3,3],[2,2],[8,8],[6,6],[7,7],[-3,-3],[-2,-4],[-7,-7]])

# gmm = GMM(mixture_weight, mean_params, cov_params)
# log_prob_list = gmm.perform_em_iterations(data, num_iterations, display=False)
# plt.plot(range(1,len(log_prob_list)+1), log_prob_list)
# plt.show()

# cluster = gmm.predict(data)
# plt.scatter(data[:,0], data[:,1], c=cluster)
# plt.show()
