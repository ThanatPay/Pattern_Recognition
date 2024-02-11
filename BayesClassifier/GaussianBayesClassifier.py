import numpy as np
import pandas as pd

class GaussianBayesClassifier:
    def fit_params(self, x, y, n_bins=10):
        self.gaussian_pos_params = []
        self.gaussian_neg_params = []
        # columns = x.columns
        x = np.array(x.astype(float))
        y = np.array(y.astype(float))

        # calculater prior probabilities
        self.n_pos = len(x[y==1.0])
        self.n_neg = len(x[y==0.0])
        self.total_samples = self.n_pos + self.n_neg
        self.prior_pos = self.n_pos / self.total_samples
        self.prior_neg = self.n_neg / self.total_samples

        for col in range(x.shape[1]):

            # Extract features for 'Yes' and 'No' samples
            pos_features = x[:, col][y==1.0]
            neg_features = x[:, col][y==0.0]

            # Remove NaN values
            pos_features_no_nan = pos_features[~np.isnan(pos_features)]
            neg_features_no_nan = neg_features[~np.isnan(neg_features)]
            
            # Compute mean and standard deviation for 'Yes' and 'No' samples
            pos_mean, pos_std = np.mean(pos_features_no_nan), np.std(pos_features_no_nan)
            neg_mean, neg_std = np.mean(neg_features_no_nan), np.std(neg_features_no_nan)
            
            # Append Gaussian parameters to the respective lists
            self.gaussian_pos_params.append((pos_mean, pos_std))
            self.gaussian_neg_params.append((neg_mean, neg_std))

        return self.gaussian_pos_params, self.gaussian_neg_params
    
    def predict(self, x, thresh=0):
        y_pred = []
        x = np.array(x.astype(float))

        for row in range(x.shape[0]):
            log_prob_pos = 0
            log_prob_neg = 0
            features=x[row,:]
            
            for i, feature in enumerate(features):
                pos_mean, pos_std = self.gaussian_pos_params[i]
                neg_mean, neg_std = self.gaussian_neg_params[i]
                
                # # Adding likelihood probabilities using Gaussian distribution
                if not np.isnan(feature):
                    epsilon = 1e-15
                    log_prob_pos += -0.5 * ((feature - pos_mean) / max(pos_std, epsilon)) ** 2 - np.log(max(pos_std, epsilon))
                    log_prob_neg += -0.5 * ((feature - neg_mean) / max(neg_std, epsilon)) ** 2 - np.log(max(neg_std, epsilon))
            
            # Adding prior probabilities
            log_prob_pos += np.log(self.prior_pos)
            log_prob_neg += np.log(self.prior_neg)
            
            # Assigning class label based on the maximum log probability
            pred = 1 if log_prob_pos-log_prob_neg > thresh else 0
            y_pred.append(pred)

        return y_pred
    
# model=GaussianBayesClassifier()
# model.fit_params(x_train, y_train)
# pred=np.array(model.predict(x_test))
# print(pred)