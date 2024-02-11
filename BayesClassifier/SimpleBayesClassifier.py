import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleBayesClassifier:
    def fit_params(self, x, y, n_bins=10):
        self.pos_params = []
        self.neg_params = []
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
            x_features = x[:, col]
            pos_features = x[:, col][y==1.0]
            neg_features = x[:, col][y==0.0]

            # Remove NaN values
            x_features_no_nan = x_features[~np.isnan(x_features)]
            pos_features_no_nan = pos_features[~np.isnan(pos_features)]
            neg_features_no_nan = neg_features[~np.isnan(neg_features)]

            # Compute histograms for 'Yes' and 'No' samples
            x_unique=np.unique(x_features_no_nan)
            if len(x_unique) < n_bins:
                _, x_edges = np.histogram(x_features_no_nan, bins=len(x_unique))
                pos_hist, pos_edges = np.histogram(pos_features_no_nan, bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features_no_nan, bins=x_edges)
            else:
                _, x_edges = np.histogram(x_features_no_nan, bins=n_bins)
                pos_hist, pos_edges = np.histogram(pos_features_no_nan, bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features_no_nan, bins=x_edges)
            
            # Append histogram parameters to the respective lists
            self.pos_params.append((pos_hist, pos_edges))
            self.neg_params.append((neg_hist, neg_edges))

        return self.pos_params, self.neg_params
    
    def predict(self, x, thresh=0):
        y_pred = []
        x = np.array(x.astype(float))

        for row in range(x.shape[0]):
            log_prob_pos = 0
            log_prob_neg = 0
            features=x[row,:]
            
            for i, feature in enumerate(features):
                pos_hist, pos_edges = self.pos_params[i]
                neg_hist, neg_edges = self.neg_params[i]

                # Adding likelihood probabilities
                if not np.isnan(feature):
                    pos_bin_index = np.digitize(feature, pos_edges) - 1
                    neg_bin_index = np.digitize(feature, neg_edges) - 1
                    
                    # Handling out-of-range minimum and maximum values
                    pos_bin_index = max(min(pos_bin_index, len(pos_hist) - 1), 0)
                    neg_bin_index = max(min(neg_bin_index, len(neg_hist) - 1), 0)
                    
                    epsilon = 1e-15
                    prob_pos = pos_hist[pos_bin_index] / max(np.sum(pos_hist), epsilon)
                    prob_neg = neg_hist[neg_bin_index] / max(np.sum(neg_hist), epsilon)

                    # Setting epsilon to minimun probbabilities 
                    prob_pos = max(prob_pos, epsilon)
                    prob_neg = max(prob_neg, epsilon)

                    log_prob_pos += np.log(prob_pos)
                    log_prob_neg += np.log(prob_neg)
            
            # Adding prior probabilities
            log_prob_pos += np.log(self.prior_pos)
            log_prob_neg += np.log(self.prior_neg)
            
            # Assigning class label based on the maximum log probability
            pred = 1 if log_prob_pos-log_prob_neg > thresh else 0
            y_pred.append(pred)

        return y_pred

# model=SimpleBayesClassifier()
# model.fit_params(x_train, y_train)
# pred=np.array(model.predict(x_test))
# print(pred)
