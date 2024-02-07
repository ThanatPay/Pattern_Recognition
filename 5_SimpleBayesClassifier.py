import numpy as np
import pandas as pd

class SimpleBayesClassifier:
    def __init__(self, n_pos, n_neg):
        self.n_pos = n_pos
        self.n_neg = n_neg
        total_samples = n_pos + n_neg
        self.prior_pos = n_pos / total_samples
        self.prior_neg = n_neg / total_samples

    def fit_params(self, x, y, n_bins=10):
        self.pos_params = []
        self.neg_params = []


        for col in range(x.shape[1]):
            # Extract features for 'Yes' and 'No' samples
            x_features = x.iloc[:, col]
            pos_features = x.iloc[:, col][y==1.0]
            neg_features = x.iloc[:, col][y==0.0]
            
            x_unique=np.unique(x_features[x_features.notna()])
            if len(x_unique) < n_bins:
                _, x_edges = np.histogram(x_features[x_features.notna()], bins=len(x_unique))
                pos_hist, pos_edges = np.histogram(pos_features[pos_features.notna()], bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features[neg_features.notna()], bins=x_edges)

            else:
                # Compute histograms for 'stay' and 'leave' samples
                _, x_edges = np.histogram(x_features[x_features.notna()], bins=n_bins)
                pos_hist, pos_edges = np.histogram(pos_features[pos_features.notna()], bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features[neg_features.notna()], bins=x_edges)
            
            # Append histogram parameters to the respective lists
            self.pos_params.append((pos_hist, pos_edges))
            self.neg_params.append((neg_hist, neg_edges))

        return self.pos_params, self.neg_params

    def predict(self, x, thresh=0):
        y_pred = []

        for row in range(x.shape[0]):
            log_prob_pos = 0
            log_prob_neg = 0
            features=x.iloc[row,:].astype(float)
            print("------------------------------")
            
            for i, feature in enumerate(features):
                pos_hist, pos_edges = self.pos_params[i]
                neg_hist, neg_edges = self.neg_params[i]
                print(pos_hist,pos_edges)
                print(neg_hist, neg_edges)
                print(feature, np.isnan(feature))

                if not np.isnan(feature):
                    pos_bin_index = np.digitize(feature, pos_edges.astype(float)) - 1
                    neg_bin_index = np.digitize(feature, neg_edges.astype(float)) - 1
                    
                    # Handling out-of-range indices
                    pos_bin_index = max(min(pos_bin_index, len(pos_hist) - 1), 0)
                    neg_bin_index = max(min(neg_bin_index, len(neg_hist) - 1), 0)
                    
                    epsilon = 1e-9
                    prob_pos = pos_hist[pos_bin_index] / (np.sum(pos_hist)+epsilon)
                    prob_neg = neg_hist[neg_bin_index] / (np.sum(neg_hist)+epsilon)

                    prob_pos = prob_pos if prob_pos > 0 else epsilon
                    prob_neg = prob_neg if prob_neg > 0 else epsilon
                    print(prob_pos)
                    print(prob_neg)
    
                    log_prob_pos += np.log(prob_pos)
                    log_prob_neg += np.log(prob_neg)
                print("---------------")
            
            # Adding prior probabilities
            log_prob_pos += np.log(self.prior_pos)
            log_prob_neg += np.log(self.prior_neg)
            
            # Assigning class label based on the maximum log probability
            pred = 1.0 if log_prob_pos > log_prob_neg else 0.0
            y_pred.append(pred)
            print(log_prob_pos, log_prob_neg)
            print(pred)

        return y_pred


n_bins=3
data = pd.read_csv("hr-employee-attrition-with-null.csv")
data.loc[data["Attrition"] == "No", "Attrition"] = 0.0
data.loc[data["Attrition"] == "Yes", "Attrition"] = 1.0
data.loc[data["BusinessTravel"] == "Travel_Rarely", "BusinessTravel"] = 0.0
data.loc[data["BusinessTravel"] == "Travel_Frequently", "BusinessTravel"] = 1.0
data.loc[data["Department"] == "Research & Development", "Department"] = 0.0
x=data.head().iloc[:, 3:7]
y=data.head().iloc[:, 2]
print(x)
print(y)
model=SimpleBayesClassifier(2,3)
model.fit_params(x,y,n_bins=3)
print('predict:',model.predict(x))


