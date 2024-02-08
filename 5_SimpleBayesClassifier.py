import numpy as np
import pandas as pd

class SimpleBayesClassifier:
    def fit_params(self, x, y, n_bins=10):
        self.pos_params = []
        self.neg_params = []
        x = np.array(x.astype(float))
        y = np.array(y.astype(float))

        # calculater prior probabilities
        self.n_pos = len(x[y==1.0])
        self.n_neg = len(x[y==0.0])
        self.total_samples = self.n_pos + self.n_neg
        self.prior_pos = self.n_pos / self.total_samples
        self.prior_neg = self.n_neg / self.total_samples
        # print(self.prior_pos, self.prior_neg)
        # print("------------------------------")

        for col in range(x.shape[1]):
            # Extract features for 'Yes' and 'No' samples
            x_features = x[:, col]
            pos_features = x[:, col][y==1.0]
            neg_features = x[:, col][y==0.0]
             
            x_unique=np.unique(x_features[pd.notna(x_features)])
            # Compute histograms for 'Yes' and 'No' samples
            if len(x_unique) < n_bins:
                _, x_edges = np.histogram(x_features[pd.notna(x_features)], bins=len(x_unique))
                pos_hist, pos_edges = np.histogram(pos_features[pd.notna(pos_features)], bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features[pd.notna(neg_features)], bins=x_edges)

            else:
                _, x_edges = np.histogram(x_features[pd.notna(x_features)], bins=n_bins)
                pos_hist, pos_edges = np.histogram(pos_features[pd.notna(pos_features)], bins=x_edges)
                neg_hist, neg_edges = np.histogram(neg_features[pd.notna(neg_features)], bins=x_edges)
            
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

                if not np.isnan(feature):
                    pos_bin_index = np.digitize(feature, pos_edges) - 1
                    neg_bin_index = np.digitize(feature, neg_edges) - 1
                    
                    # Handling out-of-range indices
                    pos_bin_index = max(min(pos_bin_index, len(pos_hist) - 1), 0)
                    neg_bin_index = max(min(neg_bin_index, len(neg_hist) - 1), 0)
                    
                    epsilon = 1e-15
                    prob_pos = pos_hist[pos_bin_index] / (np.sum(pos_hist) + epsilon)
                    prob_neg = neg_hist[neg_bin_index] / (np.sum(neg_hist) + epsilon)

                    prob_pos = prob_pos if prob_pos > 0 else epsilon
                    prob_neg = prob_neg if prob_neg > 0 else epsilon

                    log_prob_pos += np.log(prob_pos)
                    log_prob_neg += np.log(prob_neg)
            
            # Adding prior probabilities
            log_prob_pos += np.log(self.prior_pos)
            log_prob_neg += np.log(self.prior_neg)
            
            # Assigning class label based on the maximum log probability
            pred = 1 if np.exp(log_prob_pos) > thresh and log_prob_pos > log_prob_neg else 0
            y_pred.append(pred)
            # print(np.exp(log_prob_pos), np.exp(log_prob_neg))
            # print(pred)
            # print("------------------------------")

        return y_pred

n_bins=3
data = pd.read_csv("hr-employee-attrition-with-null.csv")
data.loc[data["Attrition"] == "No", "Attrition"] = 0.0
data.loc[data["Attrition"] == "Yes", "Attrition"] = 1.0
data.loc[data["BusinessTravel"] == "Travel_Rarely", "BusinessTravel"] = 0.0
data.loc[data["BusinessTravel"] == "Travel_Frequently", "BusinessTravel"] = 1.0
data.loc[data["BusinessTravel"] == "Non-Travel", "BusinessTravel"] = 2.0
data.loc[data["Department"] == "Research & Development", "Department"] = 0.0
data.loc[data["Department"] == "Sales", "Department"] = 1.0
data.loc[data["Department"] == "Human Resources", "Department"] = 2.0
x_train=data.iloc[70:, 3:7]
y_train=data.iloc[70:, 2]
x_test=data.head(70).iloc[:, 3:7]
y_test=data.head(70).iloc[:, 2]
model=SimpleBayesClassifier()
model.fit_params(x_train,y_train)
pred=model.predict(x_test)
print('predict:',pred)
print('accuracy:',np.mean(pred==y_test))


