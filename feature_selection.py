from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from itertools import combinations
import numpy as np

class BFS(object):
    """
        A greedy search to identify the feature subset resulting in best f1 score for a given classifier
        """
    def __init__(self,clf, dataset, features, min_subset):
        self.dataset = dataset #final_dataset (dict)
        self.features = features #list of feature names to specify which features to use in the selection process
        self.indices = range(1, len(self.features)) #omit poi feature
        self.min_subset = min_subset
        self.clf = clf
    
    
    def fit(self,feature_set):
        """
        The dataset is randomly split into into 20 folds that have preserved the class label percentage as
        present in the dataset. Each fold is then subjected to validation of a classifier performance using f1 
        scoring.
        The mean f1 score (over all folds) is returned.
        
        feature_set: feature dataset (array)
        """
        data = featureFormat(self.dataset, feature_set)
        labels, features = targetFeatureSplit(data)
        cv = StratifiedShuffleSplit(labels, 20, random_state = 42)
        
        # calculate score means over all cv folds
        scores = []
        for train_idx, test_idx in cv:
            features_train = []
            features_test  = []
            labels_train   = []
            labels_test    = []
            for ii in train_idx:
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
                labels_test.append( labels[jj])
            
            score = self.scoringFunction(features_train,labels_train,features_test,labels_test)
            scores.append(score)
        
        return np.mean(scores)
    
    def search(self):
        """
        For a given feature set and dimension (number of features) every possible combination of features
        is forwarded to training and testing of a classifier. For each feature dimension the best subset
        (combination of features) and the respective score (f1_score) are stored in a dictionary (object variable).
        Starting with the highest dimension (all features) training and testing is performed for all feature
        combination up to a specified dimension (specified by the user; min_subset argument).
        """
        
        self.summary = []
        dim = len(self.indices)
        score = self.fit(self.features)
        result = {"best_subset_{}".format(dim) : self.indices, "score" : score}
        self.summary.append(result)
        
        while self.min_subset < dim:
            #print dim-1
            # store information for each subset
            scores_subset = []
            features_subset = []
            indices_subset = []
            
            for subset in combinations(self.indices, dim-1):
                
                feature_set = list(np.array(self.features)[[subset]]) # convert back to list to insert poi back in set
                feature_set.insert(0,"poi")
                
                score = self.fit(feature_set)
                scores_subset.append(score)
                features_subset.append(subset)
            
            best_idx = np.argmax(scores_subset)
            best_subset = features_subset[best_idx]
            result = {"best_subset_{}".format(dim-1) : best_subset, "score" : scores_subset[best_idx]}
            self.summary.append(result)
            dim -= 1

    def scoringFunction(self,features_train,labels_train,features_test,labels_test):
        """
        Trains a classifier using a trainig dataset and cognate labels. Evaluates the classifier's performance by
        estimating the f1 score metric using a test dataset (features and labels). Returns the f1 score
        """
        self.clf.fit(features_train, labels_train)
        predicted = self.clf.predict(features_test)
        return f1_score(labels_test, predicted)