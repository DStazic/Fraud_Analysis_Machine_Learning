from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import random

class Upsampling(object):
    '''
    Adjusts stemmed emails and corresponding authors datasets to have equal class (poi vs. not poi) size by
    randomly upsampling emails and authors of the smaller sized class.
        
    emails: stemmed email dataset (list)
    email_authors: email author names dataset (list)
    '''
    
    def __init__(self,emails,email_authors,dataset):
        self.emails = emails
        self.email_authors = email_authors
        self.dataset = dataset
        self.upsampling_count = None
    
    
    def shuffle(self):
        '''
        Returns shuffeld email dataset and corresponding author and author class label (poi vs. not poi).
        '''
        authors_shuffle = pd.Series(self.email_authors).sample(frac=1) #shuffle authors and respective series index
        labels_shuffle = authors_shuffle.apply(lambda name: self.dataset["poi"].loc[name]) #get class label byshuffeled series index
        emails_shuffle = np.array(self.emails)[labels_shuffle.index.values] #get emails by shuffeled class label index
        #reset author-,class_label index to match indices of emails array
        return emails_shuffle, authors_shuffle.reset_index(drop=True), labels_shuffle.reset_index(drop=True)
    
    
    def splitClass(self,class_labels):
        '''
        takes the author class label dataset and returns a poi and not poi class label set
        '''
        self.upsampling_count = class_labels.value_counts().loc[False] #count not-poi class
        return class_labels[class_labels == True], class_labels[class_labels == False]
    
    def upsample(self):
        '''
        randomly samples the poi class label dataset with replacement. Number of samples taken is equivalent to the
        number of not poi class instances present in the main dataset. Indices of upsampled poi class label are
        used to retrieve the cognate authors and stemmed emails.
        '''
        emails, authors, class_labels = self.shuffle()
        poi_class, not_poi_class = self.splitClass(class_labels)
        # upsample class label indices; indices refer to respective author emails
        poi_class_upsampled = resample(poi_class, replace=True, n_samples = self.upsampling_count, random_state=1234)
        labels_shuffle = pd.concat([poi_class_upsampled, not_poi_class]).sample(frac=1) # reshuffle after merging
        word_data_shuffle = emails[labels_shuffle.index.values]
        authors_shuffle = authors[labels_shuffle.index.values]
        return word_data_shuffle, authors_shuffle



class TFIDFcalculator(object):
    '''
    Calculates for each employee present in the main dataset the average TFIDF values for the 5 words (based on
    the set of all words used in email communication) with the highest discriminative power between poi and not poi 
    class.
        
    authors: email author names dataset (list)
    emails: stemmed email dataset (list)
    dataset: main dataset (dataframe)
    ngram_range: ngram_range attribute of TfidfVectorizer() class
    max_df: max_df attribute of TfidfVectorizer() class
    min_df: min_df attribute of TfidfVectorizer() class
    '''
    
    def __init__(self,authors, emails, dataset, ngram_range, max_df, min_df):
        self.authors = authors
        self.emails = emails
        self.dataset = dataset
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
    
    
    def getClassLabel(self):
        '''
        Assigns class label (poi vs. not poi) to names in the author list and returns a dataframe with
        author names and corresponding class labels
        '''
        self.class_label = pd.Series(self.authors).apply(lambda name: self.dataset["poi"].loc[name])
        return pd.DataFrame({"poi":self.class_label, "name":self.authors})
    
    def calculateTFIDF(self):
        '''
        Calculates and returns TF-IDF values for the set of words present in the email dataset
        '''
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)
        return self.vectorizer.fit_transform(self.emails)
    
    
    def selectFeature(self):
        '''
        Takes TF-IDF values, calculates and returns 5 TF-IDF values with highest discriminative power with respect
        to class label
        '''
        vectorizer_fitted = self.calculateTFIDF()
        # use statistical approach to extract TF-IDF values with highest correlation to class label
        self.selector = SelectKBest(score_func=f_classif, k=5)
        return self.selector.fit_transform(vectorizer_fitted, self.class_label)
    
    def getEmployeeTFIDF(self):
        '''
        For each employee calculates the average TF-IDF values for each of the 5 selected values and returns a
        dataframe with values as columns and employee names as index
        '''
        # calculate mean TF-IDF values (for each term in bag of words) for each employee
        tfidf_by_author = dict()
        authors_df = self.getClassLabel()
        selector_fitted = self.selectFeature()
        authors_idx = authors_df.groupby("name").groups
        for name in authors_idx.keys():
            # use toarray() to transform matrix to array as result
            tfidf_by_author[name] = np.mean(selector_fitted[authors_idx[name]].toarray(), axis = 0)
        # create a dataframe with employee names as indices and selected mean TF-IDF values as features
        return pd.DataFrame.from_dict(tfidf_by_author, orient = "index")
    
    
    def createFeature(self):
        '''
        Updates the TF-IDF dataframe obtained via getEmployeeTFIDF() method to contain NaN as TF-IDF values for
        employees without any email data. Returns updated dataframe
        '''
        employees_present = self.getEmployeeTFIDF()
        # for employees without any email data, define NaN TF-IDFs
        employees_missing = dict()
        for name in self.dataset.index:
            if name not in employees_present.index:
                employees_missing[name]=[np.nan]*5
        
        # transform dict into dataframe and combine with dataframe that stores calculated TF-IDF features
        employees_missing_df = pd.DataFrame.from_dict(employees_missing, orient = "index")
        return pd.concat([employees_missing_df, employees_present])

    def extractFeature(self):
        '''
        Calls pipeline to retrieve the TF-IDF dataframe. Refactors the dataframe column names and also defines the
        object variable "terms" to store the stemmed words corresponding to the selected TF-IDF values
        '''
        feature_set = self.createFeature()
        self.terms = [self.vectorizer.get_feature_names()[idx] for idx in self.selector.get_support(indices=True)]
        feature_set.columns = ["word_feature_{}".format(idx+1) for idx in range(len(self.terms)) ]
        return feature_set

