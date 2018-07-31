#!/usr/bin/python

import sys
import os
sys.path.append("../tools/")
import pickle
import bz2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tester
import pandas as pd
import numpy as np



### TASK 1: SELECT FEATURES  ------------------------------------------------------------
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# On the rationale behind the final feature list see section 4.1, or the notebook for detailed explanation
final_features = ['poi','bonus', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive',
                 'freq_change_received', 'word_feature_2', 'word_feature_3']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# transform data dictionary to pandas data frame (format used for wrangling and manipulation of data)
data_df = pd.DataFrame.from_dict(data_dict, orient = "index")
# copy of dataframe (manipulation of data_df has no effect on default_dataset)
# default dataset used in downstream processing
default_dataset = pd.DataFrame(data_df)

### TASK 2.1: REMOVE OUTLIERS ------------------------------------------------------------
# no general predictive power for email feature as a feature; remove feature (column)
data_df.drop("email_address", inplace=True, axis=1)
# "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" are no valid employee names; remove name (row indices)
data_df.drop(["THE TRAVEL AGENCY IN THE PARK", "TOTAL"], inplace=True)
# extremely high proportion of missing values in following features; remove
data_df.drop(["deferral_payments", "restricted_stock_deferred", "loan_advances", "director_fees"], inplace=True,axis=1)


### TASK 2.2:IMPUTE THE DATASET ------------------------------------------------------------
## import relevant function
from imputation import MultipleImputation

# replace NaN string with true NaN; required for multiple imputation
data_df.replace("NaN", np.nan, inplace=True)

# imputation of single values
#---------------------------
#---------------------------
#use median salary for imputation of employee salaries
salary_median = np.median(data_df["salary"].dropna())
# both employees exhibit a stark deviation from the median salary (not valid as annual salaries)
data_df["salary"].loc["BANNANTINE JAMES M"] = salary_median
data_df["salary"].loc["GRAY RODNEY"] = salary_median
#update total_payments, which was wrong due to a negative value in deferral_payments
data_df["total_payments"].loc["BELFER ROBERT"] = 105785.0
# correct negative value in restricted_stock for BHATNAGAR SANJAY
data_df["restricted_stock"].loc["BHATNAGAR SANJAY"] = 2.60449e+06
# correct negative value in total_stock_value for BELFER ROBERT
data_df["total_stock_value"].loc["BELFER ROBERT"] = 44093.0

# multiple imputation on entire dataset
#---------------------------
#---------------------------
#split the dataset according to class affiliation
poiYes_data = data_df[data_df["poi"] == True]
poiNo_data = data_df[data_df["poi"] == False]

# split the feature set into email and financial features
# all email features
email_features = set(["to_messages", "shared_receipt_with_poi", "from_messages",
                      "from_this_person_to_poi", "from_poi_to_this_person"])

# all financial status-relevant features (remove poi feature)
financial_features = set(data_df.columns) - email_features
financial_features.remove("poi")

# perform multiple imputation on the poi subset
financial_features_poi_imputed = MultipleImputation(poiYes_data, financial_features)
email_features_poi_imputed = MultipleImputation(poiYes_data,  email_features)
# merge imputet poi datasets and add the poi feature
poi_data_filled = pd.merge(financial_features_poi_imputed, email_features_poi_imputed, left_index=True, right_index=True)
poi_data_filled["poi"] = data_df["poi"]

# perform multiple imputation on the not-poi subset
financial_features_not_poi_imputed = MultipleImputation(poiNo_data, financial_features)
email_features_not_poi_imputed = MultipleImputation(poiNo_data, email_features)
# merge imputet not-poi datasets and add the poi feature
not_poi_data_filled = pd.merge(financial_features_not_poi_imputed, email_features_not_poi_imputed, left_index=True, right_index=True)
not_poi_data_filled["poi"] = data_df["poi"]

# combine imputet poi and not-poi datasets into one dataset
data_df = pd.concat([poi_data_filled, not_poi_data_filled], axis=0)
# shuffle the merged dataset to avoid any bias in downstream processing
data_df = data_df.sample(frac=1)


# uncomment to load the imputed version of the dataset instead
'''with open("data_df_filtered_imputed.pkl", "r") as file_in:
    data_df = pickle.load(file_in)'''


### TASK 3: CREATE NEW FEATURES ------------------------------------------------------------
### 3.1 extract file paths to employee emails (required for feature engineering)
### If the user wants to check the validity of the code below and use the entire email corpus, please make sure that the email corpus exists on the local machine and the correct root paths pointing to the directory is provided!!!! Else use the test-set of the email corpus provided. Load the preprocessed dataset with all email paths provided via pickle. Comment the "pickle.open" code snippet and uncomment the code below to extract email paths (processing of the entire email corpus takes some time!!!).

with bz2.BZ2File("from_email_paths.pkl.bz2", "r") as file_in:
    from_email_paths = pickle.load(file_in)

#------------- uncomment ------------#
"""# import relevant functions
from poi_email_addresses import poiEmails
from email_paths import *

# path to directory that contains a test-set of the email dataset (can be used to test the code);
# for selected employees there is a separate subdirectory each within root
# root path depends on where the user has saved the email corpus on his harddrive!!
#root = "/Users/damirvana/Coursera_EDX/UDACITY_Data_Analyst_Nanodegree/P5_Machine_Learning/maildir_test"
root_path = os.path.join(os.getcwd(),"maildir_test")

# extract all available email addresses for each employee from the main dataset (data_df)
#--> dictinary wih email as key and employee name as value
email_processor = EmailPaths(root_path)
poi_emails = poiEmails()
email_by_name = email_processor.EmployeeEmails(poi_emails, default_dataset)
# remove inaccurate email address
email_by_name.pop("NaN")
#extract file paths to emails
#--> dictionary with employee name as key and list of paths as calue
from_email_paths = email_processor.extractPath(email_by_name)"""
#------------- uncomment ------------#

### 3.2 Feature engineering
### Prior to feature engineering, email features from the main dataset will be reconstructed (see
### notebook for more information). The entire procedure encompassing feature reconstruction and creating new features is very time consuming. Thus, the user can either load a preprocessed dataset that contains reconstructed features as well as all engineered features (for more information please see the notebook). Load the dataset via pickle. Alternatively, the user can execute the entire code below to perform reconstruction and feature engineering. Again, this can be facilitated either by processing the entire email corpus (make sure the dataset is present locally on the machine; code execution is very time consuming) or by using the email corpus test-set provided.


with open("data_df_update_tfidf.pkl", "r") as file_in:
    data_df = pickle.load(file_in)

#------------- uncomment ------------#
"""# reconstruction of email features
#---------------------------
#---------------------------
from EmailProcessing_COUNTS import *
email_counter = EmailCounts(from_email_paths, email_by_name, data_df)


# extract email counts to reconstract email features from the main dataset
dict_sent_total, dict_sent_timestamp = email_counter.fit("sent_messages")
dict_received_total, dict_received_timestamp = email_counter.fit("received_messages")
dict_sent_to_poi = email_counter.fit("sent_to_poi")
dict_received_from_poi = email_counter.fit("received_from_poi")
dict_poi_shared = email_counter.fit("shared")

#check if email counts are truly missing for any employee and update accordingly
email_datasets = [dict_sent_to_poi, dict_received_from_poi, dict_sent_total,
                  dict_received_total, dict_poi_shared]
# check for each employee in each dataset
for employee in data_df.index:
    
    all_values = [dataset[employee] for dataset in email_datasets]
    # continue if no missing values
    if np.nan not in all_values:
        continue
    
    is_int = [isinstance(value, int) for value in all_values]
    # continue if all entries are missing values
    if not any(is_int):
        continue

    # transform missing values to 0 values if at least one int entry
    for idx in range(len(is_int)):
        if np.isnan(all_values[idx]):
            email_datasets[idx][employee] = 0.

# convert email counts into dataframes and replace the cognate features in the main dataset
sent_to_poi = pd.DataFrame({"from_this_person_to_poi" : dict_sent_to_poi})
received_from_poi = pd.DataFrame({"from_poi_to_this_person" : dict_received_from_poi})
sent_total = pd.DataFrame({"from_messages" : dict_sent_total})
received_total = pd.DataFrame({"to_messages" : dict_received_total})
poi_shared = pd.DataFrame({"shared_receipt_with_poi" : dict_poi_shared})
for data in [sent_to_poi, received_from_poi, sent_total, received_total, poi_shared]:
    # update only replaces non-nan values; convert NaN to string
    data_df.update(data.replace(np.nan, "NaN"))

# convert NaN string back to true NaN (required for imputation)
data_df = data_df.replace("NaN", np.nan)

# Engineering of new features (frequency change for sent/received emails)
# (see notebook for more information).
#---------------------------
#---------------------------
counter = EmailFrequencyChange(data_df)
freq_change_sent = counter.fit(dict_sent_timestamp)
freq_change_received = counter.fit(dict_received_timestamp)

#  convert into dataframes and update the main dataset.
freq_change_sent = pd.DataFrame({"freq_change_sent" : freq_change_sent})
freq_change_received = pd.DataFrame({"freq_change_received" : freq_change_received})
for dataframe in [freq_change_sent, freq_change_received]:
    data_df = pd.merge(data_df, dataframe, left_index = True, right_index = True )

# reconstruction of features requires to repeat imputation
# split updated main dataset into poi versus not-poi subsets
poiYes_data = data_df[data_df["poi"] == True]
poiNo_data = data_df[data_df["poi"] == False]

# update email feature-list defined above (add frequency features to email feature list)
email_features.update(["freq_change_sent", "freq_change_received"])

# perform imputation for each subset and replace the original dataset with imputed features
email_features_poi_imputed = MultipleImputation(poiYes_data, email_features)
email_features_not_poi_imputed = MultipleImputation(poiNo_data, email_features)
data_df.update(email_features_poi_imputed)
data_df.update(email_features_not_poi_imputed)

# Engineering of new features (TF-IDF values)
# (see notebook for more information).
#---------------------------
#---------------------------

from EmailProcessing_NLP import *
# extract stemmed email variants and corresponding author names
process = EmailStemmer(from_email_paths,data_df)
word_data, from_data = process.ProcessEmail()


from EmailProcessing_TFIDF import *
# Upsample minority class (poi)
sampler = Upsampling(word_data, from_data, data_df)
word_data_upsampled, from_data_upsampled = sampler.upsample()

# Calculate TF-IDF values for each employee; Please see the notebook for more information on hyperparameter
# selection for TfidfVectorizer() algorithm!!
selector = TFIDFcalculator(from_data_upsampled, word_data_upsampled, data_df, ngram_range=(1, 1), max_df=0.6, min_df=1)
tfidf_features_df = selector.extractFeature()

# update the main dataset; to_frame() converts pandas series into pandas datframe
tfidf_features_df = pd.merge(tfidf_features_df, data_df["poi"].to_frame(), left_index=True, right_index=True)

# split the main dataset into poi and not-poi subsets; explicit indexing to avoid reindexing due to different df
# lengths
poiYes_data = tfidf_features_df.loc[data_df["poi"] == True]
poiNo_data = tfidf_features_df.loc[data_df["poi"] == False]

# perform imputation for each subset and replace the original dataset with imputed features
tfidf_features_poi_imputed = MultipleImputation(poiYes_data, tfidf_features_df.columns.drop("poi"))
tfidf_features_not_poi_imputed = MultipleImputation(poiNo_data, tfidf_features_df.columns.drop("poi"))

# Add imputed TF-IDF values as new features to the main dataset
tfidf_features_all_imputed = pd.concat([tfidf_features_poi_imputed, tfidf_features_not_poi_imputed])
data_df = pd.merge(data_df, tfidf_features_all_imputed, right_index=True, left_index=True)"""
#------------- uncomment ------------#




### TASK 4: TRY A VARIETY OF CLASSIFIERS

### Following algorithms will be tested:
#   - Support Vector Classifier (SVC)
#   - Naive Bayes Classifier (GaussianNB)
#   - K-Means Classifier
#   - Decision Tree Classifier
#   - Random Forest Classifier
#   - Ada Boost Classifier
#   - Logistic Regression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV,train_test_split
from itertools import combinations

# Standarize features to 0 mean and unit variance (best practise with SVC) and convert to dictionary
# (compatibility with tester.py script).
scaler = StandardScaler()
data_df_scaled = data_df.copy()
data_df_scaled[data_df.columns.drop("poi")] = scaler.fit_transform(data_df_scaled[data_df.columns.drop("poi")])
final_dataset = data_df_scaled.T.to_dict()

# define a default feature list and move poi feature to first place (compatibility with tester.py script)
# (Reason for removal of particular features, see "Check for irregularities and outliers" section in the notebook)
feature_list_default = list(default_dataset.columns.drop(["deferral_payments", "restricted_stock_deferred", "loan_advances", "director_fees", "email_address"]))
feature_list_default.insert(0, feature_list_default.pop(feature_list_default.index("poi")))


# test different classifiers with the default feature set (for more details, please see the notebook)
for clf in [GaussianNB(),KMeans(), LogisticRegression(class_weight="balanced"), SVC(class_weight="balanced"),
            ADA(), DT(class_weight="balanced"), RF(class_weight="balanced")]:
    
    tester.test_classifier(clf, final_dataset, feature_list_default)


### 4.1: Evaluate the impact of feature engineering on classification performance (for more details, please refer to the notebook). Test only with SVC (the best classifier).

print "BASELINE PERFORMANCE -default feature set and SVC with linear kernel"
print "--------------------------------------------------------------------"
clf = SVC(kernel = "linear", class_weight="balanced")
tester.test_classifier(clf, final_dataset, feature_list_default)

print "EXTENDED FEATURE SET1 PERFORMANCE -default feature set + TF-IDF features and SVC with linear kernel"
print "--------------------------------------------------------------------"
selected_feature_list = feature_list_default+['word_feature_2', 'word_feature_3']
clf = SVC(kernel = "linear", class_weight="balanced")
tester.test_classifier(clf, final_dataset, selected_feature_list)

print "EXTENDED FEATURE SET2 PERFORMANCE -default feature set + FREQ-CHANGE features and SVC with linear kernel"
print "--------------------------------------------------------------------"
selected_feature_list_2 = feature_list_default+['freq_change_sent', 'freq_change_received']
clf = SVC(kernel = "linear", class_weight="balanced")
tester.test_classifier(clf, final_dataset, selected_feature_list_2)

### 4.2 Sequential Feature Selection (for more details, please refer to the notebook). Here, we will apply a combinatorial approach to evaluate the feature space for the best feature subset combination that results in best classification performance with SVC (selected as best classifier). This greedy search selection approach is very time consuming !!!! Uncomment the code below to execute the code. However, the result can be viewed by accessing the "BFS_summary_20folds.pkl" file.

# expand default feature set with engineered features
feature_list_engineered = list(data_df.columns)
feature_list_engineered.insert(0, feature_list_engineered.pop(feature_list_engineered.index("poi")))

# remove features with a SelectKBest score < 3; preselection based on SelectKBest score
to_remove = ['freq_change_sent','from_messages','word_feature_4','word_feature_5','to_messages','from_poi_to_this_person']
preselected_features = [item for item in feature_list_engineered if item not in to_remove]

# perform feature selection; uncomment to execute greedy feature selection
from feature_selection import *
#clf = SVC(kernel = "linear", class_weight="balanced")
#bfs = BFS(clf,final_dataset, preselected_features,5)
#bfs.search()

print "BEST FEATURE SUBSET PERFORMANCE -selected features and SVC with linear kernel"
print "--------------------------------------------------------------------"
clf = SVC(kernel = "linear", class_weight="balanced")
tester.test_classifier(clf, final_dataset, final_features)


### TASK 5: HYPERPARAMETER TUNING
### uncomment to perform grid search-based hyperparameter tuning

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

"""# Extract features and labels from dataset
data = tester.featureFormat(final_dataset, final_features)
labels, features = tester.targetFeatureSplit(data)

# define the parameter grid to be tested
parameters = {'clf__kernel':['linear'],
              'clf__C':[0.01,0.2,0.4,0.8,1,2,4,8,10,100],
              'clf__class_weight':['balanced', {1:4,0:1},{1:5,0:1},{1:6,0:1}]}

# define the pipeline
pipeline = Pipeline([("clf", SVC())])

# set validation to 500-fold cross-validation via StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=500,  random_state=42)
grid_search = GridSearchCV(pipeline, parameters, scoring = "f1", cv=cv)
grid_search.fit(features, labels)"""


### TASK 6: EVALUATE CLASSIFIER PERFORMANCE AFTER HYPERPARAMETER TUNING AND DUMP CLASSIFIER, DATASET AND FEATURE SUBSET

# evaluate classifier after hyperparameter tuning; hyperparameter values based on grid search result
print "PERFORMANCE AFTER HP TUNING"
print "--------------------------------------------------------------------"
clf = SVC(kernel = "linear", C=0.4, class_weight={0: 1, 1: 6})
tester.test_classifier(clf, final_dataset, final_features)
tester.dump_classifier_and_data(clf, final_dataset, final_features)
