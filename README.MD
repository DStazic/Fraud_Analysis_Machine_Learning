#Identification of employee fraud

The goal of this project is to apply machine learning and train a classifier to predict employees involved in 
fraudulent activity at Enron, a business powerhouse in the late 90s early 2000s that basically colapsed overnight 
after massive accounting fraud was uncovered.


Two datasets were used in the course of building a classification model:
- the main dataset stores financial data of 146 Enron employees as well as summarized statistics about email-based 
communication

- the second dataset comprises of approximately 500000 emails authored by various Enron employees (this email 
corpus can be downloaded from the Carnegie Mellon School of Computer Science (https://www.cs.cmu.edu/~enron/).

#Run the analysis pipeline
The analysis pipeline/scripts can be executed via jupyter notebook (be careful and provide the correct root path to
the filder containing the email corpus for reconstructing email features and engineering new features!! email 
corpus must be downloaded and stored locally). Executing the entire notebook will take a lot of time (long runtime 
for particular scripts). Alternatively, preprocessed files (see section "Using preprocessed files" below) that are
generated via the pipeline are provided and can be loaded instead and simply feed into downstream pipeline (see 
jupyter notebook for more clarification).

Also, training and testing of different classifier can be executed via jupyter notebook as a standalone by first
loading the required main datasets and the file storing information on the best feature subset (see Notification 
in the section "Train and test different classifiers" in the jupyter notebook file). This will also generate the
final files (my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl) that can be used to validate the 
training and testing results.


###Scripts used for imputation:
- imputation.py

###Scripts used to specify file paths to emails authored by Enron employees present in the main dataset:
- email_paths.py
- poi_email_addresses.py

###Scripts used for feature engineering (involves cleaning and processing of email data):
- EmailProcessing_COUNTS.py
- EmailProcessing_NLP.py
- EmailProcessing_TFIDF.py

###Following script was provided by UDACITY to test the performance of a classifier:
- tester.py

###Following script was used to select the best feature subset:
- feature_selection.py


#Using preprocessed files
The above mentioned scripts return files that are each required for downstream processing in the analytics 
pipeline. Some of those scripts have longer runtime, so I will also provide the relevant files returned.

###Following files were generated using scripts that require longer runtime: 
- from_email_paths.pkl.bz2 (compressed pickled file that stores file paths to emails by employees)
- dict_sent_total.pkl
- dict_sent_to_poi.pkl
- dict_sent_timestamp.pkl
- dict_received_timestamp.pkl
- dict_received_from_poi.pkl
- dict_poi_shared.pkl
-> the upper 5 files store the reconstructed email features from the main dataset
- data_df_update_corrected.pkl (main dataset with reconstructed email features and imputation of missing data)
- nlp_emails_tokenized_third_cleaning.pkl (stemmed email variants after final text cleaning)
- nlp_emails_authors_third_cleaning.pkl (corresponding list of authors)
- data_df_update_tfidf.pkl (final version of the main dataset; includes new features and imputation)
- default_dataset (default version of the main dataset)
- BFS_summary_20folds.pkl (file containing the results from the sequential search for best subset)
-> all main dataset files mentioned above are in dataframe format

###Following files were used in the course of email text cleaning:
- Black_Female_Names_nlp.zip
- Black_Male_Names_nlp.zip
- White_Female_Names_nlp.zip
- White_Male_Names_nlp.zip
-> those files store canonical names of black and white american citizens, which were used to remove names from 
the email corpus prior to stemming. Files were obtained from  http://mbejda.github.io/

###Following files can be used for validating the classification results by using tester.py:
- my_dataset.pkl (final version of the main dataset in dictionary format; pickled)
- my_classifier.pkl (file containing parametrization of hyperparameter tuned classifier;pickled)
- my_feature_list.pkl (file storing names of best feature subset used; pickled)

