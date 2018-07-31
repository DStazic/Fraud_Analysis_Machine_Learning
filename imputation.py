#!/usr/bin/python

import numpy as np
import pandas as pd
from fancyimpute import MICE


def MultipleImputation(dataset, features):
    '''
    Takes a dataset and a set of feature names that refer to features to be imputed in the dataset.
    Facilitates multiple imputation technique on missing data and returns the imputed dataset.
    
    dataset: dataset with missing values (dataframe)
    features: set with feature names specifying what features to be grouped for imputation (set or list)
    '''
    
    # make copy of original dataset to prevent changes in original dataset
    dataset_copy = dataset.copy()
    
    # convert deferred_income to positive values in order to allow log10 transformation
    if "deferred_income" in features:
        dataset_copy["deferred_income"] *= -1

    # do log10 transformation; +1 to transform 0 values
    data_log = np.log10(dataset_copy[list(features)]+1)

    # restrict min value to 0 to avoid <0 imputed values
    # --> important when fitting imputation model with feature values close to 0
    data_filled = MICE(n_imputations=500, verbose=False, min_value=0).complete(np.array(data_log))
    data_filled = pd.DataFrame(data_filled)
    data_filled.index = dataset.index
    data_filled.columns = data_log.columns

    # transform back to linear scale; subtract 1 to obtain original non-imputed values
    data_filled = 10**data_filled-1
    
    # convert deferred_income back to negative values (original values)
    if "deferred_income" in features:
        data_filled["deferred_income"] *= -1

    return data_filled