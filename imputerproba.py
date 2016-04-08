# Authors: Stephane Collot <stephane.collot@gmail.com>
# License: BSD 3 clause

from operator import itemgetter
from collections import defaultdict

import numpy as np
import pandas as pd


__all__ = [
    'ImputerProba',
]



class ImputerProba():
    """Imputation transformer for tranforming categorical values to
    probability based on training set.
    Parameters
    ----------
    columns : list
        List of columns that are going to converted to probability
    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.
    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:
        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If `axis=0` and X is encoded as a CSR matrix;
        - If `axis=1` and X is encoded as a CSC matrix.
    Attributes
    ----------
    statistics_ : 
    Notes
    -----

    """

    def __init__(self, columnsEncod, columnsProba, verbose=0, copy=True, leaveOneOut=False):
        self.columnsEncod = columnsEncod
        self.columnsProba = columnsProba
        self.verbose = verbose
        self.copy = copy
        self.leaveOneOut = leaveOneOut
        
    def fit_transform(self, X, y):
        if self.verbose: print "fit_transform, lengh X: " + str(len(X))
        
        self.fit(X, y)
        return self.transform(X, y=y, training=True)
        
        #return self

    def fit(self, X, y):
        if self.verbose: print "Fit, lengh X: " + str(len(X))
        self.catEncoders = dict()
        self.catProbability = dict()
        self.catProbaSum = dict()
        self.catProbaCount = dict()

        # Create the mapping from categorie to int
        for col in self.columnsEncod:
            if col not in X.columns:
                if self.verbose: print "Column encod: " + str(col) + " is not in X, skip."
                continue
            # Create dictionnary
            uniqueValues = X[col].unique()
            dictValues = dict()
            #dictValues = defaultdict(lambda: -1, dictValues)       
            for v in uniqueValues:
                if v is np.nan:
                    continue
                if not v in dictValues:
                    dictValues[v] = len(dictValues)
            dictValues[np.nan] = -1 # Leave NaN as NaN for the imputer mean
        
            self.catEncoders[col] = dictValues

        # Create the mapping from categorie's value to probability
        _X = X.copy()
        _X['target'] = y
        for col in self.columnsProba:
            if col not in X.columns:
                if self.verbose: print "Column proba: " + str(col) + " is not in X, skip."
                continue
                
            # Cat probability
            df = pd.pivot_table(_X, values='target', columns=[col], aggfunc=np.average, fill_value=-1) # fillvalue useless
            probaDict = df.to_dict()
            if -1 in probaDict: # Removing -1: if we have a missing value then do not replace it with a proba
                del probaDict[-1]
            self.catProbability[col] = df.to_dict()
            
            # leave-one-out extra calculation
            if self.leaveOneOut:
                df = pd.pivot_table(_X, values='target', columns=[col], aggfunc=np.sum, fill_value=-1)
                self.catProbaSum[col] = df.to_dict()
                df = pd.pivot_table(_X, values='target', columns=[col], aggfunc=len, fill_value=-1)
                self.catProbaCount[col] = df.to_dict()
              

        return self


    def transform(self, X, y=None, training=False):
        if self.verbose: print "Transform, training: " + str(training) + "  leaveOneOut: " + str(self.leaveOneOut)
        for col in self.columnsEncod :
            if col not in X.columns:
                if self.verbose: print "Column encod: " + str(col) + " is not in X, skip."
                continue
            uniqueValues = X[col].unique()

            # If value is in the training but not in the testing, then map to missing -2
            uniqueValuesTraining = self.catEncoders[col]
            for v in set(uniqueValues) - set(uniqueValuesTraining):
                self.catEncoders[col][v] = -2

        for col in self.columnsProba :
            if col not in X.columns:
                if self.verbose: print "Column proba: " + str(col) + " is not in X, skip."
                continue
            uniqueValues = X[col].unique()

            # If value is in the training but not in the testing, then map to missing -2
            uniqueValuesTraining = self.catProbability[col]
            for v in set(uniqueValues) - set(uniqueValuesTraining):
                self.catProbability[col][v] = -2
        
        if len(self.catEncoders):
            X = X.replace(self.catEncoders)
        
        if self.leaveOneOut:
            if training:
                _X = X.copy() # Very weird bug? Copying is necessary.
                _X['target'] = y
                _X = _X.apply(self.rowAction, axis=1)
                _X = _X.drop('target', axis=1, inplace=False)
                X = _X
            else:
                if len(self.catProbability):
                    X = X.replace(self.catProbability)
        else:
            if len(self.catProbability):
                X = X.replace(self.catProbability)
        
        print X.columns
        return X
        
    def rowAction(self, row):
        for col in self.catProbaSum:
            caV = row[col] # Category value
            if caV == -1:
                continue
            taV = row['target'] # Target Value
            sig = self.catProbaSum[col][caV] # Sum of target value in this Cat
            nbE = self.catProbaCount[col][caV] # Nbr of elements in this Cat
            if nbE == 1:
                row[col] = -2 # if one element == to we don't know
            
            row[col] = (sig-taV)/(nbE-1.0)
        
        return row

    def get_params(self, deep=False):
        return {'columnsEncod': self.columnsEncod, 'columnsProba': self.columnsProba}


