################# RIDGECLASSIFIER ############################################
from sklearn.linear_model import RidgeClassifier

estimator = RidgeClassifier(
    alpha=0.1, #default
    fit_intercept=True, # default
    # fit_intercept=False, # when data is alrady centerd
    solver= 'auto', # default
    # solver= 'svd', # single value decompostion to find ridge coefficient
    # solver = 'cholesky', # uses  scipy.linalg.solve function to obtain closed form solution
    # solver = 'sparse_sg', # uses conjugate gradiant solver of scipy.sparse.linalg.cg -  when the data is sparse(many 0)
    # solver = 'lsqr', # uses dedicated regularized least-square reoutine scipy.sparse.linalg.lsqr and its fastest
    # solver = 'sag', # uses stochastic average gradiant descent - when both features and samples are more
    # solver = 'saga', # saga is unbaised version of sag - when both features and samples are more
    # solver = 'lbfgs', # impelmented in scipy.optimize.minimize, can be used only when the coefficients are forced to be postive
    # max_iter= 'None', # default # when using sag or saga - absolute maximum number of passes (epochs) 
    tol ='1e-3', # default 0.001 # when using sag or saga- if the improvement between two epoch is less than tol then it stops the training
    class_weight= 'None' #default
    # class_weight='balanced', # for highly imbalance class weight = total sample / total class * samples in calss
    # calss_weight = {0: 1 , 1: 3} # class 1 mistakes weighted 3 times    
)

################# RIDGECLASSIFIERCV ############################################

from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


alphas = np.arange(0,5,0.5) # only different alpha can be cross validated
custom_cv = StratifiedKFold(n_splits = 5,shuffle=True, random_state=42)
custom_cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=3, random_state=42)

cv_estimator = RidgeClassifierCV(
    cv =5,
    # cv= custom_cv,
    alphas=alphas,
    fit_intercept=True,
    scoring='accuracy', # default
    # scoring = 'f1' or 'f1_macro'
    class_weight=None,
    store_cv_values=False # must be false when the cv is int
    # store_cv_values = True # when cv=None,use Leave-One-Out validation, store cv predction for every single value
)









