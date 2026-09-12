################### LOGISTICREGRESSION ###############################
#### also known as logi regression , maximum enytropy(maxent) classifier and log-linear classifier

from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(penalty=None,
                            # penalty='l2'
                            # penalty='l1',
                            # penalty='elasticnet',
                            solver='saga', # support all 4 penalty
                            # solver='sag', # support l2 and None
                            # solver='liblinear', # support l1 and l2
                            # solver='lbfgs', # support l2 and None
                            # solver='newton-cg', # support l2 and None
                            tol=0.0001,
                            C=1, # inverse of regularization strength
                            fit_intercept=True,
                            # fit_intercept=False,
                            intercept_scaling=1.0, # only when solver=liblinear and fit_intercept=True.It scales the synthetic intercept feature to reduce its regularization relative to regular feature.
                            class_weight=None,
                            # class_weight='balanced',
                            # class_weight=custom dict
                            max_iter=1000, #maximum number of iteration to converge
                            # multi_class= depreciated # 'auto', 'ovr', multinomial' handled based on t data
                            warm_start=False,
                            n_jobs=1,
                            l1_ratio=None # ratio between 0 to 1) when penalty is elasticnet

                        )

################### LOGISTICREGRESSION CV ###################################
from sklearn.linear_model import LogisticRegressionCV

C = [1,2,3,4]

cv_estimator = LogisticRegressionCV(Cs=C,
                                    fit_intercept=True,
                                    cv=5,
                                    penalty='l2',
                                    scoring=None,
                                    solver='lbfgs',
                                    tol='0.0001',
                                    max_iter=100,
                                    class_weight=None,
                                    n_jobs=0,
                                    refit=True,
                                    intercept_scaling=1,
                                    multi_class='auto',
                                    l1_ratios=None,
                                    )


################### SGDClassifier #############################################

from sklearn.linear_model import SGDClassifier

estimator = SGDClassifier(loss='log_loss',
                          penalty='l2',                          
                          alpha=0.0001,
                          l1_ratio=0.15,
                          fit_intercept=True,
                          shuffle=True,
                          epsilon=0.001,
                          max_iter=1000,
                          n_jobs=1,
                          learning_rate='optimal',
                          eta0=0.5,
                          early_stopping=False,
                          validation_fraction=0.1,
                          n_iter_no_change=5                                                                
                          )
