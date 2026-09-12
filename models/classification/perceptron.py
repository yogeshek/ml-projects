################# PERCEPTRON ############################################
from sklearn.linear_model import Perceptron

estimator = Perceptron(
    penalty=None, # default
    #peanlty = 'l2',
    #penalty='l1',
    #penalty ='elasticnet'
    alpha='0.0001', #default
    l1_ratio='0.15', # only when peanlty is elsticnet
    fit_intercept=True,
    #filt_intercept=False,
    max_iter=1000, # max number of epoches trained
    tol=0.001, #stopping tolerance, trianing cuts after that
    shuffle=True, # control whether the training sample reshuffled after every epoch
    #shuffle=False,
    n_jobs=None, #numner of cpu used for multiclass
    #n_jobs=2
    early_stopping=False, #stops training when velidation scores stop improving
    validation_fraction=0.1, # when early stopping=True, The proportion the training dataset set aside for validation
    n_iter_no_change=5, #  when early stopping=True,number of iteration to wait with no improvement, before triggersing early_stopping
    class_weight=None,
    # class_weight=Balanced,
    # class_weight=custom,
    warm_start=False
    #warm_start=True, # reusing the previous weight leanred
    )

################# SGDCLASSIFIER - PERCEPTRON ############################################

from sklearn.linear_model import SGDClassifier

estimator = SGDClassifier(
    #loss = 'hinge', # Default Linear SVM focussing on maximizing the safety margin between classes
    loss = 'perceptron', #The simple linear loss used in the classic Perceptron algorithm.    
    # loss = 'log_loss', # Logistic Regression
    # loss = 'log',# depliciated- Logistic Regression
    # loss = 'modified_huber', # hybrid loss , cobines properties of hinge and logistic regression
    # loss ='squared_hinge', #similar to hinge- but takes the margin error and square it
    
    # # below functions used for SGDRegressor to predict continous number.
    # # SGDClassifier adopt them for classification by treating the classes as numerica1 0 and 1 / -1 and +1
    # loss = 'squared_error', # ordinary least square- minimize the error
    # loss = 'huber', # act like squared_error for minor mistakes, but swithces to absolute peanlty for major(MAE)
    # loss = 'epsilon_insensitive', # SVR- linear support vector regression- ignore the prediction erros less than threshold(epsilon)
    # loss = 'squared_epsilon_insensitive', # SVR - similar to epsilon-insensitive but once the error exceeds the epsilon the peanlty is squared
    epsilon=0.1, #when loss =[huber,epsilon_insensitive,squared_epsilon_insensitive]
    learning_rate='constant', # the stepsize stays exactly et0 for entire training duration
    # learning_rate='optimal', # [(eta = 1.0 /(alpha * (t+10)))] uses heuristic formula(by Leon Bottou) ,shrinks the step size dynamically based on the alpha (current time step t )
    # learning_rate='invascaling', # [eta = eta0 / pow(t, power_t)], default power_t= 0.5 in SGD
    # learning_rate='adaptive', #keep the eta0 as long as the training loss keep going down, if it stop improving by n_iter_no_chage timem, then devide the eta0 by 5.
    eta0=1,# when learning_rate in['constant','invscaling','adaptive']  
    penalty=None
    
)







