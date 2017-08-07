from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

class Never5Classifier(BaseEstimator):
    '''
    alwarys predicts negative
    '''
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def prep():
    '''
    binary classifier for 5 and non-5
    '''
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)
    
    sgd_clf_5 = SGDClassifier(random_state=42)
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    
    sgd_clf_5 = SGDClassifier(random_state=42)
    #sgd_clf_5.fit(X_train, y_train_5)

    return sgd_clf_5, X_train, y_train_5

#evaluating schemes
def accuracy_eval(clf, X_train, y_train):
    '''
    accuracy is ratio of correct predictions, in this case,
    5s amounts about 10% in training data, 
    so a estimator predicts everything as non-5 could reach to about 90% accuracy,
    which is clearly not a good measure of performance.
    cross_val_score evaluate clf's model using K-fold cross-validation,
    remeber that K-fold cross-validation means splitting the training set into K-folds,
    then makeing predictions and evaluating them on each fold using a model trained on the remaining folds.
    '''
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    print(scores)

def confusion_matrix_eval(clf, X_train, y_train):
    '''
    each row in matrix represents an actual class,
    each column in matrix represents a predicted class,
    '''
    preds = cross_val_predict(clf, X_train, y_train, cv=3)
    return confusion_matrix(y_train, preds), preds

def precision_eval(y_train, y_pred):
    '''
    accuracy of the positive prediction, the ratio of real positive of all positive predictions
    precision = TP / (TP+FP)
    '''
    return precision_score(y_train, y_pred)

def recall_eval(y_train, y_pred):
    '''
    sensitivity or TPR(true positive rate), ratio of positive instances that are correctly detected
    recall = TP / (TP+FN)
    '''
    return recall_score(y_train, y_pred)

def f1_score_eval(y_train, y_pred):
    '''
    f1 score is the harmonic mean of precision and recall,
    regular means treats all values equally,
    harmonic means gives much more weight to low values.
    So clf will only get high f1 score if both precison and recall are high
    f1_score = 2 / (1/precision + 1/recall)
    '''
    return f1_score(y_train, y_pred) 

def adjust_threshold(clf):
    pass

def roc_eval_and_auc_eval(clf, X_train, y_train):
    '''
    '''
    y_scores = cross_val_predict(clf, X_train, y_train, cv=3, method='decision_function')
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plt.plot(fpr, tpr, lineWidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
    print("AUC:", roc_auc_score(y_train, y_scores))



if __name__ == '__main__':
    clf, X_train, y_train = prep()
    print("ACCURACY:")
    print("SGDClassifier for 5 and non-5")
    accuracy_eval(clf, X_train, y_train)
    never5_clf = Never5Classifier()
    print("Never5Classifier for 5 and non-5")
    accuracy_eval(never5_clf, X_train, y_train)
    print('Above demonstrats why accuracy is not preferred performance measure for classifiers,\n'+\
        'especially when you\'re dealing with skewed datasets(when some class are much more frequent than others)')

    print("CONFUSION MATRIX:")
    matrix, preds  = confusion_matrix_eval(clf, X_train, y_train)
    print(matrix)
    plt.matshow(matrix)
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.show()

    print("PRECISION SCORE:")
    print(precision_eval(y_train, preds))
    print("RECALL SCORE:")
    print(recall_eval(y_train, preds))
    print("F1 SCORE:")
    print(f1_score_eval(y_train, preds))
    print("There's a precision/recall tradeoff, incresing one will reduce another")

    print("ROC CURVE:")
    roc_eval_and_auc_eval(clf, X_train, y_train)
    print("There's another tradeoff, the higher recall, the more fpr classifier produces")

