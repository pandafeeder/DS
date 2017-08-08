from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
import numpy as np


def prep():
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], test_size=1/7, random_state=42)
    sgd_clf = SGDClassifier(random_state=42)
    return sgd_clf, X_train, X_test, y_train, y_test, mnist


def trival(clf, X_train, y_train, mnist):
    '''
    classifier for multiclass produce a classes_ attributes after fit
    which represents all the target classes.
    for each instance, it produces a score array whose highest value correspond predicted class
    '''
    clf.fit(X_train, y_train)
    random_index = np.random.randint(len(mnist))
    pred = clf.predict([mnist['data'][random_index]])[0]
    target = mnist['target'][random_index]
    print("predict:{} target:{}".format(pred, target))
    random_index_scores = clf.decision_function([mnist['data'][random_index]])
    print("random_index_scores", random_index_scores)
    print("clf class_", clf.classes_)

def useOVO(X_train, y_train, mnist):
    '''
    most binary clf in scikit learn use One Versus All for multiclass
    except for SVM classifier which use OVO
    if you want to use OVO, simply create an instance and pass a binary clf to its constructor
    '''
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    print("Use OVO for SGDClassifier")
    trival(ovo_clf, X_train, y_train, mnist)




if __name__ == '__main__':
    clf, X_train, X_test, y_train, y_test, mnist = prep()
    trival(clf, X_train, y_train, mnist)
    useOVO(X_train, y_train, mnist)
