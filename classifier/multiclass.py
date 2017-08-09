from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
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


def evaluate(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, cv=4, scoring='accuracy')
    print(scores)


def errorAnalysis(clf, X_train, y_train):
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx)
    plt.colorbar()
    plt.title('Confusion matrix')
    plt.show()
    print('The following plot error explicitly')
    row_sum = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sum
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    print('From above you can see that the clf confused on 3/5 the most')
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
    plotDigits(X_aa[:25])
    plotDigits(X_ab[:25])
    plotDigits(X_ba[:25])
    plotDigits(X_bb[:25])
    plt.axis('off')
    plt.show()

def plotDigits(mx):
    images = mx.reshape(5, 5, 28, 28)
    f, axarr = plt.subplots(5, 5, sharex=True)
    for i, sub_image in enumerate(images):
        for j, image in enumerate(sub_image):
            axarr[i][j].imshow(image, cmap=matplotlib.cm.binary, interpolation='nearest')




if __name__ == '__main__':
    clf, X_train, X_test, y_train, y_test, mnist = prep()
    trival(clf, X_train, y_train, mnist)
    useOVO(X_train, y_train, mnist)
    evaluate(clf, X_train, y_train)
    errorAnalysis(clf, X_train, y_train)
