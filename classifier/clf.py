from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Data and labels
X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43]
]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#clfs
dt_clf = tree.DecisionTreeClassifier()
svc_clf = SVC()
perceptron_clf = Perceptron()
KNN_clf = KNeighborsClassifier()

#train
dt_clf.fit(X, Y)
svc_clf.fit(X, Y)
perceptron_clf.fit(X, Y)
KNN_clf.fit(X, Y)

#predict
test_data = [[188, 78, 40], [178, 70, 41], [165, 60, 38], [168, 55, 39]]
label = ['male', 'male', 'female', 'female']
for clf in [dt_clf, svc_clf, perceptron_clf, KNN_clf]:
    predict = clf.predict(test_data)
    print("{} accuracy is {}".format(clf.__class__.__name__, accuracy_score(label, predict)*100))

