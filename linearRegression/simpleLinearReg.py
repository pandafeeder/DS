import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# simple linear regression, f(x) = a + bx
train_x = np.array([6, 8, 10, 14, 18]).reshape(5, 1)
train_y = np.array([7, 9, 13, 17.5, 18]).reshape(5, 1)

_, ax = plt.subplots()
ax.plot(train_x, train_y, 'ko')
ax.grid(True)
ax.axis([0, 20, 0, 20])

# scikit learn estimator
reg = LinearRegression()
reg.fit(train_x, train_y)

t = np.arange(20).reshape(20, 1)
ax.set_title('Simple Linear Regression')
ax.plot(t, reg.predict(t), 'g-')
ax.annotate('scikit-learn predict line', \
    xy=(4, reg.predict(4)), \
    xytext=(4, reg.predict(4)-5), \
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=8))

# plot residuals
for i,v in enumerate(train_x):
    ax.plot((v[0], v[0]), (train_y[i][0], reg.predict([v])[0]), 'r-')

# hand made simple linear estimator
class myLinearReg:
    '''
    f(x) = a + bx
    Using training data to learn the values of a and b
    to produce best fit is called __ordinary least square__
    1. variance measures how far a set of values spread out;
    2. covariance is a measure of how much two variables change together;
    b = covar(x, y) / var(x)
    a = mean(y) - b*mean(x)
    '''
    def fit(self, X, Y):
        x_variance = np.var(X, ddof=1)
        xy_covariance = np.cov(X.reshape(len(X)), Y.reshape(len(Y)))[0][1]
        self._b = xy_covariance / x_variance
        self._a = np.mean(Y) - self._b*np.mean(X)
        
    def predict(self, x):
        return self._a + self._b*x

    def score(self, x, y):
        predict_y = self.predict(x)
        return 1 - ( np.sum((y - predict_y) ** 2) / np.sum((y - np.mean(y)) ** 2))

myReg = myLinearReg()
myReg.fit(train_x, train_y)

ax.text(0.5, 19, 'scikit-learn reg: coeffient:{0:.2f}, intercept:{0:.2f}'.format((reg.coef_)[0][0], (reg.intercept_)[0]))
ax.text(0.5, 17, 'my own reg: coeffient:{0:.2f}, intercept:{0:.2f}'.format(myReg._b, myReg._a))

ax.plot(t+0.5, myReg.predict(t+0.5), 'b^')
ax.annotate('myLinearReg predict line', \
    xy=(10.5, reg.predict(10.5)), \
    xytext=(10.5, reg.predict(10.5)-5), \
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=8))

# evaluat the model using r-squared
test_x = np.array([8, 9, 11, 16, 12]).reshape(5, 1)
test_y = np.array([11, 8.5, 15, 18, 11]).reshape(5, 1)
ax.text(0.5, 18, 'R-squared of scikit-learn reg: {0:.2f}'.format(reg.score(test_x, test_y)))
ax.text(0.5, 16, 'R-squared of my own reg: {0:.2f}'.format(myReg.score(test_x, test_y)))

plt.show()

