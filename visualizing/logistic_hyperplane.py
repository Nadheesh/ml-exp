"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def _invlogit(x):
    """
    sigmoid operator
    :param x: int/float or array like data-structure
    :return: sigmoid of x
    """
    return np.exp(x) / (1 + np.exp(x))


def operator(x, w, c, noise):
    """
    Operator which is used to generate datasets
    :param w:
    :param c:
    :param noise:
    :return:
    """
    return _invlogit(x * w + c + noise)


n = 1000000
d = 10

########################################
# test the operator

X = np.matrix(np.linspace(-2, 2, n)).T
w = 2
c = 1
noise = np.matrix(np.random.normal(scale=0.3, size=n)).T
y = operator(x=X, w=w, c=c, noise=0)
y = np.array(y).flatten()

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(X, y)
ax.set_ylabel("y")
ax.set_xlabel("x")
# plt.show()

#######################################
# multiple hyperplanes

_y = (y >= 0.5).astype(np.int)

lr = LogisticRegression()
lr.fit(X, _y)

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(X, y, label = "true hyperplane")
ax.plot(X, operator(x=X, w=lr.coef_[0], c=lr.intercept_, noise=0), label="estimated hyperplane")
ax.plot(X, _y, "*", label ="labels")
ax.legend(loc=0)
ax.set_ylabel("y")
ax.set_xlabel("x")
# plt.show()

#######################################
# multi dimensional test

X = np.matrix([np.random.uniform(-1, 1, d) for i in range(n)])
w = np.matrix(np.random.uniform(-10, 10, d)).T
c = float(np.random.uniform(-10, 10, 1))
noise = np.matrix(np.random.normal(scale=0.3, size=n)).T

y = operator(x=X, w=w, c=c, noise=0)
y = (y >= 0.5).astype(np.int)

lr = LogisticRegression()
lr.fit(X, y)

_w = np.array(w).flatten()
print("True coefficients : %s" % _w.tolist())
print("Estimated coefficients : %s" % lr.coef_[0])

# fig = plt.figure(3)
# ax = fig.add_subplot(111)
# ax.plot(_w, lr.coef_[0], "*")
# ax.plot(c, lr.intercept_, "o")
# ax.set_ylabel("estimated coefficients")
# ax.set_xlabel("true coefficients")
# plt.show()

__X = np.array(X.T[0].T).flatten()
__w = float(w[0])
__ew = float(lr.coef_[0][0])
__c = float(c)
__ec = float(lr.intercept_)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(__X, __X*__w+c, label = "True")
ax.plot(__X, __X*__ew+__ec, label = "Estimated")
plt.show()