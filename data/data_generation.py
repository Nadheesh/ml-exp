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
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)


def __operator(x, w, c, noise):
    """
    Operator is used to generate artificial datasets
    :param w:
    :param c:
    :param noise:
    :return:
    """
    return x * w + c + noise


def generate_data(n, d, test_size, operator = None):
    """
    generate data
    :param n: number of data samples
    :param d: number of predictors
    :param test_size: size of the test set. should be in between 0 and 1
    :param operator: send the function to generate values for dependant var using predictors
    :return: (train_X, test_X, train_y, test_y)
    """

    # checking if the operator is none
    if operator is None:
        operator = __operator

    if test_size <= 0 and n <= 0 and d <=0 :
        raise ValueError("n, d and test size should be greater than 0")

    _n = n
    n = int(n + n * test_size)
    X = np.matrix([np.random.uniform(-1, 1, d) for i in range(n)])
    _w = np.matrix(np.random.uniform(-10, 10, d)).T
    _c = float(np.random.uniform(-10, 10, 1))

    y = operator(x=X, w=_w, c=_c, noise=np.matrix(np.random.normal(scale=0.5, size=n)).T)
    y = np.array(y).flatten()

    return train_test_split(X, y, test_size=int(_n * test_size), shuffle=True)
