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


def plot_predictive_uncertainty(X, true_y, prediction, error, ax = None):
    """
    plots the prediction and their uncertainty/error/variance using error bars

    *** uses the log of the error to improve the visibility ***
    :param X:
    :param prediction:
    :param error:
    :param ax:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(np.array(X).T[0], true_y, label='true', c='r')
    ax.errorbar(np.array(X).T[0], prediction, yerr=np.log(error), fmt='o', label='prediction', c='y')
    ax.legend(loc=0)

    return ax