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
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt


def plot_kde(samples : np.array, start =None, stop = None, size = 10000, ax = None):
    """
    plot distribution using samples

    :param samples: numpy array with the shape (n,),
                    where n is the number of samples
    :param start: starting point of the plot
                default will take the minimum out of samples
    :param stop: end point of the plot
                default will take the maximum out of samples
    :param size: number of point to draw from KD
                    more points --> high precision
    :param ax: axis from pyplot figure
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()

    if start is stop or start > stop: # split the condition to capture none
        start = None
        stop = None

    if start is None:
        start = np.min(samples)
    if stop is None :
        stop = np.max(samples)

    # create the distribution
    kde = gaussian_kde(samples)

    # generate some points
    x = np.linspace(start, stop, size)
    y = kde(x)

    ax.plot(x, y)

    return ax