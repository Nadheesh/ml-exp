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
import pandas as pd
import pymc3 as pm
from theano import shared, tensor as tt

from data.data_generation import generate_data

seed = 7
PPC_CONST = 20000

def _adding_intecept(X):
    """
    Adding a column for intercept
    :param X: features
    :return:
    """
    return np.c_[X, np.ones(X.shape[0])]


def _init_model(X, y):
    """
    exmaple and default specification of a model
    specify a linear regression model
    :param X:
    :param y:
    :return:
    """
    with pm.Model() as model:
        # Define hyper-prior
        alpha = pm.Gamma("alpha", alpha=1e-2, beta=1e-4)

        # Define priors'
        w = pm.Normal("w", mu=0, sd=alpha, shape=train_X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=10)
        mu = tt.dot(w, X.T)

        # Define likelihood
        likelihood = pm.StudentT("y", nu=1, mu=mu, lam=sigma, observed=y)
    return model


class Model:

    def __init__(self, n_draws, init_model=None):
        """
        initiate the pymc3 model
        :param n_draws: number of samples drawn from the target distributions
        :param init_model: pymc3 model initiation, check example provided
        """
        self.n_draws = n_draws

        if init_model is None:
            init_model = _init_model
        self._init_model = init_model

    def fit(self, X, y):
        """
        train model
        :param X:
        :param y:
        :return:
        """
        self.shared_X = shared(X)
        self.shared_y = shared(y)
        self.model = self._init_model(self.shared_X, self.shared_y)

        with self.model:
            self.trace = pm.sample(self.n_draws, random_seed=seed, njobs=1)

    def predict(self, X, with_error = False):
        """
        predict using the train model
        :param X:
        :return:
        """
        if not hasattr(self, 'trace'):
            raise AttributeError("trace is not found. train the model first")

        train_X = self.shared_X.get_value()
        self.shared_X.set_value(X)
        ppc = pm.sample_ppc(self.trace, model=self.model, samples=PPC_CONST, random_seed=seed)

        self.shared_X.set_value(train_X)

        if with_error :
            return ppc['y'].mean(axis=0), ppc['y'].std(axis=0)  # return prediction and error
        return ppc['y'].mean(axis=0)


# HOW TO USE THE MODEL
if __name__ == '__main__':

    n = 500
    d = 5

    # use other scripts
    train_X, test_X, train_y, test_y = generate_data(n, d, 0.2)

    train_X, test_X = _adding_intecept(train_X), _adding_intecept(test_X)

    model = Model(n_draws=1000, init_model=_init_model)
    model.fit(train_X, train_y)

    pred, err = model.predict(test_X, with_error=True)

    from matplotlib import pyplot as plt
    pm.traceplot(model.trace)

    from visualizing.predictive_uncertaity import plot_predictive_uncertainty
    plot_predictive_uncertainty(test_X, test_y, pred, err)

    plt.show()


