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
import pymc3 as pm
from matplotlib import pyplot as plt

from bayesian.model_api import Model, _adding_intecept
from data.data_generation import generate_data
from visualizing.plot_kde import plot_kde
from visualizing.predictive_uncertaity import plot_predictive_uncertainty

# HOW TO USE THE MODEL
if __name__ == '__main__':
    n = 500
    d = 5

    # generating data example
    train_X, test_X, train_y, test_y = generate_data(n, d, 0.2)
    train_X, test_X = _adding_intecept(train_X), _adding_intecept(test_X)

    # using model example
    model = Model(n_draws=1000, init_model=None)
    model.fit(train_X, train_y)
    pred, err = model.predict(test_X, with_error=True)
    pm.traceplot(model.trace)

    # plotting uncertainty plots example
    plot_predictive_uncertainty(test_X, test_y, pred, err)

    # plotting kde for given samples example
    fig, ax = plt.subplots()
    coef = model.trace['w'].T
    for w in coef:
        plot_kde(w, ax=ax)

    # display the plots
    plt.show()
