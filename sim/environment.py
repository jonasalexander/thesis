import pandas as pd
import numpy as np


class DecisionEnvironment:
    """The environment in which the agent acts."""

    def __init__(self, N=12, num_trials=50, sigma=1, tau=1, k_values=None):
        """Initialize the decision environment.

        Parameters
        ----------
        N : integer of numeric, optional
            The value of the actions. Defaults to 12.
        num_trials : int, optional
            The number of trials/equal decisions in the environment. Defaults
            to 50.
        sigma : numeric, optional
            The standard deviation of Vh_i as a Gaussian function of V/Vp.
            Defaults to 1.
        tau : numeric, optional
            The standard deviation of V, the values for different actions.
        k_values : list of integers, optional
            Values of K for which to evaluate static agents. Defaults to
            [1, 2, 3, 4, 5, 7, 10].
        """

        self.N = N
        self.V = np.sort(np.random.normal(0, tau, N))[::-1]
        self.num_trials = num_trials
        self.sigma = sigma
        self.tau = tau

        self.k_values = k_values or [1, 2, 3, 4, 5, 7, 10]

        # each row is a trial
        self.vhats = pd.DataFrame(index=pd.RangeIndex(num_trials))

        # Draw Vh_i from V_i for each trial
        for i, v in enumerate(self.V):
            self.vhats[i] = np.random.normal(v, sigma, num_trials)
