import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import BaseGrid
from decision_maker import DynamicDecisionMaker


class DecisionEnvironment:
    """The environment in which the agent acts."""

    def __init__(self, N=12, num_trials=100, sigma=1, mu=0, tau=1):
        """Initialize the decision environment.

        Parameters
        ----------
        N : integer of numeric, optional
            The value of the actions. Defaults to 12.
        num_trials : int, optional
            The number of trials/equal decisions in the environment. Defaults
            to 100.
        sigma : numeric, optional
            The standard deviation of V as a Gaussian function of Vhat_i.
            Defaults to 1.
        mu : numeric, optional
            The mean of the value of actions (generative). Defaults to 0.
        tau : numeric, optional
            The standard deviation of V_hat, the context-free values for
            different actions (st. deviation from mu).
        """

        self.N = N
        # each row of V (axis=1) is 1 trial
        unsorted_Vhat = np.random.multivariate_normal(
            np.repeat(mu, N), np.diag(np.repeat(tau, N)), num_trials
        )
        self.Vhat = pd.DataFrame(np.sort(unsorted_Vhat, 1)[:, ::-1])
        self.num_trials = num_trials
        self.sigma = sigma
        self.tau = tau

        # each row is a trial
        self.V = pd.DataFrame(index=pd.RangeIndex(num_trials))

        # Draw Vh_i from V_i for each trial
        for action in self.Vhat.columns:
            self.V[action] = np.random.multivariate_normal(
                self.Vhat[action], np.diag(np.repeat(sigma, num_trials))
            )

        # the value of the best action
        self.optimal = self.V.max(axis=1)


class DecisionEnvironmentGrid(BaseGrid):
    def __init__(self, params):
        """Initialize decision maker grid."""

        super().__init__(params)

    def plot_complex(self, title, normalize=False):
        """TODO

        Parameters
        ----------
        title: str
            Title to use for the plot produced.
        normalize : Boolean, optional
            Whether to display optimal as well, or just normalize to optimal=0.
            Defaults to False.
        """
        data = pd.DataFrame()

        for i, parameters in self.param_settings.iterrows():
            params_dict = {k: v for k, v in zip(parameters.index, parameters.values)}
            env = DecisionEnvironment(**parameters)
            dm = DynamicDecisionMaker(env)
            dm.decide()
            df = dm.summary(table=True, normalize=normalize)
            for k, v in params_dict.items():
                df[k] = v

            data = data.append(df)

        self.data = (
            data.drop(columns="median")
            .reset_index()
            .melt(id_vars=["index"] + parameters.index.tolist(), value_name="util")
            .rename(columns={"index": "type"})
        )

        self.plot(title)
