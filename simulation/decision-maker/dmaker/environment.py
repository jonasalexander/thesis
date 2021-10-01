import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmaker.base import BaseGrid
from dmaker.decision_maker import DEFAULT_COST_EVAL, DynamicDecisionMaker


class DecisionEnvironment:
    """The environment in which the agent acts."""

    def __init__(self, N=12, num_trials=100, sigma=1, mu=0, tau=1):
        """Initialize the decision environment.

        Parameters
        ----------
        N : integer or numeric, optional
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

    def calculate_gain(self, cost_eval=None):
        """Calculate gain as a function of environmental variables.

        Parameters
        ----------
        cost_eval : numeric, optional
            The utility cost to evaluate an action. Defaults to DEFAULT_COST_EVAL.
        """
        cost_eval = cost_eval or DEFAULT_COST_EVAL

        self.data = (
            self.V.reset_index()
            .rename(columns={"index": "trial"})
            .melt(value_name="V", id_vars="trial", var_name="action index")
            .sort_values(by=["trial", "action index"])
            .reset_index(drop=True)
        )
        vhat_data = (
            self.Vhat.reset_index()
            .rename(columns={"index": "trial"})
            .melt(value_name="Vhat", id_vars="trial", var_name="action index")
            .sort_values(by=["trial", "action index"])
            .reset_index(drop=True)
        )
        self.data = self.data.join(vhat_data, rsuffix="_todrop").drop(
            columns=["trial_todrop", "action index_todrop"]
        )

        self.data["Vb"] = 0
        for idx in self.data.index:
            trial = self.data.loc[idx, "trial"]
            V = self.data.loc[self.data["trial"] == trial, "V"]
            for action_idx in V.index:
                self.data.loc[action_idx, "Vb"] = max(V.loc[:action_idx])

        self.data["gain"] = np.NaN
        self.data["Vhat_next"] = np.NaN
        for trial in self.data["trial"].unique():
            trial_idx = self.data["trial"] == trial
            self.data.loc[trial_idx, "gain"] = (
                np.roll(self.data.loc[trial_idx, "Vb"].diff(), shift=-1) - cost_eval
            )
            vhat_next = np.roll(self.data.loc[trial_idx, "Vhat"], shift=-1)
            vhat_next[-1] = np.NaN
            self.data.loc[trial_idx, "Vhat_next"] = vhat_next

    def plot_gain(self, metric, cost_eval=None):
        """Plot gain (net utility) as a function of a specific environmental variable.


        For continuous independent variables, use histogram for smooth
        bucketing. For discrete, just take average (groupby with agg).

        Parameters
        ----------
        metric : string
            The environmental variable to use as the independent variable for
            plotting.
        cost_eval : numeric, optional
            The utility cost to evaluate an action. Defaults to DEFAULT_COST_EVAL.
        """

        try:
            self.data
        except AttributeError as e:
            self.calculate_gain(cost_eval)

        if metric == "num_eval":
            print("3")
            plt.scatter(
                range(self.N), self.data.groupby("action index").agg("mean")["gain"]
            )
            plt.plot(range(self.N), self.N * [0], color="r")
            # plt.show()

        elif metric == "Vb":
            # bin by Vb and then for each bin, calculate mean gain

            cutoffs = np.linspace(self.data["Vb"].min(), self.data["Vb"].max())
            bins = pd.cut(self.data["Vb"], cutoffs)
            Vb_values = self.data.groupby(bins)["gain"].agg("mean")

            plt.figure(figsize=(20, 10))
            plt.scatter(cutoffs[:-1], Vb_values.values)
            plt.xticks(cutoffs[:-1:2])
            plt.xlabel("value of best action so far")
            plt.ylabel("net gain")
            plt.show()

        elif metric == "Vhat_next":
            cutoffs = np.linspace(
                self.data["Vhat_next"].min(), self.data["Vhat_next"].max()
            )
            bins = pd.cut(self.data["Vhat_next"], cutoffs)
            Vb_values = self.data.groupby(bins)["gain"].agg("mean")

            plt.figure(figsize=(20, 10))
            plt.scatter(cutoffs[:-1], Vb_values.values)
            plt.xticks(cutoffs[:-1:2])
            plt.xlabel("context-free value of next best action")
            plt.ylabel("net gain")
            plt.show()

        else:
            raise NotImplementedError(f"Unexpected metric type {metric}")


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
