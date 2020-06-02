import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import BaseGrid


class DecisionMaker:
    """Expexted evaluation gain threshold decision maker."""

    def __init__(self, cost_eval=0.2, threshold=None):
        """Initialize the decision maker.

        Parameters
        ----------
        cost_eval : numeric
            The utility cost to evaluate an action.
        threshold : numeric
            The value where when the expected value of gain is above this,
            the agent evaluates. Defaults to cost_eval
        """

        self.cost_eval = cost_eval
        self.threshold = threshold or cost_eval

        self.data = pd.DataFrame()

    def decide(self, env):
        """Make decisions in a given environment.

        Parameters
        ----------
        env : DecisionEnvironment
        """

        # indices where evaluate
        idx = env.data["gain"] > self.threshold
        # 1 if agent evaluated
        self.data["did_eval"] = 0
        self.data.loc[idx, "did_eval"] = 1

        self.data["util"] = env.data["default"].copy()
        self.data.loc[idx, "util"] = env.data.loc[idx, "dynamic"] - self.cost_eval

        return self.data["util"].copy()

    def plot_exp_gain(self, env):
        self.decide(env)


class DecisionMakerGrid(BaseGrid):
    """Compare decision makers for different parameter settings."""

    def __init__(self, params):
        """Initialize decision maker grid."""

        super().__init__(params)

    def plot_complex(self, title, mode="gain", env=None, **kwargs):
        """Plot results across the environments in the grid.

        Parameters
        ----------
        title : str
            The title of the chart (facet grid) to be produced.
        mode : str, optional
            The type of chart to draw. Defaults to "gain", except if env is
            passed then it is set to "env".
        env : DecisionEnvironment, optional
            If passed and gain is False, evaluate the decision makers on this
            environment. Defaults to None.
        **kwargs
            Passed on to DecisionMaker.
        """

        if env is not None:
            mode = "env"

        self.data = pd.DataFrame()
        for i, row in self.param_settings.iterrows():
            for j, param in enumerate(self.param_names):
                kwargs.update({param: row[j]})
            dm = DecisionMaker(**kwargs)

            temp = pd.DataFrame()
            if mode == "gain":
                temp = temp.append(
                    {"type": "gain", "util": np.mean(env.data["gain"])},
                    ignore_index=True,
                )
            elif mode == "env":
                temp = temp.append(
                    {"type": "dynamic", "util": np.mean(dm.decide(env))},
                    ignore_index=True,
                )
                temp = temp.append(
                    {"type": "always_eval", "util": env.V - dm.cost_eval},
                    ignore_index=True,
                )
                temp = temp.append(
                    {"type": "default", "util": np.mean(env.data["default"])},
                    ignore_index=True,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")

            for j, param in enumerate(self.param_names):
                temp[param] = row[j]

            self.data = self.data.append(self.param_settings.merge(temp))

        self.plot(title)
