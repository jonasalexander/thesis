import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import BaseGridMixin


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


class DecisionMakerGrid(BaseGridMixin):
    """Compare decision makers for different parameter settings."""

    def __init__(self, params, env, average=True, **kwargs):
        """Initialize decision maker grid.

        Parameters
        ----------
        param : str
        min_val : numeric
        max_val : numeric
        num : int
            Number of decision makers to compare between min_val and max_val.
        num_samples : int
        **kwargs
            Passed on to
        """

        self.params = params
        self.param_names = list(params.keys())
        self.num_params = len(self.param_names)
        self.env = env

        if self.num_params == 0:
            raise ValueError("Params has no keys, must have at least one.")
        elif self.num_params > 2:
            raise NotImplementedError(
                f"DecisionMakerGrid currently supports at most 2 parameters, "
                f"but got {self.num_params}."
            )

        parameter_settings = pd.DataFrame({"key": [1]})
        for k, v in self.params.items():
            temp = pd.DataFrame(v, columns=[k])
            temp["key"] = 1
            parameter_settings = parameter_settings.merge(temp, on="key", how="outer")
        parameter_settings = parameter_settings.drop(columns="key")

        self.data = pd.DataFrame()
        for i, row in parameter_settings.iterrows():
            for j, param in enumerate(self.param_names):
                kwargs.update({param: row[j]})
            dm = DecisionMaker(**kwargs)

            temp = pd.DataFrame()
            if average:
                temp = temp.append(
                    {"type": "dynamic", "util": np.mean(dm.decide(self.env))},
                    ignore_index=True,
                )
                temp = temp.append(
                    {"type": "default", "util": np.mean(self.env.data["default"])},
                    ignore_index=True,
                )
                for j, param in enumerate(self.param_names):
                    temp[param] = row[j]

                self.data = self.data.append(parameter_settings.merge(temp))
