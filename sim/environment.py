import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import BaseGridMixin


class DecisionEnvironment:
    """The environment in which the agent acts."""

    def __init__(self, V=1, Vp=0, N=50, sigma=1):
        """Initialize the decision environment.

        Parameters
        ----------
        V : numeric, optional
            The value of the better action. Defaults to 1.
        Vp : numeric, optional
            The value of the worse action. Defaults to 0.
        N : int, optional
            The number of trials/equal decisions in the environment. Defaults
            to 50.
        sigma : numeric, optional
            The standard deviation of Vh_i as a Gaussian function of V/Vp.
            Defaults to 1.
        """

        self.V = V
        self.Vp = Vp
        self.N = N
        self.sigma = sigma

        self.data = pd.DataFrame(index=pd.RangeIndex(N))

        # draw N Vh_1 from V
        self.data["Vh_1"] = np.random.normal(V, sigma, N)
        # draw N Vh_2 from V'
        self.data["Vh_2"] = np.random.normal(Vp, sigma, N)

        # the better of V(A_1) and V(A_2)
        self.data["dynamic"] = V

        # whether Vh_1 < Vh_2
        self.data["switch"] = 0

        for i in range(N):

            Vh_1 = self.data.loc[i, "Vh_1"]
            Vh_2 = self.data.loc[i, "Vh_2"]

            if Vh_1 < Vh_2:
                self.data.loc[i, "switch"] = 1
                Vh_1, Vh_2 = Vh_2, Vh_1  # switch so that Vh_1 is larger

            p1 = np.exp(-0.5 * (((Vh_1 - V) / sigma) ** 2 + ((Vh_2 - Vp) / sigma) ** 2))
            p2 = np.exp(-0.5 * (((Vh_1 - Vp) / sigma) ** 2 + ((Vh_2 - V) / sigma) ** 2))

            denom = p1 + p2
            # probability that Vh_1 is drawn from V (as opposed to V')
            self.data.loc[i, "probs"] = p1 / denom

            # expected gain of evaluating
            self.data.loc[i, "gain"] = V - (V * p1 + Vp * p2) / denom
            # V(A_1)
            self.data.loc[i, "default"] = [V, Vp][self.data.loc[i, "switch"]]

    def plot_gain(self, only_exp=False):
        """Plot gain for each trial.
        Parameters
        ----------
        only_exp : Boolean, optional
            Whether to display only expected gain or delta of Vh_i as well.
            Defaults to False.
        """

        fig, ax1 = plt.subplots()
        fig.suptitle(f"V={self.V}, V'={self.Vp}, sigma={self.sigma}")

        ax1.plot(self.data["gain"], label="expected gain from eval", color="r")
        ax1.set_ylabel("Expected gain from eval")

        if not only_exp:
            diffs = abs(self.data["Vh_1"] - self.data["Vh_2"])
            ax2 = ax1.twinx()
            ax2.plot(diffs, label="diff between vhats", color="b")
            ax2.set_ylabel("Difference between vhats")
            fig.legend()

        plt.show()


class DecisionEnvironmentGrid(BaseGridMixin):
    def __init__(self, params):
        """Initialize decision environment grid.

        Parameters
        ----------
        params : dict, str -> list
            Dict where the key is the string parameter name and the value is a
            list of values that parameter should take on.
        """

        self.params = params
        self.param_names = list(params.keys())
        self.num_params = len(self.param_names)

        if self.num_params == 0:
            raise ValueError("Params has no keys, must have at least one.")
        elif self.num_params > 2:
            raise NotImplementedError(
                "DecisionEnvironmentGrid currently supports at most 2 parameters."
            )

        # Cartesian product of parameter values, each row is 1 environment
        self.param_settings = pd.DataFrame({"key": [1]})
        for k, v in self.params.items():
            temp = pd.DataFrame(v, columns=[k])
            temp["key"] = 1
            self.param_settings = self.param_settings.merge(temp, on="key", how="outer")
        self.param_settings = self.param_settings.drop(columns="key")

    def plot_compare(self, title, mode="gain", dm=None, **kwargs):
        """Plot results across the environments in the grid.

        Parameters
        ----------
        title : str
            The title of the chart (facet grid) to be produced.
        mode : str, optional
            The type of chart to draw. Defaults to "gain", except if dm is
            passed then it is set to "dm".
        dm : DecisionMaker, optional
            If passed and gain is False, evaluate the decision maker on the
            environment. Defaults to None.
        **kwargs
            Passed on to DecisionEnvironment.
        """

        if dm is not None:
            mode = "dm"

        self.data = pd.DataFrame()
        for i, row in self.param_settings.iterrows():
            for j, param in enumerate(self.param_names):
                kwargs.update({param: row[j]})
            env = DecisionEnvironment(**kwargs)

            temp = pd.DataFrame()
            if mode == "gain":
                temp = temp.append(
                    {"type": "gain", "util": np.mean(env.data["gain"])},
                    ignore_index=True,
                )
            elif mode == "dm":
                temp = temp.append(
                    {"type": "dynamic", "util": np.mean(dm.decide(env))},
                    ignore_index=True,
                )
                temp = temp.append(
                    {"type": "always_eval", "util": env.V - dm.cost_eval},
                    ignore_index=True,
                )
            elif mode == "diff":
                temp = temp.append(
                    {
                        "type": "difference",
                        "util": np.mean(env.data["dynamic"])
                        - np.mean(env.data["default"]),
                    },
                    ignore_index=True,
                )
            elif mode == "compare":
                temp = temp.append(
                    {"type": "dynamic", "util": np.mean(env.data["dynamic"])},
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
