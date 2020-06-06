import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import BaseGrid


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

    def plot_gain(self):
        """Plot gain for each trial as a function of delta vhats."""

        diffs = abs(self.data["Vh_1"] - self.data["Vh_2"])

        fig, ax1 = plt.subplots()
        fig.suptitle(f"V={self.V}, V'={self.Vp}, sigma={self.sigma}")

        plt.scatter(diffs, self.data["gain"])
        plt.xlabel("Difference between vhats")
        plt.ylabel("Expected gain from eval")

        plt.show()


class DecisionEnvironmentGrid(BaseGrid):
    def __init__(self, params):
        """Initialize decision environment grid."""

        super().__init__(params)

    def plot_complex(self, title, mode="gain", dm=None, **kwargs):
        """Plot results across the environments in the grid.

        Parameters
        ----------
        title : str
            The title of the chart (facet grid) to be produced.
        mode : str, optional
            The type of chart to draw. Defaults to "gain", except if dm is
            passed then it is set to "dm".
        dm : DecisionMaker, optional
            If passed and gain is False, evaluate this decision maker on the
            environments. Defaults to None.
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
