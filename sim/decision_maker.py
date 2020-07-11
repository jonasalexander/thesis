import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


class DecisionMaker:
    """Expexted evaluation gain threshold decision maker."""

    def __init__(self, env, cost_eval=0.2, threshold=None):
        """Initialize the decision maker.

        Parameters
        ----------
        env : DecisionEnvironment
        cost_eval : numeric
            The utility cost to evaluate an action.
        threshold : numeric
            The value where when the expected value of gain is above this,
            the agent evaluates. Defaults to cost_eval (rational).
        """

        self.env = env

        self.cost_eval = cost_eval
        self.threshold = threshold or cost_eval

        self.data = pd.DataFrame()

    def decide(self, num_samples=1000):
        """Make decisions in a given environment.

        Parameters
        ----------
        num_samples : integer, optional
            The number of samples to draw for each V as a function of Vhat.
            Defaults to 1000.
        """

        self.data = pd.DataFrame(index=pd.RangeIndex(self.env.num_trials))

        # used to add increasing cost of eval to actions evaluated later
        cost_eval_list = [i * self.cost_eval for i in range(self.env.N)]
        cost_eval_adjuster = np.transpose(
            np.repeat(cost_eval_list, num_samples).reshape((self.env.N, num_samples))
        )

        # the best actiona
        self.data["optimal"] = self.env.V.max(axis=1)

        # Assign value of highest V action among the k-best actions by Vhat
        for k, k_name in zip(self.env.k_values, self.env.k_value_names):
            # by Vhat, for each trial
            k_best_actions = np.array(
                [
                    self.env.Vhat.loc[r].nlargest(k).index.values
                    for r in self.env.Vhat.index
                ]
            )
            self.data[k_name] = [
                max(self.env.V.loc[i, x]) - k * self.cost_eval
                for i, x in enumerate(k_best_actions)
            ]

        # for each trial
        for s in range(self.env.num_trials):

            Vhats = self.env.Vhat.loc[s]
            V_sorted = (
                self.env.V.loc[s].sort_values(ascending=False).rename("V").reset_index()
            )

            # Empirical distribution of V given Vhat, excluding cost_eval
            cov_matrix = np.diag(np.repeat(self.env.sigma, self.env.N))
            v_dist = np.random.multivariate_normal(Vhats, cov_matrix, num_samples)

            # For now, assume agent has to get the value of the first action

            # TODO: just change initialization to Vhat[0]?

            Vb = -float("inf")  # best so far
            for i in range(self.env.N + 1):

                if i == self.env.N:
                    # have evaluated all the actions
                    # know the agent will be able to get the best actions
                    Vb_index = 0
                    break

                v_floored_dist = v_dist[:, i:].copy()
                v_floored_dist[v_floored_dist < Vb] = Vb
                # Subtract increasing costs of evaluation for each action
                # further down the list of Vhats (but not including sunk cost
                # of actions already evaluated
                v_floored_dist = (
                    v_floored_dist - cost_eval_adjuster[:, : self.env.N - i]
                )

                # utility of continuing to evaluate, based on empirical distr.
                V_eval = max(v_floored_dist.mean(axis=0)) - self.cost_eval

                if Vb > V_eval:
                    # Vb is better than expectation of continuing to evaluate
                    break
                else:
                    # keep evaluating, recurse
                    V_i = self.env.V.loc[s, Vhats.index[i]]
                    if V_i > Vb:
                        Vb_index = V_sorted[V_sorted["V"] == V_i].index[0]
                        Vb = V_i

            self.data.loc[s, "num_eval"] = i
            self.data.loc[s, "dynamic"] = Vb - i * self.cost_eval
            self.data.loc[s, "dynamic_index"] = Vb_index

    def normalize_util(self):
        normalized_data = self.data.copy()
        for agent in self.env.k_value_names + ["dynamic"]:
            normalized_data[agent] -= normalized_data["optimal"]
        return normalized_data.drop(columns="optimal")

    def plot_util(self, limit=50, plot_num_eval=True):
        """Plot utility of different agents for each trial.

        Parameters
        ----------
        limit : integer, optional
            Number of trials for which to display utility. Defaults to 50.
        plot_num_eval : Boolean, optional
            Whether to show the number of actions the dynamic agent evaluates
            on another y axis. Defaults to True.
        """

        normalized_data = self.normalize_util()

        subset = normalized_data.loc[: limit - 1]
        to_plot = subset.reset_index().drop(columns=["num_eval", "dynamic_index"])

        fig, ax = plt.subplots(figsize=(20, 10))

        ax = sns.lineplot(
            ax=ax,
            data=to_plot.melt(id_vars="index", var_name="type", value_name="util"),
            x="index",
            y="util",
            hue="type",
        )
        plt.xlabel("Trial number")
        plt.ylabel("Utility")
        plt.title("Utility across different trials (0 = optimal)")

        if plot_num_eval:
            color = "black"
            ax2 = ax.twinx()
            ax2.set_ylabel(
                "Number of actions evaluated by dynamic agent (black)/Rank of the action chosen (red)"
            )
            ax2.scatter(subset.index, subset["num_eval"], color=color, linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color)

            color = "red"
            ax2.scatter(subset.index, subset["dynamic_index"], color=color, linewidth=2)

        plt.show()

    def summary(self, table=False, normalize=False):
        """Create string summarizing agents' average performance.

        Parameters
        ----------
        table : Boolean, optional
            If True, return formatted as table and not string.
        normalize: Boolean, optional
            If True, normalize utility so that optimal=0. Defaults to False.
        """

        if normalize:
            data = self.normalize_util()
        else:
            data = self.data

        if table:
            df = (
                pd.DataFrame(data.mean(axis=0))
                .drop(["num_eval", "dynamic_index"])
                .rename(columns={0: "mean"})
                .assign(
                    median=lambda df: np.median(
                        data.drop(columns=["num_eval", "dynamic_index"]), axis=0
                    )
                )
            )
            return df

        BOLD = "\033[1m"
        END = "\033[0m"

        results = data.mean()
        k_agent_results = "\n".join(
            [
                col + f": {BOLD} {results[col]:.2f} {END}"
                for col in results.index
                if "K" in col
            ]
        )

        return (
            f'Dynamic agent achieved average utility: {BOLD} {results["dynamic"]:.2f} {END}'
            f"\nwhile the fixed-K agents achieved average utility:"
            f"\n{k_agent_results}"
        )

    def plot_num_eval_index(self, limit=50):
        """Plot index of the action chosen as a function of the num_eval.

        Parameters
        ----------
        limit : integer, optional
            Number of trials for which to display utility. Defaults to 50.
        """

        subset = self.data.loc[: limit - 1]

        num_eval = subset["num_eval"]
        index = subset["dynamic_index"]

        weights = [
            20 * i for i in Counter(zip(num_eval, index)).values() for j in range(i)
        ]

        plt.scatter(num_eval, index, s=weights)
        plt.xlabel("number evaluated")
        plt.ylabel("index of action chosen")
        plt.title(
            "Index of action chosen as a function of number evaluated for first 50 trials"
        )
        plt.show()
