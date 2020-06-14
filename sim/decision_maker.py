import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt


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
            Defaults to 100.
        """

        self.data = pd.DataFrame(index=pd.RangeIndex(self.env.num_trials))

        # used to add increasing cost of eval to actions evaluated later
        cost_eval_list = [i * self.cost_eval for i in range(self.env.N)]
        cost_eval_adjuster = np.transpose(
            np.repeat(cost_eval_list, num_samples).reshape((self.env.N, num_samples))
        )

        # the best action
        self.data["optimal"] = self.env.V.max()

        # Assign value of highest V action among the k-best actions by Vhat
        for k in self.env.k_values:
            # by Vhat, for each trial
            k_best_actions = np.array(
                [
                    self.env.vhats.loc[r].nlargest(k).index.values
                    for r in self.env.vhats.index
                ]
            )
            self.data["K" + str(k)] = [
                max(self.env.V[x]) - k * self.cost_eval for x in k_best_actions
            ]

        # for each trial
        for s in range(self.env.num_trials):

            vhats = self.env.vhats.loc[s]

            # Empirical distribution of V given Vhat, excluding cost_eval
            cov = sqrt(self.env.tau ** 2 + self.env.sigma ** 2)
            cov_matrix = np.diag(np.repeat(cov, self.env.N))
            v_dist = np.random.multivariate_normal(vhats, cov_matrix, num_samples)

            # For now, assume agent has to get the value of the first action

            # TODO: just change initialization to vhats[0]?

            Vb = -float("inf")  # best so far
            for i in range(self.env.N + 1):

                if i == self.env.N:
                    # have evaluated all the actions
                    self.data.loc[s, "num_eval"] = i
                    self.data.loc[s, "dynamic"] = Vb - i * self.cost_eval
                    break

                v_floored_dist = v_dist[:, i:]
                v_floored_dist[v_floored_dist < Vb] = Vb
                # After each time period, we've already paid one cost_eval,
                # sunk cost
                v_floored_dist -= cost_eval_adjuster[:, : self.env.N - i]

                # utility of continuing to evaluate, based on empirical distr.
                V_eval = max(v_floored_dist.mean(axis=0)) - self.cost_eval

                if Vb > V_eval:
                    # Vb is better than expectation of continuing to evaluate
                    self.data.loc[s, "num_eval"] = i
                    self.data.loc[s, "dynamic"] = Vb - i * self.cost_eval

                    break
                else:
                    # keep evaluating, recurse
                    V_i = self.env.V[vhats.sort_values(ascending=False).index[i]]
                    if V_i > Vb:
                        Vb = V_i

    def plot_util(self, limit=50):
        """Plot utility of different agents for each trial.

        Parameters
        ----------
        limit : integer, optional
            Number of trials for which to display utility
        """

        to_plot = self.data.loc[: limit - 1].reset_index().drop(columns="num_eval")

        fig, ax = plt.subplots(figsize=(15, 10))

        ax = sns.lineplot(
            ax=ax,
            data=to_plot.melt(id_vars="index", var_name="type", value_name="util"),
            x="index",
            y="util",
            hue="type",
        )

    def summary(self):
        """Create string summarizing agents' average performance."""

        BOLD = "\033[1m"
        END = "\033[0m"

        results = self.data.mean()
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
