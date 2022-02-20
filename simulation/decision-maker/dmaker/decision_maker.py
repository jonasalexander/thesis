from abc import ABC, abstractmethod
from collections import Counter
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_COST_EVAL = 0.2
AGENT_COLUMNS = [
    "num_eval",
    "utility",
    "utility_second_best",
    "gross_utility_last_eval",
    "V_index",
    "Vhat_index",
]
EXPERIMENT_COLUMNS = ["subject.id", "last", "value", "order"]


class BaseDecisionMaker(ABC):
    def __init__(self, env, name, cost_eval):
        """Initialize the dynamic decision maker.

        Parameters
        ----------
        env : DecisionEnvironment
        cost_eval : numeric, optional
            The utility cost to evaluate an action.
        """
        self.env = env
        self.cost_eval = cost_eval or DEFAULT_COST_EVAL
        self.name = name

        """
        num_eval: number of actions evaluated.
        utility: utility of action chosen, minus cost of evaluation (net)
        V_index: rank of the action chosen among V
        Vhat_index: rank of the action chosen among Vhat (order of consideration)
        """
        self.data = pd.DataFrame(
            index=pd.RangeIndex(self.env.num_trials),
            columns=AGENT_COLUMNS,
        )

        self.experiment_data = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    @abstractmethod
    def decide(self):
        pass

    def normalize_util(self):
        normalized_data = self.data.copy()
        normalized_data["utility"] -= self.env.optimal
        normalized_data["utility_second_best"] -= self.env.optimal
        normalized_data["gross_utility_last_eval"] -= self.env.optimal
        return normalized_data


class FixedDecisionMaker(BaseDecisionMaker):
    def __init__(self, env, num_eval, cost_eval=None):
        """Initialize fixed K decision maker

        Parameters
        ----------
        num_eval: integer
            The number of actions (in order of Vhat) to evaluate for each trial.
        """
        name = "K" + str(num_eval)
        self.num_eval = num_eval
        super().__init__(env, name, cost_eval)

    def decide(self):
        # Assign value of highest V action among the k-best actions by Vhat
        # by Vhat, for each trial

        self.data["num_eval"] = self.num_eval

        indices_evaluated = range(self.num_eval)
        for i in range(self.env.num_trials):
            sorted_V_with_indices = (
                self.env.V.loc[i]
                .rename("utility")
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "Vhat_index"})
            )
            Vs_evaluated = (
                sorted_V_with_indices.loc[
                    sorted_V_with_indices["Vhat_index"].isin(indices_evaluated)
                ]
                .reset_index()
                .rename(columns={"index": "V_index"})
            )
            action_chosen = Vs_evaluated.loc[0]
            self.data.loc[i, "utility"] = (
                action_chosen["utility"] - self.num_eval * self.cost_eval
            )
            self.data.loc[i, "V_index"] = action_chosen["V_index"]
            self.data.loc[i, "Vhat_index"] = action_chosen["Vhat_index"]

            if self.num_eval > 1:
                second_best_action = Vs_evaluated.loc[1]
                self.data.loc[i, "utility_second_best"] = (
                    second_best_action["utility"] - self.num_eval * self.cost_eval
                )

            self.data.loc[i, "gross_utility_last_eval"] = Vs_evaluated.loc[
                len(Vs_evaluated) - 1, "utility"
            ]


class DynamicDecisionMaker(BaseDecisionMaker):
    """Dynamic decision maker."""

    def __init__(self, env, num_samples=1000, cost_eval=None):
        """Create dynamic decision maker in a given environment.

        Parameters
        ----------
        num_samples : integer, optional
            The number of samples to draw for each V as a function of Vhat (for
            the empirical distribution). Defaults to 1000.
        """

        name = "Dynamic"
        self.num_samples = num_samples
        super().__init__(env, name, cost_eval)

    def decide(self):
        # used to add increasing cost of eval to actions evaluated later
        cost_eval_list = [i * self.cost_eval for i in range(self.env.N)]
        cost_eval_adjuster = np.transpose(
            np.repeat(cost_eval_list, self.num_samples).reshape(
                (self.env.N, self.num_samples)
            )
        )

        # for each trial
        for i in range(self.env.num_trials):

            Vhats = self.env.Vhat.loc[i]
            V_sorted = (
                self.env.V.loc[i].sort_values(ascending=False).rename("V").reset_index()
            )

            # Agent's simulated distribution of V given Vhat, excluding cost_eval
            cov_matrix = np.diag(np.repeat(self.env.sigma ** 2, self.env.N))
            v_dist = np.random.multivariate_normal(Vhats, cov_matrix, self.num_samples)

            # Agent has to get the value of the first action
            Vb_initialization = -float("inf")
            Vb = Vb_initialization  # best so far
            Vb2 = Vb_initialization  # second best
            V_prev = None  # initialize to empty
            for j in range(self.env.N + 1):
                if j == self.env.N:
                    # have evaluated all the actions
                    # know the agent will be able to get the best action
                    V_index = 0
                    self.experiment_data = self.experiment_data.append(
                        {"subject.id": i, "last": True, "value": V_prev, "order": j},
                        ignore_index=True,
                    )
                    break

                v_floored_dist = v_dist[:, j:].copy()
                v_floored_dist[v_floored_dist < Vb] = Vb
                # Subtract increasing costs of evaluation for each action
                # further down the list of Vhats (but not including sunk cost
                # of actions already evaluated
                v_floored_dist = (
                    v_floored_dist - cost_eval_adjuster[:, : self.env.N - j]
                )

                # Utility of continuing to evaluate, based on simulated distribution
                # Take the mean over num_samples
                # and then take the max over evaluating 1, 2, ... more actions
                V_eval = max(v_floored_dist.mean(axis=0)) - self.cost_eval

                if Vb > V_eval:
                    # Vb is better than expectation of continuing to evaluate
                    # not j + 1 because actually want to refer to the previous iteration
                    self.experiment_data = self.experiment_data.append(
                        {"subject.id": i, "last": True, "value": V_prev, "order": j},
                        ignore_index=True,
                    )
                    break
                else:
                    if (
                        not V_prev is None
                    ):  # checks that we aren't on the first iteration
                        # not j + 1 because actually want to refer to the previous iteration
                        self.experiment_data = self.experiment_data.append(
                            {
                                "subject.id": i,
                                "last": False,
                                "value": V_prev,
                                "order": j,
                            },
                            ignore_index=True,
                        )

                    # Evaluate current action and recurse
                    V_prev = self.env.V.loc[i, Vhats.index[j]]
                    self.data.loc[i, "gross_utility_last_eval"] = V_prev

                    if V_prev < Vb:
                        # Don't need to update Vb
                        # But may need to update second best value
                        # Also check that this is not the first value we've evaluated
                        if j > 0 and Vb2 < V_prev:
                            Vb2 = V_prev
                    else:
                        if j > 0:
                            Vb2 = Vb
                        V_index = V_sorted[V_sorted["V"] == V_prev].index[0]
                        Vhat_index = j
                        Vb = V_prev

            self.data.loc[i, "num_eval"] = j
            self.data.loc[i, "utility"] = Vb - j * self.cost_eval
            self.data.loc[i, "V_index"] = V_index
            self.data.loc[i, "Vhat_index"] = Vhat_index
            self.data.loc[i, "utility_second_best"] = Vb2 - j * self.cost_eval
