import pandas as pd
import numpy as np
from math import sqrt


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
            the agent evaluates. Defaults to cost_eval (rational).
        """

        self.cost_eval = cost_eval
        self.threshold = threshold or cost_eval

        self.data = pd.DataFrame()

    def decide(self, env, num_samples=1000):
        """Make decisions in a given environment.

        Parameters
        ----------
        env : DecisionEnvironment
        num_samples : integer, optional
            The number of samples to draw for each V as a function of Vhat.
            Defaults to 100.
        """

        self.data = pd.DataFrame(index=pd.RangeIndex(env.num_trials))

        # used to add increasing cost of eval to actions evaluated later
        cost_eval_list = [i * self.cost_eval for i in range(env.N)]
        cost_eval_adjuster = np.transpose(
            np.repeat(cost_eval_list, num_samples).reshape((env.N, num_samples))
        )

        # the best action
        self.data["optimal"] = env.V.max()

        # Assign value of highest V action among the k-best actions by Vhat
        for k in env.k_values:
            # by Vhat, for each trial
            k_best_actions = np.array(
                [env.vhats.loc[r].nlargest(k).index.values for r in env.vhats.index]
            )
            self.data["K: " + str(k)] = [
                max(env.V[x]) - k * self.cost_eval for x in k_best_actions
            ]

        # for each trial
        for s in range(env.num_trials):

            vhats = env.vhats.loc[s]

            # Empirical distribution of V given Vhat, excluding cost_eval
            cov = sqrt(env.tau ** 2 + env.sigma ** 2)
            cov_matrix = np.diag(np.repeat(cov, env.N))
            v_dist = np.random.multivariate_normal(vhats, cov_matrix, num_samples)

            # For now, assume agent has to get the value of the first action

            # TODO: just change initialization to vhats[0]?

            Vb = -float("inf")  # best so far
            for i in range(env.N + 1):

                if i == env.N:
                    # have evaluated all the actions
                    self.data.loc[s, "num_eval"] = i
                    self.data.loc[s, "dynamic"] = Vb - i * self.cost_eval
                    break

                v_floored_dist = v_dist[:, i:]
                v_floored_dist[v_floored_dist < Vb] = Vb
                # After each time period, we've already paid one cost_eval,
                # sunk cost
                v_floored_dist -= cost_eval_adjuster[:, : env.N - i]

                # utility of continuing to evaluate, based on empirical distr.
                V_eval = max(v_floored_dist.mean(axis=0)) - self.cost_eval

                if Vb > V_eval:
                    # Vb is better than expectation of continuing to evaluate
                    self.data.loc[s, "num_eval"] = i
                    self.data.loc[s, "dynamic"] = Vb - i * self.cost_eval

                    break
                else:
                    # keep evaluating, recurse
                    V_i = env.V[vhats.sort_values(ascending=False).index[i]]
                    if V_i > Vb:
                        Vb = V_i
