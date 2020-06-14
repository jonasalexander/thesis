import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DecisionEnvironment:
    """The environment in which the agent acts."""

    def __init__(self, N=12, num_trials=50, sigma=1, tau=1, k_values=None, mu=0):
        """Initialize the decision environment.

        Parameters
        ----------
        N : integer of numeric, optional
            The value of the actions. Defaults to 12.
        num_trials : int, optional
            The number of trials/equal decisions in the environment. Defaults
            to 50.
        sigma : numeric, optional
            The standard deviation of Vh_i as a Gaussian function of V/Vp.
            Defaults to 1.
        tau : numeric, optional
            The standard deviation of V, the values for different actions.
        k_values : list of integers, optional
            Values of K for which to evaluate static agents. Defaults to
            [1, 2, 3, 4, 5, 7, 10].
        mu : numeric, optional
            The mean of the value of actions (generative). Defaults to 0.
        """

        self.N = N
        # TODO: get different V each trial too
        self.V = np.sort(np.random.normal(mu, tau, N))[::-1]
        self.num_trials = num_trials
        self.sigma = sigma
        self.tau = tau

        self.k_values = k_values or [1, 2, 3, 4, 5, 7, 10]

        # each row is a trial
        self.vhats = pd.DataFrame(index=pd.RangeIndex(num_trials))

        # Draw Vh_i from V_i for each trial
        for i, v in enumerate(self.V):
            self.vhats[i] = np.random.normal(v, sigma, num_trials)

    def plot_action_values(self):
        """Plot a number line with the values of the actions."""
        fig, ax = plt.subplots(figsize=(8, 0.15))
        plt.scatter(self.V, np.repeat(0, len(self.V)))
        ax.get_yaxis().set_visible(False)
        plt.xlabel("Value of action")
        plt.show()

    def plot_vhats(self, action_idx=0):
        """Plot distribution of vhats for a specific action (across trials).

        Parameters
        ----------
        action_idx : integer, optional
            The index within self.V (column in self.vhats) corresponding to
            the action for which to show vhats for the different trials
        """
        plt.hist(self.vhats[action_idx])
        plt.xlabel("Vhat")
        plt.ylabel("Count")
        plt.title(
            f"Distribution of vhats for action {action_idx} with value "
            f"{self.V[action_idx]:.2f} (across trials)"
        )
        plt.show()
