import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class BaseGrid:
    def __init__(self, params):
        """Initialize grid.

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
                "DeciionEnvironmentGrid currently supports at most 2 parameters."
            )

        # Cartesian product of parameter values, each row is 1 environment
        self.param_settings = pd.DataFrame({"key": [1]})
        for k, v in self.params.items():
            temp = pd.DataFrame(v, columns=[k])
            temp["key"] = 1
            self.param_settings = self.param_settings.merge(temp, on="key", how="outer")
        self.param_settings = self.param_settings.drop(columns="key")

    def plot(self, title):
        if self.num_params == 1:
            ax = sns.relplot(
                data=self.data,
                x=self.param_names[0],
                y="util",
                hue="type",
                kind="line",
            )
        else:
            ax = sns.relplot(
                data=self.data,
                x=self.param_names[0],
                y="util",
                hue="type",
                col=self.param_names[1],
                col_wrap=2,
                facet_kws={"sharey": False},
                kind="line",
            )
        ax.fig.suptitle(title)
        plt.show()
