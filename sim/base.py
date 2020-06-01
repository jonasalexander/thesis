import seaborn as sns
import matplotlib.pyplot as plt


class BaseGridMixin:
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
