import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DecisionEnvironment:
    def __init__(
        self,
        num_decisions=10,
        num_options=100,
        util_dist_type="normal",
        util_param="pareto",
        seed=40,
    ):

        np.random.seed(seed)

        self.num_decisions = num_decisions
        self.num_options = num_options

        if util_dist_type == "normal":
            if util_param == "pareto":
                shape = 2
                mode = 10
                self.opt_vars = (np.random.pareto(shape, self.num_decisions) + 1) * mode
                cov = np.diag(self.opt_vars)
            else:
                raise NotImplementedError(f"Unrecognized util_param {util_param}")

            self.opt_mean = 10
            loc = [self.opt_mean] * self.num_decisions
            # np.random.normal(loc=10, scale=3, size=self.num_decisions)
            self.util = np.transpose(
                np.random.multivariate_normal(loc, cov, self.num_options)
            )
            # print(np.amin(self.util, axis=0))
        else:
            raise NotImplementedError(f"Unrecognized util_dist_type {util_dist_type}")

        # sort utility, decreasing
        self.util.sort(axis=1)
        self.util = np.flip(self.util, axis=1)

    def visualize(self, num_rows=2):
        if self.num_decisions % num_rows == 0:
            num_cols = int(self.num_decisions / num_rows)
            fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
            for i in range(num_rows):
                for j in range(num_cols):
                    axs[i][j].hist(self.util[i * num_cols + j])
                    axs[i][j].set_title(f"Decision {i*num_cols+j+1}")

        else:
            fig, axs = plt.subplots(1, self.num_decisions, sharey=True)

            for i in range(self.num_decisions):
                axs[i].hist(self.util[i])

        fig.suptitle(f"Distribution of {self.num_options} options for each decision")
        plt.show()

        avg_opt_util = [
            sum(self.util[i]) / self.num_options for i in range(self.num_decisions)
        ]
        plt.bar(range(1, self.num_decisions + 1), avg_opt_util)
        plt.title("Average utility in each decision")
        plt.show()  # should be similar due to LLN

        diff_opt_util = [
            max(self.util[i]) - min(self.util[i]) for i in range(self.num_decisions)
        ]
        plt.bar(range(1, self.num_decisions + 1), diff_opt_util)
        plt.title("Max difference of options for each decision")
        plt.show()  # should be different, some decisions more important than others

        plt.bar(
            range(1, self.num_decisions + 1),
            [max(self.util[i]) for i in range(self.num_decisions)],
        )
        plt.title("Max value of options for each decision")
        plt.show()


if __name__ == "__main__":
    de = DecisionEnvironment()
    de.visualize()
