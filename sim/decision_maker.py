import numpy as np
from scipy.special import softmax
import logging
from math import sqrt
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DecisionMaker(ABC):
    """Base class for decision makers"""

    def __init__(self, denv, cheap_expensive_rel_accuracy=3.0, cheap_sdev=4.0):
        self.denv = denv  # the decision environment this agent is in

        self.cheap_sdev = cheap_sdev
        self.expensive_sdev = cheap_sdev / float(cheap_expensive_rel_accuracy)

        self.cost = np.zeros(self.denv.num_decisions)
        self.generate_cost = 0.5
        self.cheap_cost = 1.0
        self.expensive_cost = 2.0

        # list of maps: indices in util -> (est util, est variance)
        self.est_util = [{} for _ in range(self.denv.num_decisions)]

    def decide_all(self, **kwargs):
        for i in range(self.denv.num_decisions):
            self.curr_dec_idx = i
            self.decide(**kwargs)

        self.score()

    @abstractmethod
    def decide(self):
        return

    def generate(self):
        gen_idc = list(self.est_util[self.curr_dec_idx].keys())
        # Zero out already generated options
        mult = np.ones(self.denv.num_options)
        if gen_idc:
            np.put(mult, gen_idc, 0)
        gen_util = np.multiply(mult, self.denv.util[self.curr_dec_idx])

        probs = softmax(gen_util)
        # probs = np.divide(gen_util, sum(gen_util)) # problem: negative util

        opt_idx = np.random.choice(self.denv.num_options, p=probs)

        if opt_idx in gen_idc:
            raise RuntimeError("Generated index that had already been generated.")

        self.cost[self.curr_dec_idx] += self.generate_cost

        # set prior
        mean = self.denv.opt_mean
        sdev = sum(self.denv.opt_vars) / float(len(self.denv.opt_vars))
        self.est_util[self.curr_dec_idx][opt_idx] = (mean, sdev)

        logging.debug(f"Generated option {opt_idx} for cost {self.generate_cost}")
        return opt_idx

    def estimate(self, opt_idx, mode="cheap"):
        old_est, old_sdev = self.est_util[self.curr_dec_idx][opt_idx]

        if mode == "cheap":
            sdev = self.cheap_sdev
            cost = self.cheap_cost
        elif mode == "expensive":
            sdev = self.expensive_sdev
            cost = self.expensive_cost
        else:
            raise NotImplementedError(f"Unrecognized mode: {mode}")

        new_est = np.random.normal(self.denv.util[self.curr_dec_idx][opt_idx], sdev)
        upd_val, upd_sdev = self.update(old_est, old_sdev, new_est, sdev)
        self.est_util[self.curr_dec_idx][opt_idx] = (upd_val, upd_sdev)

        self.cost[self.curr_dec_idx] += cost

        logging.debug(
            f"{mode.capitalize()}ly estimated option {opt_idx} and updated from {old_est} to {upd_val} with new reading {new_est}"
        )

        return upd_val

    def update(self, old_est, old_sdev, new_est, new_sdev):
        # Bayesian updating
        upd_val = (old_est * old_sdev ** 2 + new_est * self.cheap_sdev ** 2) / (
            old_sdev ** 2 + self.cheap_sdev ** 2
        )
        upd_sdev = sqrt(1.0 / (old_sdev ** (-2) + self.cheap_sdev ** (-2)))

        return (upd_val, upd_sdev)

    def get_best(self, mode, dec_idx=None):
        if dec_idx is None:
            return [
                self.get_best(mode, dec_idx=i) for i in range(self.denv.num_decisions)
            ]
        else:
            idx = sorted(self.est_util[dec_idx].items(), key=lambda x: x[1][0])[0][0]
            if mode == "idx":
                # sort by estimated util, the keys in self.est_util
                return idx
            elif mode == "val":
                return self.denv.util[dec_idx][idx]
            else:
                raise NotImplementedError(f"Unrecognized mode: {mode}")

    def score(self):

        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.suptitle(f"Results for {self.__class__}")
        axs[0].bar(range(self.denv.num_decisions), self.cost)
        axs[0].set_title("Cost for each decision")

        gross_util = self.get_best("val")
        net_util = [
            gross_util[i] - self.cost[i] for i in range(self.denv.num_decisions)
        ]

        logging.info(
            f"{self.__class__} finished with average cost {sum(self.cost)/float(self.denv.num_decisions)} in {self.denv.num_decisions} decisions."
        )
        logging.info(
            f"{self.__class__} finished with average gross utility {sum(gross_util)/float(self.denv.num_decisions)} in {self.denv.num_decisions} decisions."
        )
        logging.info(
            f"{self.__class__} finished with average net utility {sum(net_util)/float(self.denv.num_decisions)} in {self.denv.num_decisions} decisions."
        )

        axs[1].bar(range(self.denv.num_decisions), gross_util)
        axs[1].set_title("Gross utility for each decision")

        axs[2].bar(range(self.denv.num_decisions), net_util)
        axs[2].set_title("Net utility for each decision")
        plt.show()


class ThresholdCheapDecisionMaker(DecisionMaker):
    def decide(self, threshold=None, max_iters=10):
        iters = 0

        while iters < max_iters:
            idx = self.generate()

            if threshold is None:
                break

            else:

                est_util = self.estimate(idx, mode="cheap")

                if est_util <= threshold:
                    logging.debug(
                        f"Found option {idx} estimated at {est_util} better than threshold {threshold}"
                    )
                    break

            iters += 1


class FixedDecisionMaker(DecisionMaker):
    def decide(self, num_gen, num_cheap_est, num_expensive_est):
        for _ in range(num_gen):
            idx = self.generate()

        for _ in range(num_cheap_est):
            best_idx = self.get_best("idx", dec_idx=self.curr_dec_idx)
            self.estimate(best_idx, mode="cheap")

        for _ in range(num_expensive_est):
            best_idx = self.get_best("idx", dec_idx=self.curr_dec_idx)
            self.estimate(best_idx, mode="expensive")
