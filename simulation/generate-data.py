import argparse

import numpy as np
import pandas as pd
from dmaker.decision_maker import DynamicDecisionMaker
from dmaker.environment import DecisionEnvironment, DecisionEnvironmentGrid

from utils import generate_data


def main(n: int, random: bool):

    N = 12
    sigma = 100
    mu = 13
    tau = 8
    cost_eval = 1
    num_samples = 1000

    if random:

        # random based on average stopping probabilities of model
        env = DecisionEnvironment(N=N, num_trials=n, sigma=sigma, mu=mu, tau=tau)
        dm = DynamicDecisionMaker(env=env, num_samples=num_samples, cost_eval=cost_eval)
        dm.decide()

        dm.experiment_data["last"] = dm.experiment_data["last"].astype(int)
        model_stop_proba = dm.experiment_data.groupby("order").agg({"last": "mean"})

        def stop_proba_random(_):
            return model_stop_proba.append(
                pd.DataFrame({"last": [1] * (N - len(model_stop_proba) + 1)})
            )

        options = pd.Series([x * 2 + 1 for x in range(13)], name="options")

        random = generate_data(n, options, stop_proba_random)
        random.to_csv(f"~/Desktop/thesis/data/generated_random_n={n}.csv")

    else:
        env = DecisionEnvironment(N=N, num_trials=n, sigma=sigma, mu=mu, tau=tau)
        dm = DynamicDecisionMaker(env=env, num_samples=num_samples, cost_eval=cost_eval)
        dm.decide()
        dm.experiment_data.to_csv(
            f"~/Desktop/thesis/data/generated_gaussian_optimal_n={n}.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="number of subjects to simulate", type=int)
    parser.add_argument(
        "--random",
        help="whether to use random (null) model or statistical model",
        action="store_true",
    )

    args = parser.parse_args()

    main(args.n, args.random)
