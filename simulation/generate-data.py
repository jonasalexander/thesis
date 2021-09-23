import argparse

import numpy as np
import pandas as pd
from dmaker.decision_maker import DynamicDecisionMaker
from dmaker.environment import DecisionEnvironment, DecisionEnvironmentGrid

from utils import generate_data


def main(n: int, random: bool):

    if random:

        # random based on real-life average stopping probabilities based on number evaluated so far
        raw_df = pd.read_csv(
            "~/Desktop/thesis/data/Adams_experiment_cleaned_filtered.csv"
        )
        raw_df["last"] = ~raw_df["did_continue_eval"]

        options = raw_df.groupby("word").agg({"s2_value": "mean"}).reset_index()
        empirical_stop_proba = (
            raw_df.groupby("order").agg({"last": "mean"})["last"].values
        )

        def stop_proba_random(_):
            return empirical_stop_proba

        random = generate_data(n, options, stop_proba_random)
        random.to_csv(f"~/Desktop/thesis/data/generated_random_n={n}.csv")

    else:
        env = DecisionEnvironment(N=12, num_trials=n, sigma=0, mu=13, tau=8)
        dm = DynamicDecisionMaker(env=env, num_samples=1000, cost_eval=0.2)
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
