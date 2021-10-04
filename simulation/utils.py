from typing import List

import numpy as np
import pandas as pd


def generate_data(n: int, options: List[any], stop_proba_func: List[float]):
    """
    Options
    n: number of samples to draw (subjects)
    options: List of options and their values from which subjects are choosing
    stop_proba_func: Takes in df of subject's 12 months and must return 12 probabilities of stopping *after* the current one

    Returns
    df: pd.DataFrame
        With columns: last, subject.id, value, order
    """

    df = pd.DataFrame({"subject.id": range(n)})
    df = (
        df.merge(options, how="cross")
        .groupby("subject.id")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df["order"] = df.groupby("subject.id").cumcount() + 1
    df = df.rename(columns={"s2_value": "value"})

    def random_stopped(df):
        stop_proba = stop_proba_func(df)
        df["last"] = [
            True if x == 1 else False for x in np.random.binomial(1, stop_proba)
        ]
        for i, x in enumerate(df["last"]):
            if x:
                df = df.iloc[: i + 1]
                break
        return df

    return df.groupby("subject.id").apply(random_stopped).reset_index(drop=True)
