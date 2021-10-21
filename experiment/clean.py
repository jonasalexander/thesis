import numpy as np
import pandas as pd

df = pd.read_csv("../data/thesis_sample.csv")

# Drop and rename columns
columns = {
    "ResponseId": "rid",
    "Q11": "mturkID",
    "Q13": "paragraph_text",
    "Q14_Page Submit": "paragraph_time",
    "Q21": "word",
    "Q18_1": "cs1",
    "Q18_2": "cs2",
    "Q18_3": "cs3",
    "Q18_4": "cs4",
    "Q18_5": "cs5",
    "Q18_6": "cs6",
    "Q18_7": "cs7",
    "Q18_8": "cs8",
    "Q18_9": "cs9",
    "Q18_10": "cs10",
    "Q18_11": "cs11",
    "Q18_12": "cs12",
    "Q18_13": "cs13",
    "Q19_Page Submit": "cs_time",
    "Q25_1": "memory1",
    "Q25_3": "memory3",
    "Q25_2": "memory2",
    "Q25_4": "memory4",
    "Q25_5": "memory5",
    "Q25_6": "memory6",
    "Q25_7": "memory7",
    "Q25_8": "memory8",
    "Q25_9": "memory9",
    "Q25_10": "memory10",
    "Q25_11": "memory11",
    "Q25_12": "memory12",
    "Q25_13": "memory13",
    "Q26_Page Submit": "memory_time",
    "Q16": "feedback",
}
df = df[columns.keys()]
df = df.rename(columns=columns)
df = df[2:].reset_index(drop=True)  # drop first two rows, qualtrics weirdness

# Add new columns
df["letter_order"] = df["word"].map(lambda word: ord(word[2].lower()) - 96)
df["bonus"] = df["letter_order"].map(lambda letter: (26 - letter) * 4)
df = df.rename(columns={"word": "choice"})

df.to_csv("../data/thesis_sample_clean.csv", index=False)

# Turn wide into long format
# Need columns = ["last", "subject.id", "value", "order"]
wide_df = pd.melt(
    df,
    id_vars=["rid"],
    value_vars=[
        "cs1",
        "cs2",
        "cs3",
        "cs4",
        "cs5",
        "cs6",
        "cs7",
        "cs8",
        "cs9",
        "cs10",
        "cs11",
        "cs12",
        "cs13",
    ],
    value_name="word",
)
wide_df = wide_df.dropna()

wide_df["raw_order"] = wide_df["variable"].map(lambda x: int(x[2:]))
wide_df["word"] = wide_df["word"].map(lambda w: w.lower())

wide_df = wide_df.rename(columns={"rid": "subject.id"})
wide_df = wide_df.drop(columns="variable")


# Drop words not in the list
words = set(
    [
        "wizard",
        "anxiety",
        "javelin",
        "autumn",
        "firefly",
        "baptism",
        "canvas",
        "silver",
        "injury",
        "school",
        "coffee",
        "kidney",
        "cabinet",
    ]
)
wide_df["word"] = wide_df["word"].map(lambda w: w if w in words else np.NaN)
wide_df = wide_df.dropna()

# Add value column
wide_df["value"] = wide_df["word"].map(lambda word: 26 - (ord(word[2].lower()) - 97))

# Calculate order from raw_order
wide_df["order"] = wide_df.sort_values("raw_order").groupby("subject.id").cumcount() + 1

# Calculate last boolean
wide_df["last"] = False
last_indexes = wide_df.loc[wide_df.groupby("subject.id")["order"].idxmax()].index
wide_df.loc[last_indexes, "last"] = True

wide_df.drop(columns=["raw_order", "word"])

# Write to disk
wide_df.to_csv("../data/thesis_sample_clean_wide.csv")