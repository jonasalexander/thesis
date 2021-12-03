import numpy as np
import pandas as pd

df = pd.read_csv("../data/thesis_full.csv")

words_a = set(
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
words_z = set(
    [
        "dragon",
        "jacket",
        "energy",
        "magnet",
        "chimney",
        "bakery",
        "almond",
        "pumpkin",  # should have been "flower",
        "liquid",
        "casino",
        "church",
        "powder",
        "royalty",
    ]
)

# Drop and rename columns
columns = {
    "ResponseId": "rid",
    "Q11": "mturkID",
    "Q13": "paragraph_text_a",
    "Q31": "paragraph_text_z",
    "Q14_Page Submit": "paragraph_time_a",
    "Q32_Page Submit": "paragraph_time_z",
    "Q21": "word_a",
    "Q33": "word_z",
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
    "Q21B": "task_comprehension",
}
df = df[df["Finished"] != "False"]
df = df[columns.keys()]
df = df.rename(columns=columns)
df = df[2:].reset_index(drop=True)  # drop first two rows, qualtrics weirdness

# Add new columns
df["condition_a"] = df["word_z"].isna()
df["word"] = df["word_a"].combine(
    df["word_z"],
    lambda x, z: x.lower().strip() if type(x) is str else z.lower().strip(),
)
word_to_value = (
    lambda word: 26 - (ord(word[2].lower()) - 97)
    if word in words_a
    else ord(word[2].lower()) - 96
)
word_cols = ["cs" + str(i + 1) for i in range(13)] + [
    "memory" + str(i + 1) for i in range(13)
]
for col in word_cols:
    df[col] = df[col].apply(lambda x: x.strip() if type(x) is str else x)
df["value"] = df["word"].map(word_to_value)
df["bonus"] = df["value"].map(lambda value: value * 4)
df = df.rename(columns={"word": "choice"})

fixed_words = [
    ("R_2UVkjpw8nqNp8Fr", "memory1", "MAGNET"),
    ("R_2VmOr6DZAIP0iGb", "memory1", "coffee"),
    ("R_2VmOr6DZAIP0iGb", "memory6", "autumn"),
    ("R_2VmOr6DZAIP0iGb", "cs9", "firefly"),
    ("R_1jNRzCcvYInlI6C", "memory1", "pumpkin"),
    ("R_1jNRzCcvYInlI6C", "memory7", "chimney"),
    ("R_3pi2iY83AzmJ4oX", "memory1", "pumpkin"),
    ("R_3pi2iY83AzmJ4oX", "memory3", "royalty"),
    ("R_3pi2iY83AzmJ4oX", "memory4", "energy"),
    ("R_2uy1CEMczz3ObiL", "memory2", "canvas"),
    ("R_3RrtwB187v71qz1", "memory7", "cabinet"),
    ("R_1DINrjhbjLrwJTc", "memory6", "powder"),
    ("R_xyK3e6wWFB9NoCR", "memory6", "powder"),
    ("R_6tED4N15yxqiuoV", "memory7", "jacket"),
    ("R_beLjzWv6qkcnTTX", "memory8", "casino"),
    ("R_24w4JqAK6TkMmV8", "memory8", "liquid"),
    ("R_2R7Udgsy3HLIwE2", "cs1", "anxiety"),
    ("R_2czP4efH7RPc0oZ", "cs2", "chimney"),
    ("R_2D0lBznW12gmlsG", "cs4", "baptism"),
    ("R_9HaQxaDLsy8nJE5", "cs5", "powder"),
    ("R_3PcoY7M2FpV4kZ0", "cs5", "baptism"),
    ("R_3LZjdb5uyZohCYG", "cs5", "powder"),
    ("R_BDMrgaUZbEH1QrL", "cs9", "firefly"),
    ("R_3DcJu0qDWE0u2c1", "cs10", "anxiety"),
    ("R_3DcJu0qDWE0u2c1", "cs10", "anxiety"),
]
for rid, col, word in fixed_words:
    df.loc[df["rid"] == rid, col] = word

df.to_csv("../data/thesis_full_clean.csv", index=False)

# Turn wide into long format
# Need columns = ["last", "subject.id", "value", "order"]
long_df = pd.melt(
    df,
    id_vars=["rid", "condition_a"],
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
long_df = long_df.dropna()

long_df["raw_order"] = long_df["variable"].map(lambda x: int(x[2:]))

long_df = long_df.rename(columns={"rid": "subject.id"})
long_df = long_df.drop(columns="variable")

# Drop words not in the list
long_df["word"] = long_df["word"].map(lambda w: w if w in words_a | words_z else np.NaN)
long_df = long_df.dropna()

# Add value column
long_df["value"] = long_df["word"].map(word_to_value)

# Calculate order from raw_order
long_df["order"] = long_df.sort_values("raw_order").groupby("subject.id").cumcount() + 1

# Calculate last boolean
long_df["last"] = False
last_indexes = long_df.loc[long_df.groupby("subject.id")["order"].idxmax()].index
long_df.loc[last_indexes, "last"] = True

long_df.drop(columns=["raw_order", "word"])

# Write to disk
long_df.to_csv("../data/thesis_full_clean_long.csv")
