import os
import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

FEATURES_FILE = os.path.join(
    INPUT_DIR,
    "cleaned_features_final.csv"
)

LABEL_FILE = os.path.join(
    INPUT_DIR,
    "label.csv"
)


# ============================================================
# LOAD EXACT METABRIC USED BY DEEPHIT
# ============================================================
x = pd.read_csv(
    FEATURES_FILE
).values.astype(np.float32)

labels_df = pd.read_csv(
    LABEL_FILE
)

t = labels_df[
    "event_time"
].values.astype(np.float32)

e = labels_df[
    "label"
].values.astype(np.int32)


print("X shape:", x.shape)
print("T shape:", t.shape)
print("E shape:", e.shape)

print("Events:", np.unique(e, return_counts=True))


if not (
    len(x) == len(t)
    and len(x) == len(e)
):
    raise RuntimeError(
        "Features, times and labels have different lengths."
    )


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


indices = np.arange(
    len(x)
)


# ============================================================
# EXACT SAME SPLIT PROTOCOL AS DEEPHIT
#
# OUTER:
#   80% development
#   20% untouched test
#
# INNER:
#   80% train
#   20% validation
#
# seed = 1234 + out_itr
# ============================================================
for out_itr in range(5):

    seed = 1234 + out_itr

    dev_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed
    )

    train_idx, valid_idx = train_test_split(
        dev_idx,
        test_size=0.20,
        random_state=seed
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "metabric_common_outer_%d.h5" % out_itr
    )

    with h5py.File(
        output_file,
        "w"
    ) as f:

        for name, idx in [
            ("train", train_idx),
            ("valid", valid_idx),
            ("test", test_idx)
        ]:

            grp = f.create_group(
                name
            )

            grp.create_dataset(
                "x",
                data=x[idx]
            )

            grp.create_dataset(
                "t",
                data=t[idx]
            )

            grp.create_dataset(
                "e",
                data=e[idx]
            )

    print(
        "Outer %d -> train=%d valid=%d test=%d"
        % (
            out_itr,
            len(train_idx),
            len(valid_idx),
            len(test_idx)
        )
    )

    print(
        "Saved:",
        output_file
    )