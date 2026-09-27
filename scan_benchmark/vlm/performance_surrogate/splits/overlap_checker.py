from itertools import combinations

import pandas as pd

if __name__ == "__main__":
    train_folds = {
        i: pd.read_csv(f"train_split_{i}.csv")
        for i in range(1, 6)
    }

    test_folds = {
        i: pd.read_csv(f"test_split_{i}.csv")
        for i in range(1, 6)
    }

    for i in range(1, 6):
        train_ids = set(train_folds[i]["config_id"])
        test_ids = set(test_folds[i]["config_id"])

        overlap = train_ids & test_ids

        print(
            f"Split {i}: "
            f"{len(train_ids)} train configs, "
            f"{len(test_ids)} test configs, "
            f"{len(overlap)} overlapping configs"
        )

        if overlap:
            print(f"Overlapping config IDs: {sorted(overlap)}")

    for i, j in combinations(range(1, 6), 2):
        ids_i = set(test_folds[i]["config_id"])
        ids_j = set(test_folds[j]["config_id"])

        overlap = ids_i & ids_j

        print(
            f"Test split {i} vs Test split {j}: "
            f"{len(overlap)} overlapping configs"
        )

        if overlap:
            print(f"Overlapping config IDs: {sorted(overlap)}")
