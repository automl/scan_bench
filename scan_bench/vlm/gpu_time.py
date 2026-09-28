import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("data.csv")

    df["gpu_hours"] = df["train_duration(s)"] * df["world_size"] / 3600

    result = df.groupby("train_gpu_name")["gpu_hours"].sum().reset_index()

    print(result)

    # time spent for the 36 configs in grid search
    grid_search_time_hours = 1129.28

    # estimated by the time tracked across each configs directories, which are not aggregated in data.csv
    downstream_eval = {'NVIDIA GeForce RTX 2080 Ti': 731.8379402000373, 'NVIDIA GeForce RTX 3080': 373.5332672378755,
                       'NVIDIA A100-SXM4-40GB': 827.6199665179518, 'H100': 3132.89}

    upstream_eval = {
        "NVIDIA GeForce RTX 3080": 13.467310171259774,
        "NVIDIA GeForce RTX 2080 Ti": 11.388316521048546,
        "NVIDIA L40S": 9.843000194297897,
        "NVIDIA A100-SXM4-40GB": 198.65985053433312,
        "H100": 158.96
    }
