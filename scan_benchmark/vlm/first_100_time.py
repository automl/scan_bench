import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("data.csv")

    df["gpu_hours"] = df["train_duration(s)"] * df["world_size"] / 3600
    df_a100 = df[df["train_gpu_name"] == "NVIDIA A100-SXM4-40GB"]
    total_runtime = df_a100["gpu_hours"].sum()

    num_configs = df_a100["config_id"].nunique()
    avg_per_config = total_runtime / num_configs
    runtime_100_configs = avg_per_config * 100

    print("avg per config:", avg_per_config)
    print("estimated for 100 configs:", runtime_100_configs)
