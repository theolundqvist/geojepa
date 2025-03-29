import glob
import os
from pathlib import Path

import pandas as pd
import torch.nn.functional
from autogluon.multimodal import MultiModalPredictor

torch.set_float32_matmul_precision("medium")

script_path = os.path.dirname(os.path.realpath(__file__))


def load_split(temp_dir):
    return pd.read_pickle(temp_dir + "/data.pkl")


def main():
    dir = script_path + "/autogluon_data/"
    dataset = "tiny_test"
    df = load_split(dir + "/" + dataset)
    # Initialize the ImagePredictor for regression
    latest = sorted(list(glob.glob("AutogluonModels/**/*.ckpt")), reverse=True)
    print(f"picking {latest[0]}")
    predictor = MultiModalPredictor.load(latest[0])

    print(predictor.fit_summary())
    print(f"Testing the model on {dataset}...")
    res = predictor.predict(df, as_pandas=False)
    df["pred"] = res
    df["image"] = df["image"].apply(lambda image: Path(image).name)
    print(
        "MSE",
        torch.nn.functional.mse_loss(
            torch.tensor(df["pred"]), torch.tensor(df["label"])
        ),
    )
    print(
        "MAE",
        torch.nn.functional.l1_loss(
            torch.tensor(df["pred"]), torch.tensor(df["label"])
        ),
    )
    print(df[df["label"] == 0][:5])
    print(df[df["label"] != 0][:10])

    performance = predictor.evaluate(df)
    print("Test tiny performance:")
    print(performance)


if __name__ == "__main__":
    main()
