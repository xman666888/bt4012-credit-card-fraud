import json

import pandas as pd

from bt4012_fraud.baseline import run_baselines


def test_baseline_smoke_run(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="h")
    frame = pd.DataFrame(
        {
            "trans_date_trans_time": dates,
            "is_fraud": [int(index % 7 == 0) for index in range(40)],
            "amt": [10.0 + index for index in range(40)],
            "category": ["grocery_pos" if index % 2 else "shopping_net" for index in range(40)],
            "gender": ["F" if index % 2 else "M" for index in range(40)],
            "state": ["CA"] * 40,
            "city_pop": [1000] * 40,
            "dob": ["1990-01-01"] * 40,
            "lat": [34.0] * 40,
            "long": [-118.0] * 40,
            "merch_lat": [34.1] * 40,
            "merch_long": [-118.1] * 40,
        }
    )
    train_csv = tmp_path / "fraudTrain.csv"
    output_dir = tmp_path / "baseline"
    frame.to_csv(train_csv, index=False)

    run_baselines(train_csv, output_dir, validation_fraction=0.20)

    results = json.loads((output_dir / "metrics.json").read_text())
    assert set(results["models"]) == {"dummy_prior", "logistic_regression"}
    assert (output_dir / "dummy_prior.joblib").exists()
    assert (output_dir / "logistic_regression.joblib").exists()
