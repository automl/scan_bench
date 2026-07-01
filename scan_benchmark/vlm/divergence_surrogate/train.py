from scan_benchmark.vlm.divergence_surrogate.data import DivergenceDataset
from scan_benchmark.vlm.divergence_surrogate.predictors.xgb import BinaryBaggingEnsemble

if __name__ == "__main__":
    dataset = DivergenceDataset(
        train_csv_path="scan_benchmark/vlm/divergence_surrogate/splits/train.csv",
        test_csv_path="scan_benchmark/vlm/divergence_surrogate/splits/test.csv",
    )

    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    X_all, y_all = dataset.get_all_data()

    ensemble_model = BinaryBaggingEnsemble(load=False)
    ensemble_model._BinaryBaggingEnsemble__fit(X_train, y_train)
    ensemble_model.validate(X_test, y_test)

    # used to store final model
    ensemble_model._BinaryBaggingEnsemble__fit(X_all, y_all, True)