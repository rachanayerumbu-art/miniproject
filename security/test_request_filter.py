import numpy as np
import pandas as pd

try:
    from security.request_filter import predict_request
except ModuleNotFoundError:
    from request_filter import predict_request

try:
    from security.baseline import TEST_FILE
except ModuleNotFoundError:
    from baseline import TEST_FILE


def test_normal_request_is_allowed():
    test_data = pd.read_csv(TEST_FILE)
    normal_rows = test_data.loc[test_data["Label"] == 0]

    assert not normal_rows.empty, "Independent test data contains no normal records"

    record = normal_rows.iloc[0].drop(labels=["Label"])
    result = predict_request(record)

    assert result["status"] == "allowed", result
    assert result["prediction"] == 0, result
    print("PASS: Normal traffic is allowed")


def test_ddos_request_is_blocked():
    test_data = pd.read_csv(TEST_FILE)
    ddos_rows = test_data.loc[test_data["Label"] == 1]

    assert not ddos_rows.empty, "Independent test data contains no DDoS records"

    record = ddos_rows.iloc[0].drop(labels=["Label"])
    result = predict_request(record)

    assert result["status"] == "blocked", result
    assert result["prediction"] == 1, result
    print("PASS: DDoS traffic is blocked")


def test_mapping_with_missing_features_is_invalid():
    test_data = pd.read_csv(TEST_FILE)
    record = test_data.drop(columns=["Label"]).iloc[0].to_dict()
    record.pop(next(iter(record)))

    result = predict_request(record)

    assert result["status"] == "invalid", result
    assert result["prediction"] is None, result
    assert "Missing feature values" in result["reason"], result
    print("PASS: Missing feature values are rejected")


def test_invalid_feature_values_are_rejected():
    test_data = pd.read_csv(TEST_FILE)
    record = test_data.drop(columns=["Label"]).iloc[0].to_dict()

    invalid_records = []
    for value in (np.nan, np.inf, -np.inf, "not-a-number"):
        invalid_record = record.copy()
        invalid_record[next(iter(invalid_record))] = value
        invalid_records.append(invalid_record)

    for invalid_record in invalid_records:
        result = predict_request(invalid_record)
        assert result["status"] == "invalid", result
        assert result["prediction"] is None, result

    print("PASS: NaN, infinite, and non-numeric feature values are rejected")


def test_extra_features_and_wrong_sequence_lengths_are_rejected():
    test_data = pd.read_csv(TEST_FILE)
    record = test_data.drop(columns=["Label"]).iloc[0].to_dict()

    extra_record = record.copy()
    extra_record["unexpected_feature"] = 1.0
    extra_result = predict_request(extra_record)

    assert extra_result["status"] == "invalid", extra_result
    assert "Unexpected features" in extra_result["reason"], extra_result

    short_result = predict_request(list(record.values())[:-1])
    assert short_result["status"] == "invalid", short_result
    assert "features" in short_result["reason"], short_result

    print("PASS: Extra features and incorrect sequence lengths are rejected")


if __name__ == "__main__":
    test_normal_request_is_allowed()
    test_ddos_request_is_blocked()
    test_mapping_with_missing_features_is_invalid()
    test_invalid_feature_values_are_rejected()
    test_extra_features_and_wrong_sequence_lengths_are_rejected()
    print("\nAll request filter tests passed!")
