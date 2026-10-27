import os
import json
import pytest
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.preprocess import (
    EXPECTED_FEATURES,
    CATEGORY_MAPPING,
    CATEGORY_TO_ID,
    SUBTYPE_TO_ID,
    parse_filename_label,
    validate_schema,
    clean_chunk,
    PreprocessingConfig,
    Preprocessor,
    discover_csv_files
)


def test_label_mapping_known_attacks():
    """Verify filename label inference for diverse attack types and benign traffic."""
    test_cases = [
        ("Benign_train.pcap.csv", "Benign", "Benign", 0),
        ("ARP_Spoofing_test.pcap.csv", "ARP_Spoofing", "ARP Spoofing", 1),
        ("MQTT-DDoS-Connect_Flood_train.pcap.csv", "MQTT-DDoS-Connect_Flood", "MQTT attacks", 1),
        ("MQTT-DoS-Publish_Flood_test.pcap.csv", "MQTT-DoS-Publish_Flood", "MQTT attacks", 1),
        ("MQTT-Malformed_Data_train.pcap.csv", "MQTT-Malformed_Data", "MQTT attacks", 1),
        ("Recon-Port_Scan_train.pcap.csv", "Recon-Port_Scan", "Reconnaissance", 1),
        ("Recon-Ping_Sweep_test.pcap.csv", "Recon-Ping_Sweep", "Reconnaissance", 1),
        ("TCP_IP-DDoS-ICMP4_train.pcap.csv", "TCP_IP-DDoS-ICMP", "TCP/IP DoS/DDoS", 1),
        ("TCP_IP-DoS-SYN_test.pcap.csv", "TCP_IP-DoS-SYN", "TCP/IP DoS/DDoS", 1),
    ]

    for filename, expected_sub, expected_cat, expected_attack in test_cases:
        meta = parse_filename_label(filename)
        assert meta["subtype"] == expected_sub
        assert meta["category"] == expected_cat
        assert meta["is_attack"] == expected_attack
        assert meta["category_id"] == CATEGORY_TO_ID[expected_cat]
        assert meta["subtype_id"] == SUBTYPE_TO_ID[expected_sub]


def test_label_mapping_invalid_filename():
    """Verify that an unrecognized attack name raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognized attack subtype"):
        parse_filename_label("Unknown_ZeroDay_Attack_train.pcap.csv")


def test_feature_schema_validation():
    """Verify that validate_schema correctly accepts valid schemas and rejects invalid ones."""
    # Valid DataFrame
    valid_df = pd.DataFrame(np.zeros((2, 45)), columns=EXPECTED_FEATURES)
    validate_schema(valid_df, "dummy_valid.csv")  # Should not raise

    # Missing column
    invalid_missing = valid_df.drop(columns=["Duration"])
    with pytest.raises(ValueError, match=r"(?s)Schema mismatch.*Missing features"):
        validate_schema(invalid_missing, "dummy_missing.csv")

    # Extra unexpected column
    invalid_extra = valid_df.copy()
    invalid_extra["Extra_Col"] = 1.0
    with pytest.raises(ValueError, match=r"(?s)Schema mismatch.*Unexpected extra features"):
        validate_schema(invalid_extra, "dummy_extra.csv")


def test_clean_chunk_invalid_values():
    """Verify that infinities and NaNs are properly cleaned and tracked."""
    data = np.zeros((4, 45))
    df = pd.DataFrame(data, columns=EXPECTED_FEATURES)

    # Inject infs and nans
    df.loc[0, "Rate"] = np.inf
    df.loc[1, "Rate"] = -np.inf
    df.loc[2, "Duration"] = np.nan

    cleaned, inf_cnt, nan_cnt = clean_chunk(df)
    assert inf_cnt == 2
    assert nan_cnt == 1
    assert not np.isinf(cleaned.values).any()
    assert not np.isnan(cleaned.values).any()
    assert cleaned.loc[0, "Rate"] == 0.0
    assert cleaned.loc[1, "Rate"] == 0.0
    assert cleaned.loc[2, "Duration"] == 0.0
    assert cleaned.dtypes["Rate"] == np.float32


def test_train_only_fitting_and_no_data_leakage():
    """Verify that StandardScaler fits ONLY on training data and test data does not affect parameters."""
    config = PreprocessingConfig(chunk_size=10, sample_rows_per_file=None)
    preprocessor = Preprocessor(config)

    # Train data: mean = 10.0, variance = 4.0 (std = 2.0)
    np.random.seed(42)
    train_vals = np.random.normal(loc=10.0, scale=2.0, size=(100, 45)).astype(np.float32)
    train_df = pd.DataFrame(train_vals, columns=EXPECTED_FEATURES)

    # Partially fit on train
    preprocessor.scaler.partial_fit(train_df.values)
    preprocessor.fitted = True

    fitted_mean_initial = preprocessor.scaler.mean_.copy()
    fitted_var_initial = preprocessor.scaler.var_.copy()

    # Test data with completely different distribution (mean = 100.0)
    test_vals = np.random.normal(loc=100.0, scale=10.0, size=(50, 45)).astype(np.float32)
    test_df = pd.DataFrame(test_vals, columns=EXPECTED_FEATURES)

    # Transform test data
    scaled_test = preprocessor.transform_inference(test_df)

    # Verify that the scaler parameters did NOT change
    assert np.allclose(preprocessor.scaler.mean_, fitted_mean_initial)
    assert np.allclose(preprocessor.scaler.var_, fitted_var_initial)

    # Verify test scaling math: (x - train_mean) / train_std
    expected_scaled_0_0 = (test_vals[0, 0] - fitted_mean_initial[0]) / np.sqrt(fitted_var_initial[0])
    assert np.isclose(scaled_test[0, 0], expected_scaled_0_0, atol=1e-5)


def test_chunk_reading_and_parquet_streaming(tmp_path):
    """Verify streaming chunked processing and incremental Parquet generation."""
    # Create two temporary mock CSV files
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    
    file1 = train_dir / "Benign_train.pcap.csv"
    file2 = train_dir / "ARP_Spoofing_train.pcap.csv"

    data1 = pd.DataFrame(np.random.randn(25, 45).astype(np.float32), columns=EXPECTED_FEATURES)
    data2 = pd.DataFrame(np.random.randn(25, 45).astype(np.float32), columns=EXPECTED_FEATURES)
    data1.to_csv(file1, index=False)
    data2.to_csv(file2, index=False)

    config = PreprocessingConfig(
        train_dir=str(train_dir),
        test_dir=str(train_dir),
        output_dir=str(tmp_path / "out"),
        artifact_dir=str(tmp_path / "artifacts"),
        chunk_size=10,
        sample_rows_per_file=None
    )

    preprocessor = Preprocessor(config)
    preprocessor.fit([str(file1), str(file2)])

    out_parquet = str(tmp_path / "out" / "train.parquet")
    stats = preprocessor.process_and_save_parquet([str(file1), str(file2)], out_parquet, split_name="train")

    assert stats["total_rows"] == 50
    assert os.path.exists(out_parquet)

    # Read back parquet
    df_read = pq.read_table(out_parquet).to_pandas()
    assert df_read.shape == (50, 45 + 6)  # 45 features + 6 metadata columns
    assert "label_raw" in df_read.columns
    assert "label_category" in df_read.columns
    assert "category_id" in df_read.columns
    assert set(df_read["label_category"].unique()) == {"Benign", "ARP Spoofing"}


def test_save_and_load_inference(tmp_path):
    """Verify that saved preprocessing artifacts can be loaded and used for inference without refitting."""
    artifact_dir = str(tmp_path / "artifacts")
    config = PreprocessingConfig(artifact_dir=artifact_dir)
    preprocessor = Preprocessor(config)

    # Fit on small sample
    np.random.seed(123)
    sample_data = np.random.randn(20, 45).astype(np.float32)
    preprocessor.scaler.partial_fit(sample_data)
    preprocessor.fitted = True
    preprocessor.save(artifact_dir)

    # Check files created
    assert os.path.exists(os.path.join(artifact_dir, "scaler.pkl"))
    assert os.path.exists(os.path.join(artifact_dir, "schema.json"))
    assert os.path.exists(os.path.join(artifact_dir, "label_mappings.json"))
    assert os.path.exists(os.path.join(artifact_dir, "config.json"))

    # Load into new instance
    loaded = Preprocessor.load(artifact_dir)
    assert loaded.fitted

    # Transform new data
    new_df = pd.DataFrame(sample_data[:5], columns=EXPECTED_FEATURES)
    orig_scaled = preprocessor.transform_inference(new_df)
    loaded_scaled = loaded.transform_inference(new_df)

    assert np.allclose(orig_scaled, loaded_scaled)


def test_reproducibility():
    """Verify that running preprocessing on identical inputs yields identical scalers."""
    np.random.seed(999)
    data = np.random.randn(50, 45).astype(np.float32)

    pre1 = Preprocessor(PreprocessingConfig())
    pre1.scaler.partial_fit(data)

    pre2 = Preprocessor(PreprocessingConfig())
    pre2.scaler.partial_fit(data)

    assert np.allclose(pre1.scaler.mean_, pre2.scaler.mean_)
    assert np.allclose(pre1.scaler.var_, pre2.scaler.var_)
