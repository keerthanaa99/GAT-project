"""
Data Preprocessing Pipeline for GAT-XIDS (CICIoMT2024 Dataset)
==============================================================
Provides memory-efficient, chunked preprocessing, schema validation,
filename-based label inference, out-of-core standardization (train-only fitting),
and incremental Parquet serialization.
"""

import os
import glob
import re
import json
import pickle
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Generator, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gat_xids.preprocess")

# 45 Numerical Features verified in CICIoMT2024 CSVs
EXPECTED_FEATURES: List[str] = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight"
]

# Mapping from 19 granular attack subtypes to 5 broad categories
CATEGORY_MAPPING: Dict[str, str] = {
    "Benign": "Benign",
    "ARP_Spoofing": "ARP Spoofing",
    "MQTT-DDoS-Connect_Flood": "MQTT attacks",
    "MQTT-DDoS-Publish_Flood": "MQTT attacks",
    "MQTT-DoS-Connect_Flood": "MQTT attacks",
    "MQTT-DoS-Publish_Flood": "MQTT attacks",
    "MQTT-Malformed_Data": "MQTT attacks",
    "Recon-OS_Scan": "Reconnaissance",
    "Recon-Ping_Sweep": "Reconnaissance",
    "Recon-Port_Scan": "Reconnaissance",
    "Recon-VulScan": "Reconnaissance",
    "TCP_IP-DDoS-ICMP": "TCP/IP DoS/DDoS",
    "TCP_IP-DDoS-SYN": "TCP/IP DoS/DDoS",
    "TCP_IP-DDoS-TCP": "TCP/IP DoS/DDoS",
    "TCP_IP-DDoS-UDP": "TCP/IP DoS/DDoS",
    "TCP_IP-DoS-ICMP": "TCP/IP DoS/DDoS",
    "TCP_IP-DoS-SYN": "TCP/IP DoS/DDoS",
    "TCP_IP-DoS-TCP": "TCP/IP DoS/DDoS",
    "TCP_IP-DoS-UDP": "TCP/IP DoS/DDoS",
}

# 5 Broad Category Encodings
CATEGORY_TO_ID: Dict[str, int] = {
    "Benign": 0,
    "ARP Spoofing": 1,
    "MQTT attacks": 2,
    "Reconnaissance": 3,
    "TCP/IP DoS/DDoS": 4
}
ID_TO_CATEGORY: Dict[int, str] = {v: k for k, v in CATEGORY_TO_ID.items()}

# 19 Subtype Encodings (Benign is 0, rest sorted)
ALL_SUBTYPES: List[str] = sorted(list(CATEGORY_MAPPING.keys()))
# Ensure Benign is index 0
ALL_SUBTYPES.remove("Benign")
ALL_SUBTYPES = ["Benign"] + ALL_SUBTYPES
SUBTYPE_TO_ID: Dict[str, int] = {s: idx for idx, s in enumerate(ALL_SUBTYPES)}
ID_TO_SUBTYPE: Dict[int, str] = {v: k for k, v in SUBTYPE_TO_ID.items()}


def parse_filename_label(filepath: str) -> Dict[str, Any]:
    """
    Infers raw stem, attack subtype, and broad category from a CICIoMT2024 CSV path.
    
    Examples:
        - ARP_Spoofing_train.pcap.csv -> stem: ARP_Spoofing, subtype: ARP_Spoofing, category: ARP Spoofing
        - TCP_IP-DDoS-ICMP1_train.pcap.csv -> stem: TCP_IP-DDoS-ICMP1, subtype: TCP_IP-DDoS-ICMP, category: TCP/IP DoS/DDoS
    """
    basename = os.path.basename(filepath)
    # Strip capture/split extensions: _train.pcap.csv, _test.pcap.csv, .pcap.csv, .csv
    stem = re.sub(r"(_train|_test)?\.pcap\.csv$", "", basename, flags=re.IGNORECASE)
    stem = re.sub(r"\.csv$", "", stem, flags=re.IGNORECASE)
    
    # Granular subtype strips capture partition numbers (e.g. ICMP1 -> ICMP)
    subtype = re.sub(r"\d+$", "", stem)
    
    if subtype not in CATEGORY_MAPPING:
        raise ValueError(
            f"Unrecognized attack subtype '{subtype}' parsed from file '{basename}'. "
            f"Expected one of: {list(CATEGORY_MAPPING.keys())}"
        )
        
    category = CATEGORY_MAPPING[subtype]
    is_attack = 0 if category == "Benign" else 1
    
    return {
        "raw_stem": stem,
        "subtype": subtype,
        "category": category,
        "subtype_id": SUBTYPE_TO_ID[subtype],
        "category_id": CATEGORY_TO_ID[category],
        "is_attack": is_attack
    }


def discover_csv_files(directory: str) -> List[str]:
    """Discovers all CSV files recursively in the given directory without hardcoding names."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Configured dataset directory does not exist: {directory}")
    pattern = os.path.join(directory, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    files = sorted([f for f in files if os.path.isfile(f)])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return files


def validate_schema(df: pd.DataFrame, filepath: str) -> None:
    """Verifies that the DataFrame contains the expected 45 numerical features in exact order."""
    cols = list(df.columns)
    if cols != EXPECTED_FEATURES:
        missing = [c for c in EXPECTED_FEATURES if c not in cols]
        extra = [c for c in cols if c not in EXPECTED_FEATURES]
        raise ValueError(
            f"Schema mismatch in file: {filepath}\n"
            f"Missing features ({len(missing)}): {missing}\n"
            f"Unexpected extra features ({len(extra)}): {extra}"
        )


def clean_chunk(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Cleans infinite and null values from numerical features.
    
    Rules:
    - Replaces positive/negative infinities with NaN.
    - Fills NaNs with 0.0 (baseline neutral feature value in network flows).
    - Ensures all features are float32.
    
    Returns:
        (cleaned_df, inf_count, nan_count)
    """
    features_df = df[EXPECTED_FEATURES].copy()
    
    # Count original nans before inf replacement
    nan_mask = features_df.isna().values
    nan_count = int(np.sum(nan_mask))

    # Count infs and replace with nan
    inf_mask = np.isinf(features_df.values)
    inf_count = int(np.sum(inf_mask))
    if inf_count > 0:
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
    # Impute missing values with 0.0
    if nan_count > 0 or inf_count > 0:
        features_df = features_df.fillna(0.0)
        
    features_df = features_df.astype(np.float32)
    return features_df, inf_count, nan_count


@dataclass
class PreprocessingConfig:
    train_dir: str = "WIFI & MQTT/attacks/CSV/train"
    test_dir: str = "WIFI & MQTT/attacks/CSV/test"
    output_dir: str = "results/preprocessed"
    artifact_dir: str = "models/preprocessing"
    chunk_size: int = 10000
    sample_rows_per_file: Optional[int] = 1000  # None for full dataset
    drop_constant_features: bool = False
    variance_threshold: float = 1e-9


class Preprocessor:
    """
    Production preprocessor supporting out-of-core standardization, schema enforcement,
    and incremental Parquet streaming.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False
        self.constant_features: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.fit_metrics: Dict[str, Any] = {}

    def fit(self, train_files: List[str]) -> "Preprocessor":
        """
        Fits StandardScaler incrementally on training CSV chunks.
        Never loads all files into RAM simultaneously.
        """
        logger.info(f"Fitting Preprocessor on {len(train_files)} training CSV files...")
        total_rows = 0
        total_infs = 0
        total_nans = 0
        class_distribution: Dict[str, int] = {}

        for file_idx, fpath in enumerate(train_files):
            file_meta = parse_filename_label(fpath)
            cat = file_meta["category"]
            rows_read_file = 0

            for chunk in pd.read_csv(fpath, chunksize=self.config.chunk_size):
                validate_schema(chunk, fpath)
                
                if self.config.sample_rows_per_file is not None:
                    remaining = self.config.sample_rows_per_file - rows_read_file
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining]

                clean_feats, inf_cnt, nan_cnt = clean_chunk(chunk)
                total_infs += inf_cnt
                total_nans += nan_cnt

                self.scaler.partial_fit(clean_feats.values)
                
                rows_in_chunk = len(clean_feats)
                total_rows += rows_in_chunk
                rows_read_file += rows_in_chunk
                class_distribution[cat] = class_distribution.get(cat, 0) + rows_in_chunk

                if self.config.sample_rows_per_file is not None and rows_read_file >= self.config.sample_rows_per_file:
                    break

        self.fitted = True

        # Check for non-informative / constant features across training set
        for idx, feat in enumerate(EXPECTED_FEATURES):
            var = float(self.scaler.var_[idx])
            mean = float(self.scaler.mean_[idx])
            self.feature_stats[feat] = {"mean": mean, "variance": var}
            if var <= self.config.variance_threshold:
                self.constant_features.append(feat)

        if self.constant_features:
            logger.info(
                f"Identified {len(self.constant_features)} constant features (var <= {self.config.variance_threshold}): "
                f"{self.constant_features}. Features are preserved in model input schema."
            )

        self.fit_metrics = {
            "total_rows_fitted": total_rows,
            "total_files_fitted": len(train_files),
            "total_infs_handled": total_infs,
            "total_nans_handled": total_nans,
            "class_distribution": class_distribution,
            "is_sampled": self.config.sample_rows_per_file is not None,
            "sample_rows_per_file": self.config.sample_rows_per_file
        }
        logger.info(f"Fit complete. Rows: {total_rows:,}. Class breakdown: {class_distribution}")
        return self

    def transform_inference(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw prediction-time DataFrame into scaled feature matrix without refitting.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before calling transform_inference.")
        validate_schema(df, "inference_data")
        clean_feats, _, _ = clean_chunk(df)
        return self.scaler.transform(clean_feats.values).astype(np.float32)

    def process_and_save_parquet(
        self,
        files: List[str],
        output_parquet_path: str,
        split_name: str
    ) -> Dict[str, Any]:
        """
        Transforms files chunk-by-chunk using the fitted scaler and writes incrementally
        to a single Parquet file.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming dataset splits.")

        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        logger.info(f"Processing and streaming {split_name} split to: {output_parquet_path}")

        # Build Arrow Schema
        fields = [pa.field(feat, pa.float32()) for feat in EXPECTED_FEATURES]
        fields.extend([
            pa.field("label_raw", pa.string()),
            pa.field("label_subtype", pa.string()),
            pa.field("label_category", pa.string()),
            pa.field("subtype_id", pa.int32()),
            pa.field("category_id", pa.int32()),
            pa.field("is_attack", pa.int32()),
        ])
        arrow_schema = pa.schema(fields)

        total_rows = 0
        total_infs = 0
        total_nans = 0
        class_dist: Dict[str, int] = {}
        subtype_dist: Dict[str, int] = {}

        writer = pq.ParquetWriter(output_parquet_path, schema=arrow_schema, compression="snappy")

        try:
            for fpath in files:
                meta = parse_filename_label(fpath)
                cat = meta["category"]
                sub = meta["subtype"]
                rows_read_file = 0

                for chunk in pd.read_csv(fpath, chunksize=self.config.chunk_size):
                    validate_schema(chunk, fpath)

                    if self.config.sample_rows_per_file is not None:
                        remaining = self.config.sample_rows_per_file - rows_read_file
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining]

                    clean_feats, inf_cnt, nan_cnt = clean_chunk(chunk)
                    total_infs += inf_cnt
                    total_nans += nan_cnt

                    # Apply training scaler
                    scaled_values = self.scaler.transform(clean_feats.values).astype(np.float32)

                    # Build output DataFrame for chunk
                    chunk_out = pd.DataFrame(scaled_values, columns=EXPECTED_FEATURES)
                    n_rows = len(chunk_out)
                    chunk_out["label_raw"] = meta["raw_stem"]
                    chunk_out["label_subtype"] = sub
                    chunk_out["label_category"] = cat
                    chunk_out["subtype_id"] = np.int32(meta["subtype_id"])
                    chunk_out["category_id"] = np.int32(meta["category_id"])
                    chunk_out["is_attack"] = np.int32(meta["is_attack"])

                    # Stream chunk to Parquet
                    table = pa.Table.from_pandas(chunk_out, schema=arrow_schema, preserve_index=False)
                    writer.write_table(table)

                    total_rows += n_rows
                    rows_read_file += n_rows
                    class_dist[cat] = class_dist.get(cat, 0) + n_rows
                    subtype_dist[sub] = subtype_dist.get(sub, 0) + n_rows

                    if self.config.sample_rows_per_file is not None and rows_read_file >= self.config.sample_rows_per_file:
                        break
        finally:
            writer.close()

        logger.info(
            f"Successfully wrote {split_name} Parquet ({total_rows:,} rows) -> {output_parquet_path}"
        )

        return {
            "split": split_name,
            "output_file": output_parquet_path,
            "total_rows": total_rows,
            "total_files": len(files),
            "total_infs": total_infs,
            "total_nans": total_nans,
            "category_distribution": class_dist,
            "subtype_distribution": subtype_dist,
            "is_sampled": self.config.sample_rows_per_file is not None,
            "sample_rows_per_file": self.config.sample_rows_per_file
        }

    def save(self, artifact_dir: str) -> None:
        """Saves fitted scaler, mappings, schema, and config artifacts for inference and reproducibility."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted Preprocessor.")
        os.makedirs(artifact_dir, exist_ok=True)

        # 1. Scaler
        scaler_path = os.path.join(artifact_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # 2. Schema and feature stats
        schema_path = os.path.join(artifact_dir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump({
                "features": EXPECTED_FEATURES,
                "num_features": len(EXPECTED_FEATURES),
                "constant_features": self.constant_features,
                "feature_statistics": self.feature_stats
            }, f, indent=2)

        # 3. Label mappings
        labels_path = os.path.join(artifact_dir, "label_mappings.json")
        with open(labels_path, "w") as f:
            json.dump({
                "category_mapping": CATEGORY_MAPPING,
                "category_to_id": CATEGORY_TO_ID,
                "id_to_category": ID_TO_CATEGORY,
                "subtype_to_id": SUBTYPE_TO_ID,
                "id_to_subtype": ID_TO_SUBTYPE
            }, f, indent=2)

        # 4. Preprocessing config & fit metrics
        config_path = os.path.join(artifact_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "config": asdict(self.config),
                "fit_metrics": self.fit_metrics
            }, f, indent=2)

        logger.info(f"Saved all preprocessing artifacts to {artifact_dir}")

    @classmethod
    def load(cls, artifact_dir: str) -> "Preprocessor":
        """Loads a fitted Preprocessor from saved artifacts for inference."""
        config_path = os.path.join(artifact_dir, "config.json")
        scaler_path = os.path.join(artifact_dir, "scaler.pkl")
        schema_path = os.path.join(artifact_dir, "schema.json")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler at {scaler_path}")

        with open(config_path, "r") as f:
            data = json.load(f)
            config = PreprocessingConfig(**data["config"])

        instance = cls(config)
        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)

        with open(schema_path, "r") as f:
            schema_data = json.load(f)
            instance.constant_features = schema_data.get("constant_features", [])
            instance.feature_stats = schema_data.get("feature_statistics", {})

        instance.fitted = True
        logger.info(f"Loaded Preprocessor from {artifact_dir}")
        return instance


def run_pipeline(
    config: PreprocessingConfig,
    is_sample: bool = True
) -> Dict[str, Any]:
    """Runs the complete preprocessing workflow: discover -> fit train -> transform train/test -> save artifacts."""
    train_files = discover_csv_files(config.train_dir)
    test_files = discover_csv_files(config.test_dir)

    logger.info(f"Discovered {len(train_files)} train CSVs and {len(test_files)} test CSVs.")
    if is_sample:
        logger.info(
            f"RUNNING IN SAMPLE MODE: Processing max {config.sample_rows_per_file} rows per file."
        )
    else:
        logger.info("RUNNING IN FULL DATASET MODE: Processing all records across all files.")

    preprocessor = Preprocessor(config)
    preprocessor.fit(train_files)

    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "_sample" if is_sample else ""
    train_out = os.path.join(config.output_dir, f"train{suffix}.parquet")
    test_out = os.path.join(config.output_dir, f"test{suffix}.parquet")

    train_stats = preprocessor.process_and_save_parquet(train_files, train_out, split_name="train")
    test_stats = preprocessor.process_and_save_parquet(test_files, test_out, split_name="test")

    preprocessor.save(config.artifact_dir)

    return {
        "is_sample": is_sample,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "artifact_dir": config.artifact_dir
    }


def inspect_samples(train_dir: str, num_files: int = 5, rows_per_file: int = 3) -> None:
    """Prints schema and label inferences for a small set of diverse attack files."""
    files = discover_csv_files(train_dir)
    print(f"\n{'='*70}\nCICIoMT2024 SAMPLE INSPECTION ({len(files)} total training files discovered)\n{'='*70}")
    sample_files = files[:num_files]
    
    for idx, f in enumerate(sample_files):
        meta = parse_filename_label(f)
        df = pd.read_csv(f, nrows=rows_per_file)
        print(f"\n[{idx+1}/{len(sample_files)}] File: {os.path.basename(f)}")
        print(f"  -> Inferred Subtype : {meta['subtype']} (ID: {meta['subtype_id']})")
        print(f"  -> Inferred Category: {meta['category']} (ID: {meta['category_id']})")
        print(f"  -> Is Attack Flag   : {meta['is_attack']}")
        print(f"  -> Shape Sampled    : {df.shape}")
        validate_schema(df, f)
        print(f"  -> Schema Validation: 45 features match expected schema EXACTLY.")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="GAT-XIDS Preprocessing Pipeline")
    parser.add_argument("--inspect", action="store_true", help="Inspect sample rows from training files")
    parser.add_argument("--sample", action="store_true", help="Run preprocessing on small sample per file (default)")
    parser.add_argument("--full", action="store_true", help="Run full preprocessing on all 8.77M records (explicit)")
    parser.add_argument("--sample-rows-per-file", type=int, default=1000, help="Rows per file in sample mode (default: 1000)")
    parser.add_argument("--chunk-size", type=int, default=10000, help="CSV read chunk size (default: 10000)")
    parser.add_argument("--train-dir", type=str, default="WIFI & MQTT/attacks/CSV/train")
    parser.add_argument("--test-dir", type=str, default="WIFI & MQTT/attacks/CSV/test")
    parser.add_argument("--output-dir", type=str, default="results/preprocessed")
    parser.add_argument("--artifact-dir", type=str, default="models/preprocessing")

    args = parser.parse_args()

    if args.inspect:
        inspect_samples(args.train_dir)
        return

    is_sample = not args.full  # Default is sample mode unless --full is explicitly specified
    sample_rows = args.sample_rows_per_file if is_sample else None

    config = PreprocessingConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        chunk_size=args.chunk_size,
        sample_rows_per_file=sample_rows
    )

    results = run_pipeline(config, is_sample=is_sample)
    print("\n--- Preprocessing Pipeline Summary ---")
    print(f"Mode: {'SAMPLE (1000 rows/file)' if is_sample else 'FULL (8.77M records)'}")
    print(f"Train Parquet: {results['train_stats']['output_file']} ({results['train_stats']['total_rows']:,} rows)")
    print(f"Test Parquet : {results['test_stats']['output_file']} ({results['test_stats']['total_rows']:,} rows)")
    print(f"Saved Artifacts: {results['artifact_dir']}")


if __name__ == "__main__":
    main()
