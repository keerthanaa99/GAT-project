"""
Model Service for GAT-XIDS
==========================
Handles cached model loading, artifact validation, CSV ingestion, flow-similarity
graph construction, and inference execution.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import HTTPException, UploadFile, status

# Resolve reliable project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from src.data.preprocess import (
    EXPECTED_FEATURES,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    Preprocessor,
    clean_chunk,
    validate_schema
)
from src.graph.graph_builder import (
    GraphBuildConfig,
    build_flow_similarity_graph
)
from src.models.gat_model import GATClassifier, GATConfig
from backend.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    AnalyzeCSVResponse,
    FlowPredictionItem,
    GraphSummary
)

logger = logging.getLogger("gat_xids.backend.model_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
MAX_RECORDS_DEFAULT = 2000         # Upper limit per uploaded batch to preserve RAM


class ModelService:
    """
    Cached service orchestrating preprocessing, graph building, and GAT inference.
    """
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.checkpoint_path = self.project_root / "models" / "gat_best_checkpoint.pt"
        self.preprocessing_dir = self.project_root / "models" / "preprocessing"
        
        self.model: Optional[GATClassifier] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.schema_data: Dict[str, Any] = {}
        self.label_mappings: Dict[str, Any] = {}
        self.initialized = False

    def initialize(self) -> None:
        """Loads and caches model checkpoint and preprocessing scaler."""
        if self.initialized:
            return

        logger.info(f"Initializing ModelService from project root: {self.project_root}")
        
        # 1. Load Preprocessor
        if not self.preprocessing_dir.exists():
            raise FileNotFoundError(f"Preprocessing artifacts directory missing at: {self.preprocessing_dir}")
        self.preprocessor = Preprocessor.load(str(self.preprocessing_dir))

        # 2. Load schema and label mappings
        schema_file = self.preprocessing_dir / "schema.json"
        labels_file = self.preprocessing_dir / "label_mappings.json"

        if schema_file.exists():
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schema_data = json.load(f)
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                self.label_mappings = json.load(f)

        # 3. Load GAT model checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GAT model checkpoint missing at: {self.checkpoint_path}")
        self.model = GATClassifier.from_checkpoint(str(self.checkpoint_path))
        self.model.eval()

        self.initialized = True
        logger.info("ModelService initialized successfully with cached GAT model and scaler.")

    def get_health_status(self) -> HealthResponse:
        """Evaluates health of model checkpoint and dependencies."""
        model_ready = False
        preprocessing_ready = False
        dependencies_ready = False
        details: Dict[str, Any] = {}

        try:
            import torch
            import torch_geometric
            import sklearn
            dependencies_ready = True
            details["torch_version"] = torch.__version__
            details["pyg_version"] = torch_geometric.__version__
        except Exception as e:
            details["dependencies_error"] = str(e)

        try:
            if not self.initialized:
                self.initialize()
            model_ready = (self.model is not None)
            preprocessing_ready = (self.preprocessor is not None and self.preprocessor.fitted)
        except Exception as e:
            details["init_error"] = str(e)

        overall_status = "ok" if (model_ready and preprocessing_ready and dependencies_ready) else "degraded"

        return HealthResponse(
            status=overall_status,
            model_ready=model_ready,
            checkpoint_path=str(self.checkpoint_path),
            preprocessing_ready=preprocessing_ready,
            dependencies_ready=dependencies_ready,
            details=details
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns metadata, hyperparameter configuration, and feature schema."""
        if not self.initialized:
            self.initialize()

        assert self.model is not None
        config_dict = self.model.config.__dict__
        supported_classes = list(CATEGORY_TO_ID.keys())

        return ModelInfoResponse(
            model_name="GAT-XIDS (Inherently Explainable Graph Attention Network)",
            architecture=f"{self.model.config.num_layers}-Layer GAT with {self.model.config.heads} Multi-Head Attention",
            expected_feature_count=len(EXPECTED_FEATURES),
            feature_names=EXPECTED_FEATURES,
            supported_classes=supported_classes,
            class_id_mapping=CATEGORY_TO_ID,
            checkpoint_status="Verified and Loaded" if self.model is not None else "Missing",
            hyperparameters=config_dict,
            methodology_note=(
                "Graph topology represents statistical behavioral similarity in standardized feature space "
                "via sparse k-NN (k=5). It does not represent physical device communication endpoints or hardware wiring."
            )
        )

    async def analyze_csv(self, file: UploadFile, max_records: int = MAX_RECORDS_DEFAULT) -> AnalyzeCSVResponse:
        """
        Validates, preprocesses, graphs, and classifies flows from an uploaded CSV.
        """
        if not self.initialized:
            self.initialize()

        # 1. Validate file extension
        filename = file.filename or "uploaded_traffic.csv"
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for '{filename}'. Only CSV files (.csv) are accepted."
            )

        # 2. Read contents safely
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(contents) / (1024*1024):.2f} MB) exceeds maximum allowed size (25 MB)."
            )

        # 3. Parse CSV into DataFrame
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse CSV file: {str(e)}"
            )

        if df.empty or len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV contains no records."
            )

        # Cap records to avoid laptop memory overload
        if len(df) > max_records:
            logger.info(f"Limiting uploaded dataset from {len(df)} to {max_records} records.")
            df = df.iloc[:max_records].copy()

        # 4. Validate schema: Check for missing 45 numerical features
        cols = list(df.columns)
        missing_features = [f for f in EXPECTED_FEATURES if f not in cols]
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"CSV schema mismatch. Missing {len(missing_features)} of 45 required features: {missing_features[:5]}... "
                    f"Ensure input matches the CICIoMT2024 standardized traffic feature schema."
                )
            )

        # 5. Extract ONLY the 45 numerical features, excluding any labels, filenames, or metadata
        features_df = df[EXPECTED_FEATURES].copy()

        # Clean non-numeric, infs, and nans according to standard preprocessing policy
        for col in EXPECTED_FEATURES:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        cleaned_feats, inf_cnt, nan_cnt = clean_chunk(features_df)
        if inf_cnt > 0 or nan_cnt > 0:
            logger.info(f"Handled {inf_cnt} Infs and {nan_cnt} NaNs in uploaded CSV according to baseline policy.")

        # 6. Apply existing training scaler (never refit on uploaded data!)
        assert self.preprocessor is not None
        scaled_features = self.preprocessor.scaler.transform(cleaned_feats.values).astype(np.float32)

        # 7. Construct flow-similarity graph using existing k-NN methodology
        config = GraphBuildConfig(k_neighbors=5, make_undirected=True, remove_self_loops=True)
        graph_data = build_flow_similarity_graph(scaled_features, labels=None, config=config)

        # 8. GAT Inference in evaluation mode without gradients
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            preds_dict = self.model.predict(graph_data.x, graph_data.edge_index)

        probs_tensor = preds_dict["probabilities"].cpu().numpy()
        preds_tensor = preds_dict["predicted_classes"].cpu().numpy()

        # 9. Format structured predictions
        predictions: List[FlowPredictionItem] = []
        category_counts: Dict[str, int] = {cat: 0 for cat in CATEGORY_TO_ID.keys()}

        for idx in range(len(df)):
            class_id = int(preds_tensor[idx])
            cat_name = ID_TO_CATEGORY.get(class_id, "Unknown")
            conf = float(probs_tensor[idx, class_id])

            prob_map = {
                ID_TO_CATEGORY.get(c_id, str(c_id)): float(probs_tensor[idx, c_id])
                for c_id in range(len(CATEGORY_TO_ID))
            }

            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            predictions.append(
                FlowPredictionItem(
                    flow_index=idx,
                    predicted_category=cat_name,
                    predicted_class_id=class_id,
                    confidence=round(conf, 4),
                    probabilities=prob_map
                )
            )

        num_nodes = int(graph_data.num_nodes)
        num_edges = int(graph_data.edge_index.shape[1])
        density = float(num_edges / max(1, num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0

        graph_summary = GraphSummary(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=round(density, 4),
            k_neighbors=config.k_neighbors,
            metric=config.metric
        )

        return AnalyzeCSVResponse(
            status="success",
            filename=filename,
            total_records=len(predictions),
            category_counts=category_counts,
            predictions=predictions,
            graph_summary=graph_summary,
            disclaimer=(
                "Predictions are generated by the GAT-XIDS neural network over a statistical flow-similarity graph. "
                "Graph topology reflects high-dimensional behavioral feature clustering, not verified physical device wiring."
            )
        )


# Global singleton instance
model_service = ModelService()
