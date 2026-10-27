"""
Pydantic Request & Response Schemas for GAT-XIDS FastAPI
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status: 'ok' or 'degraded'")
    model_ready: bool = Field(..., description="Whether the GAT model checkpoint is loaded and ready")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    preprocessing_ready: bool = Field(..., description="Whether scaler and preprocessing schema are loaded")
    dependencies_ready: bool = Field(..., description="Whether PyTorch and PyG dependencies are verified")
    details: Dict[str, Any] = Field(default_factory=dict, description="Supplementary diagnostic information")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Official model name and purpose")
    architecture: str = Field(..., description="Architecture specification (layers, heads, dimensions)")
    expected_feature_count: int = Field(..., description="Total numerical input features required (45)")
    feature_names: List[str] = Field(..., description="Ordered list of the 45 numerical feature names")
    supported_classes: List[str] = Field(..., description="List of supported attack category names")
    class_id_mapping: Dict[str, int] = Field(..., description="Category name to class ID mapping")
    checkpoint_status: str = Field(..., description="Checkpoint validation status")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    methodology_note: str = Field(..., description="Scientific methodology statement")


class FlowPredictionItem(BaseModel):
    flow_index: int = Field(..., description="Zero-indexed flow record ID")
    predicted_category: str = Field(..., description="Predicted broad attack category")
    predicted_class_id: int = Field(..., description="Numeric class ID corresponding to prediction")
    confidence: float = Field(..., description="Softmax posterior probability (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities across all 5 classes")


class GraphSummary(BaseModel):
    num_nodes: int = Field(..., description="Number of flow nodes in the constructed graph")
    num_edges: int = Field(..., description="Number of similarity edges in the graph")
    density: float = Field(..., description="Graph density ratio")
    k_neighbors: int = Field(..., description="Nearest neighbor parameter k")
    metric: str = Field(..., description="Distance metric utilized (e.g., euclidean)")


class AnalyzeCSVResponse(BaseModel):
    status: str = Field(..., description="Operation status ('success')")
    filename: str = Field(..., description="Name of uploaded file")
    total_records: int = Field(..., description="Number of valid flow records analyzed")
    category_counts: Dict[str, int] = Field(..., description="Summary counts of predictions per category")
    predictions: List[FlowPredictionItem] = Field(..., description="List of individual flow predictions")
    graph_summary: GraphSummary = Field(..., description="Topological summary of flow-similarity graph")
    disclaimer: str = Field(..., description="Forensic and scientific disclaimer")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Detailed description of the validation or processing error")
    missing_features: Optional[List[str]] = Field(default=None, description="Features missing from input")
    extra_features: Optional[List[str]] = Field(default=None, description="Unexpected features detected")
