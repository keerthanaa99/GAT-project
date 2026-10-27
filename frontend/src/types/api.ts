export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_ready: boolean;
  checkpoint_path: string;
  preprocessing_ready: boolean;
  dependencies_ready: boolean;
  details: Record<string, any>;
}

export interface ModelInfoResponse {
  model_name: string;
  architecture: string;
  expected_feature_count: number;
  feature_names: string[];
  supported_classes: string[];
  class_id_mapping: Record<string, number>;
  checkpoint_status: string;
  hyperparameters: Record<string, any>;
  methodology_note: string;
}

export interface FlowPredictionItem {
  flow_index: number;
  predicted_category: string;
  predicted_class_id: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface GraphSummary {
  num_nodes: number;
  num_edges: number;
  density: number;
  k_neighbors: number;
  metric: string;
}

export interface AnalyzeCSVResponse {
  status: string;
  filename: string;
  total_records: number;
  category_counts: Record<string, number>;
  predictions: FlowPredictionItem[];
  graph_summary: GraphSummary;
  disclaimer: string;
}

export interface ApiError {
  detail: string;
  missing_features?: string[];
  extra_features?: string[];
}
