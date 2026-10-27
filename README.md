# GAT-XIDS

**An Inherently Explainable Graph Attention Network for Intrusion Detection in Medical IoT Networks**

GAT-XIDS leverages Graph Attention Networks (GAT) to model behavioral relationships and statistical similarities between network flows in Internet of Medical Things (IoMT) environments. Built on the **CICIoMT2024** benchmark dataset, it combines multi-head self-attention with full-stack observability via a FastAPI backend and a React + TypeScript dashboard.

---

## Key Features

- **Standardized Feature Preprocessing**: Out-of-core standardization on 45 numerical traffic features, with robust NaN/Inf cleaning and support for 19 attack subtypes mapped to 5 broad categories (`Benign`, `ARP Spoofing`, `MQTT attacks`, `Reconnaissance`, `TCP/IP DoS/DDoS`).
- **Graph Attention Network (GAT)**: 2-layer GAT architecture with 4 multi-head attention mechanisms, preserving attention weights per layer for explainability.
- **Explainable AI (XAI)**: Inherent attention extraction attributing decision weights across neighboring flows without post-hoc black-box approximations.
- **FastAPI REST API**: High-performance backend providing health verification, model introspection, and flow CSV classification with inductive similarity graph generation.
- **React + TypeScript Frontend**: Modern dashboard featuring network flow ingestion, live GAT classification, category distributions, interactive prediction tables, and CSV exports.

---

## Project Structure

```
├── app.py                     # Interactive Streamlit exploratory application
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI application entry point & CORS configuration
│   │   ├── schemas.py         # Pydantic request/response schemas
│   │   └── services/          # ModelService cached loading and inference pipeline
│   └── tests/                 # FastAPI REST API test suite
├── frontend/                  # React 19 + TypeScript + Vite web dashboard
│   ├── src/
│   │   ├── components/        # TrafficAnalysis, Header, Sidebar
│   │   ├── services/          # API fetch client
│   │   └── types/             # TypeScript API interfaces
├── models/
│   ├── gat_best_checkpoint.pt # Trained GAT model weights
│   └── preprocessing/         # Serialized scaler.pkl, schema.json, label mappings
├── results/
│   ├── explanations/          # Serialized node explanation JSONs & text reports
│   ├── graphs/                # Serialized train & test PyG graph artifacts
│   ├── preprocessed/          # Scaled sample Parquet partitions
│   └── training_history.json  # 40-epoch training loss, accuracy & F1 history
├── sample_input.csv           # 50-flow benchmark template for live testing
├── src/
│   ├── dashboard/             # Visualization and dashboard utilities
│   ├── data/                  # Preprocessing, schema validation & label inference
│   ├── explain/               # AttentionExplainer and explanation reporting
│   ├── graph/                 # Sparse k-NN flow-similarity graph builder
│   └── models/                # GATClassifier architecture and training pipeline
└── tests/                     # 46 unit and integration tests
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/keerthanaa99/GAT-project.git
cd GAT-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run core test suite (46 tests)
pytest tests/

# Run backend API tests (7 tests)
pytest backend/tests/test_api.py
```

### 3. Launch Application

**Start the FastAPI Backend:**
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Start the React Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in your browser.
