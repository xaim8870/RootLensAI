# RootLensAI

### Graph Neural Network Root Cause Analysis for Observable Microservices

[![Python](https://img.shields.io/badge/Python-3.10%2B-111111?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GraphSAGE-111111?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Instrumented-111111?logo=opentelemetry&logoColor=white)](https://opentelemetry.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Live%20Telemetry-111111?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-111111?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Operator%20Console-111111?logo=streamlit&logoColor=white)](https://streamlit.io/)

**RootLensAI** is an end-to-end intelligent root cause analysis platform for distributed microservice systems. It combines live OpenTelemetry and Prometheus telemetry, controlled fault experiments, leakage-safe datasets, a node-preserving GraphSAGE model, and grounded hybrid RAG explanations in one reproducible research workflow.

The system answers a practical SRE question:

> When a distributed application degrades, which microservice is the most likely root cause—and what telemetry evidence supports that diagnosis?

RootLensAI is designed for research, technical demonstrations, final-year projects, AIOps experimentation, and reproducible microservice observability studies.

---

## Why RootLensAI?

Traditional monitoring tools show metrics, traces, and dashboards. RootLensAI turns those signals into a system-level diagnosis.

- **Real distributed telemetry** from the OpenTelemetry Demo testbed
- **Controlled and reversible fault injection** for research-grade experiments
- **Graph-aware RCA** using the actual microservice dependency topology
- **Leakage-safe model development** with complete experiments grouped by `run_id`
- **Live inference** over the same telemetry semantics used during training
- **Hybrid RAG explainability** using numeric incident similarity and semantic context
- **Operator-controlled recovery loop** with telemetry stabilization and stale-result protection
- **Reproducible provenance** through manifests, checksums, frozen splits, MLflow, and model-bundle metadata

The GraphSAGE model remains the root cause decision engine. The LLM explanation layer supports the prediction with retrieved evidence; it does not replace or override the model.

---

## End-to-end architecture

![RootLensAI end-to-end intelligent root cause analysis workflow](docs/rootlens_end_to_end_workflow.svg)

The editable architecture source and repository evidence map are available in:

- [Mermaid workflow source](docs/rootlens_end_to_end_workflow.mmd)
- [Architecture evidence notes](docs/rootlens_workflow_notes.md)

### System lifecycle

```text
OpenTelemetry Demo microservices
        ↓
OpenTelemetry Collector + Prometheus
        ↓
Controlled healthy / fault / recovery experiments
        ↓
Validated and frozen RootLens datasets
        ↓
Run-grouped tabular and graph datasets
        ↓
Classical baselines + GraphSAGE training tracked in MLflow
        ↓
Frozen GraphSAGE RCA inference bundle
        ↓
Live telemetry → root cause prediction → confidence ranking
        ↓
Numeric + semantic retrieval → grounded AI explanation
        ↓
Streamlit operator console → restoration → observed recovery
```

---

## Core capabilities

### Controlled microservice fault experiments

The experiment framework uses YAML configurations and a safety-first orchestration layer:

- Prometheus and current-telemetry prechecks
- Exact service environment verification
- Target-service-only recreation with `--no-deps --force-recreate`
- Paired fault and recovery collection
- Immediate structural and fault-specific validation
- Deterministic run IDs and immutable run directories
- Quarantine instead of deletion or silent replacement
- Verified restoration when a fault may still be active

Dataset v2 contains accepted examples from these controlled families:

| Root service | Fault family |
|---|---|
| Payment | Latency +500 ms, error 50% |
| Cart | Latency +500 ms |
| Checkout | Latency +500 ms, error 50% |
| Product Catalog | Latency +500 ms, error 50% |

Rejected, unobservable, smoke, and quarantined experiments are preserved for provenance but excluded from accepted research data.

### Frozen telemetry representation

RootLens monitors 12 services in a fixed order:

```text
frontend · frontend-proxy · checkout · payment · cart · currency
shipping · product-catalog · recommendation · email · ad · quote
```

Every service contributes seven telemetry features:

```text
cpu · memory · request_rate · has_requests
latency_ms · error_rps · error_rate
```

This produces two equivalent representations of each system telemetry window:

- **Classical ML:** `12 services × 7 features = 84 features`
- **Graph learning:** `x = [12 nodes, 7 features]` with a frozen service topology

### Leakage-safe research design

RootLensAI never randomly distributes windows from the same experiment across model splits.

```text
Processed Dataset
      ↓
Group complete samples by run_id
      ↓
Train / Validation / Sealed Test
```

Dataset v2 uses seed `42` and has zero run overlap:

| Split | Runs | System windows |
|---|---:|---:|
| Train | 24 | 1,440 |
| Validation | 8 | 480 |
| Sealed test | 8 | 480 |

Recovery runs remain part of the accepted research dataset but are excluded from supervised RCA classification.

### GraphSAGE root cause model

The canonical model is **GraphSAGE RCA v2 node-preserving**:

```text
Per-service telemetry
        ↓
Two residual GraphSAGE message-passing layers
        ↓
Ordered 12-node embedding concatenation
        ↓
MLP classification head
        ↓
Root-cause probabilities
```

Verified model properties:

| Property | Value |
|---|---|
| Parameters | 61,445 |
| Nodes | 12 |
| Features per node | 7 |
| Message-passing edges | 44 |
| Classes | healthy, payment, cart, checkout, product_catalog |
| MLflow run | `c5b8c95fd91d4ce2ab27df44e846ddec` |
| Default device | CPU |

The frozen bundle records the checkpoint, class mapping, node order, feature order, topology checksum, and training-derived normalization statistics.

### Validation-only model reliability

The repository includes confidence diagnostics computed on the canonical Dataset v2 validation population:

| Metric | Validation result |
|---|---:|
| Accuracy | 99.38% |
| Mean model confidence | 97.85% |
| Expected Calibration Error | 2.36% |
| Accuracy–confidence gap | 1.52% |
| Negative log likelihood | 0.0341 |
| Brier score | 0.0114 |

These are **validation metrics for raw softmax probabilities**, not live confidence guarantees and not sealed-test performance. The canonical bundle explicitly records `sealed_test_evaluated: false`.

### Hybrid RAG explainability

RootLensAI explains an RCA result through two local retrieval channels:

1. **Numeric retrieval** — training-normalized cosine nearest neighbors over 84-feature historical incident windows, diversified by `run_id`.
2. **Semantic retrieval** — `sentence-transformers/all-MiniLM-L6-v2` embeddings over validated incident summaries, service topology, and protocol context.

Retrieved evidence is assembled into traceable evidence IDs and sent to a configurable Hugging Face Inference Provider for structured explanation generation.

The RAG layer returns:

- Incident summary
- Evidence-backed claims
- Explicit uncertainty
- Suggested investigation checks
- Similar validated incidents
- Relevant topology and protocol context

Only canonical training and validation incidents are indexed. **Sealed-test runs indexed: 0.**

---

## RootLensAI operator console

The active application is the modular Streamlit interface at:

```text
rootlens/app/app.py
```

It provides:

- Live and demo telemetry modes
- System health and model-indicated root cause
- Five-class probability ranking
- Current-window confidence and offline reliability context
- Twelve-service status grid and dependency topology
- Live telemetry table
- Allow-listed fault injection and restoration controls
- Telemetry stabilization and recovery countdowns
- Stale RCA and stale RAG invalidation
- Grounded AI-assisted incident investigation
- Transparent retrieved-evidence inspection

Fault-selection metadata is never passed into GraphSAGE. The model receives only current telemetry and the frozen service graph.

---

## Dataset v2 at a glance

| Dataset property | Count |
|---|---:|
| Healthy runs | 5 |
| Accepted fault/recovery pairs | 35 |
| Accepted official runs | 75 |
| Raw service-window rows | 54,000 |
| Supervised classifier runs | 40 |
| System-window classifier samples | 2,400 |
| Flattened telemetry features | 84 |
| Graph nodes | 12 |
| Node features | 7 |

Dataset manifests, processed metadata, split definitions, graph metadata, and checksums are retained under `rootlens/data/`.

---

## Repository structure

```text
RootLensAI/
├── docs/                         Architecture diagram and evidence notes
├── experiments/                  Safe experiment runner, validators and YAML configs
├── opentelemetry-demo/           Nested OpenTelemetry Demo testbed
├── rootlens/
│   ├── app/                      Modular Streamlit operator console
│   ├── collector/                Prometheus telemetry collection
│   ├── config/                   Frozen service dependency graph
│   ├── data/
│   │   ├── raw/                  Immutable experimental runs
│   │   ├── manifests/            Dataset acceptance and provenance boundaries
│   │   ├── processed/            Model-ready tabular and graph datasets
│   │   └── reports/              Evaluation, calibration and comparison artifacts
│   ├── demo/                     Allow-listed fault controller
│   ├── inference/                Live telemetry adapter and GraphSAGE inference
│   ├── models/                   Frozen model bundle metadata
│   ├── rag/                      Hybrid retrieval and grounded explanation
│   └── scripts/                  Dataset, split, graph, training and audit tools
├── mlruns/                       Local MLflow artifacts
├── .env.example                 Optional Hugging Face environment template
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.10 or newer
- Docker Desktop with Docker Compose
- A running RootLens-compatible OpenTelemetry Demo stack
- Optional: Hugging Face token for hosted AI explanations

### 1. Clone the project

```bash
git clone https://github.com/xaim8870/RootLensAI.git
cd RootLensAI
```

The project expects its pinned OpenTelemetry Demo checkout at `opentelemetry-demo/`. If your clone does not populate that nested Git repository automatically, create it at the recorded Dataset v2 testbed commit:

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo.git opentelemetry-demo
git -C opentelemetry-demo checkout 775f0fd9ee2f9fd43ad2254670e3a464c981b46e
```

### 2. Create a Python environment

PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r rootlens/requirements-mvp.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r rootlens/requirements-mvp.txt
```

### 3. Start the OpenTelemetry Demo testbed

From `opentelemetry-demo/`, preserve the RootLens Compose configuration:

```powershell
docker compose --env-file .env --env-file .env.override -f compose.yaml -f compose.observability.yaml up -d
```

RootLens experiment and demo controls intentionally recreate only the selected target service. They do not restart the entire stack or recreate the OpenTelemetry Collector during fault activation.

### 4. Launch RootLensAI

From the repository root:

```powershell
streamlit run rootlens/app/app.py
```

Open the local URL displayed by Streamlit, normally `http://localhost:8501`.

### 5. Optional Hugging Face explanations

Copy the environment template without adding secrets to Git:

```powershell
Copy-Item .env.example .env
```

Set:

```text
HF_TOKEN=your_fine_grained_token
ROOTLENS_HF_MODEL=your_supported_model_id
```

Alternatively, set variables for the current PowerShell session:

```powershell
$env:HF_TOKEN="hf_..."
$env:ROOTLENS_HF_MODEL="Qwen/Qwen3-4B-Thinking-2507"
```

Never commit `.env` or a real Hugging Face token.

---

## Using the live demonstration safely

1. Select **Restore System** and allow the recovery gate to finish.
2. Select **Refresh RCA** to capture a fresh telemetry window.
3. Choose one allow-listed service/fault combination.
4. Select **Inject Fault**.
5. Wait for the 30-second telemetry stabilization gate.
6. Select **Refresh RCA** again.
7. Review the GraphSAGE prediction, probabilities, topology, and telemetry.
8. Optionally select **Generate AI Incident Analysis**.
9. Select **Restore System**, wait for recovery, then explicitly refresh RCA.

Previous diagnoses are hidden after a testbed change because they describe stale telemetry. Switching demo mode can also trigger a Streamlit rerun, but it is not the intended mechanism for clearing a live fault; restoration, stabilization, and a fresh RCA are the correct workflow.

---

## Experiment framework

Preview an experiment without touching Docker or Prometheus:

```powershell
python experiments/run_experiment.py `
  --config experiments/configs/payment_latency_500.yaml `
  --repetitions 1 `
  --dry-run
```

Read-only validation example:

```powershell
python experiments/validate_run.py `
  --config experiments/configs/payment_latency_500.yaml `
  --run-dir rootlens/data/raw/payment_latency_500_001 `
  --condition fault
```

Real experiments change running service configuration and should only be executed with explicit operator approval. See [experiments/README.md](experiments/README.md) for the safety contract and orchestration details.

---

## Research integrity and reproducibility

RootLensAI treats provenance as part of the system—not as an afterthought.

- Raw runs are not overwritten automatically.
- Failed and quarantined artifacts remain identifiable.
- Dataset manifests explicitly include accepted runs and exclude invalid ones.
- Complete `run_id` groups remain in exactly one model split.
- Feature normalization is fitted on training data only.
- Live inference never fits preprocessing.
- The graph topology, node order, feature order, and class mapping are frozen.
- RAG retrieval excludes the sealed test split.
- Hosted LLM output cannot alter GraphSAGE probabilities or predictions.
- Fault metadata remains separate from model input.
- The canonical model bundle records that the sealed test was not evaluated.

---

## Technology stack

| Layer | Technology |
|---|---|
| Microservice testbed | OpenTelemetry Demo, Docker Compose |
| Instrumentation | OpenTelemetry |
| Telemetry backend | Prometheus |
| Data processing | Python, pandas, NumPy |
| Classical ML | scikit-learn |
| Graph ML | PyTorch, PyTorch Geometric, GraphSAGE |
| Experiment tracking | MLflow |
| Numeric retrieval | StandardScaler, cosine NearestNeighbors |
| Semantic retrieval | Sentence Transformers, MiniLM |
| Explanation generation | Hugging Face Inference Providers |
| Operator interface | Streamlit |

---

## Current scope

Implemented:

- Controlled microservice latency and error experiments
- Dataset validation and frozen Dataset v1/v2 provenance
- Classical RCA baselines
- Node-preserving GraphSAGE root cause classification
- Live Prometheus-to-graph inference
- Validation-only confidence diagnostics
- Hybrid RAG evidence retrieval and explanation
- Streamlit observability and fault-demonstration console
- Operator-controlled fault restoration and observed recovery

Not currently claimed as implemented:

- Autonomous remediation
- Learned intervention-effect prediction
- Production Kubernetes deployment
- Alerting or ticketing integrations
- Sealed-test performance claims for the canonical bundle

---

## Documentation

- [Complete architecture workflow](docs/rootlens_end_to_end_workflow.svg)
- [Architecture evidence notes](docs/rootlens_workflow_notes.md)
- [Experiment orchestration guide](experiments/README.md)
- [Dataset v2 manifest](rootlens/data/manifests/dataset_v2.yaml)
- [Frozen service graph](rootlens/config/service_graph_v1.yaml)
- [Canonical model bundle](rootlens/models/graphsage_rca_v2/model_bundle.json)
- [RAG provenance](rootlens/rag/rag_provenance.json)

---

## Responsible use

RootLensAI is a research and demonstration system. A model-indicated root cause is an evidence-driven hypothesis, not a substitute for production incident procedures, service-owner judgment, or direct trace and log inspection. Historical incidents retrieved by RAG are supporting analogies, not proof of the current diagnosis.

---

## Contributing

Contributions should preserve the project’s reproducibility boundaries:

1. Do not rewrite accepted raw datasets.
2. Do not mix windows from one run across train, validation, and test.
3. Keep telemetry definitions and model-input ordering explicit.
4. Add new fault families through reversible smoke-tested configurations.
5. Record provenance for datasets, splits, models, and retrieval indexes.
6. Keep fault-control metadata isolated from RCA inference.

Open an issue or pull request with a clear description of the research or engineering change and its validation evidence.

---

<p align="center">
  <strong>RootLensAI</strong><br>
  Observable microservices. Graph-aware diagnosis. Evidence-grounded explanations.
</p>
