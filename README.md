# TALUS — Physics-Informed AI for Open-Pit Mine Slope Risk Intelligence

> **Phase II + Phase III implementation** for an end-to-end Machine Learning / Data Science project.

TALUS is a decision-support prototype that combines **geotechnical telemetry, preprocessing, physics-based Factor of Safety, supervised ML, local explainability, temporal history and an API-first control-room UI**.

It is deliberately designed so the dashboard does not contain a hidden copy of the prediction logic. The architecture is:

**Processed data → Physics Engine + ML → XAI → SQLite → FastAPI → Streamlit Control Room**

## What is implemented

### Phase II
- **User Interface + Backend Connectivity:** Streamlit communicates with FastAPI over HTTP. The UI does not silently run the model locally when the API is down.
- **Preprocessing + EDA:** 1,200 slope telemetry records, 480 rock/soil records and 320 image-derived feature records are available in processed artifacts. The Data Intelligence page exposes distributions, correlations, material statistics and image-texture analysis.
- **Model Building:** Logistic Regression, SVM, Random Forest and Gradient Boosting are evaluated. The persisted metadata records the selected model and selection metric.
- **Result Analysis + Business Insights:** physics category, ML probability, local contributors, engineering recommendations, risk distribution and temporal trajectories.

### Phase III
- Dockerized backend and frontend
- Docker Compose for local demonstration
- Render blueprint for cloud deployment
- `/health`, `/docs`, `/model-info`, `/data-summary`, `/predict` and telemetry history endpoints
- Automated API/physics/data-driven tests
- GitHub Actions CI
- Demo runbook, viva questions, report and presentation

## Data-driven sector registry

The original prototype contained a manually seeded six-sector database even though the processed telemetry contains more sectors. The Phase III version derives the sector registry from `data/processed/cleaned_features.csv`. Therefore the dashboard currently reflects the dataset's **10 sectors (S1–S10)** instead of an artificial S1–S6 limit.

## Model evidence

The extended benchmark selects the serving classifier by **mean stratified 5-fold macro-F1**, not by accuracy alone. The current selected model is Support Vector Machine. The selected-model held-out split and all four candidate CV results are persisted in `models/model_metadata.json`, `results/model_comparison.csv`, and `results/ml_rigor_summary.json`.

| Selected-model held-out metric | Value |
|---|---:|
| Selected model | Support Vector Machine |
| Selection rule | 5-fold stratified CV macro-F1 |
| Accuracy | 0.8417 |
| Macro precision | 0.7643 |
| Macro recall | 0.8403 |
| Macro-F1 | 0.7940 |
| Weighted-F1 | 0.8447 |
| ROC-AUC (macro OVR) | 0.9668 |
| Training records | 960 |
| Test records | 240 |

These are offline metrics on the supplied **synthetic** dataset, not evidence of operational mine-site performance. See `results/ml_rigor_summary.json` for CV mean ± standard deviation, the factor-of-safety ablation, sector hold-out, per-class metrics and calibration score.

## Phase A — data provenance and label circularity

The repository's monitoring telemetry, “raw field” export, and rock/soil samples
are generated synthetic data; the raw-field files are synthetic records with
injected data-quality faults, not mine observations. Run
`python scripts/audit_synthetic_labels.py` to regenerate the label-construction
audit in `results/synthetic_label_audit.json` and its row-level companion CSV.
For the checked-in 1,200-record synthetic dataset, the generating rule without
its unrecorded random noise agrees with 90.4167% of labels (macro-F1 0.8557), and
all five direct score inputs are model features. This is a circularity
diagnostic, not validation of the classifier.

No real mine FoS/risk-labelled external dataset has been evaluated, so no
real-data accuracy or sim-to-real gap is reported. The NASA Global Landslide
Catalog is documented as a candidate *event inventory* only; it does not contain
the mine FoS/risk labels required for this model. Its historical export returned
HTTP 404 in the Phase A environment and its redistribution licence was not
verified. The guarded optional loader is `scripts/download_nasa_glc.py`; it does
not bundle or fabricate records. Dataset provenance, known biases, access status,
and exact audit interpretation are in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
The dashboard's “Synthetic vs Real / Leakage Audit” module presents the same
generated findings and explicitly reports that real external validation is
unavailable.

## Local run

### 1. Install
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start backend
```powershell
python run_backend.py
```
Open `http://127.0.0.1:8000/docs` for Swagger.

### 3. Start frontend in another terminal
```powershell
python run_frontend.py
```
Open `http://127.0.0.1:8501`.

### 4. Test
```powershell
pytest -q
```

## Docker

```powershell
docker compose up --build
```
- Frontend: `http://localhost:8501`
- API: `http://localhost:8000/docs`

## Cloud deployment

The repository includes `render.yaml`, `Dockerfile.backend` and `Dockerfile.frontend`. Deploy the blueprint, then verify:

1. backend `/health` returns `healthy`
2. frontend environment variable `TALUS_API_URL` points to the backend service
3. frontend can load the Mine Overview without local fallback

## Project structure

```text
TALUS/
├── backend/                 # FastAPI + physics + ML inference + SQLite
├── frontend/                # Streamlit control room
├── data/raw/                # source datasets
├── data/processed/          # preprocessing outputs
├── models/                  # persisted model/preprocessor/metadata
├── notebooks/               # preprocessing/EDA + model training
├── results/                 # model comparison and figures
├── scripts/                 # reproducible data pipeline
├── tests/                   # automated tests
├── docs/                    # demo/viva/GitHub material
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── render.yaml
```

## Viva-ready architecture explanation

**Why FastAPI?** It separates the inference service from the UI and creates a reusable API for future web/mobile/voice clients.

**Why physics-informed ML?** Factor of Safety provides an engineering stability signal while ML learns patterns from multiple telemetry variables.

**Why XAI?** A risk label alone is not enough for a decision-support system. Local contributors help the user understand which inputs influenced the prediction.

**Why SQLite?** It provides lightweight persistence for the academic/demo environment and makes temporal risk analysis reproducible. A production system could replace it with PostgreSQL/TimescaleDB.

## Important limitation

TALUS is **not an operational mine-safety authority or autonomous control system**. The supplied project data is synthetic/demo-oriented. Before real deployment, the system needs real mine-site data, external validation, uncertainty estimation, sensor QA/QC, temporal/sector holdout evaluation, drift monitoring, access control and certified geotechnical review.

See:
- `docs/DEMO_RUNBOOK.md`
- `docs/PHASE_RUBRIC.md`
- `docs/GITHUB_CHECKLIST.md`

---

## Extended control-room build (October 2026)

The project now includes the following additional offline-first prototype features:

- **Executive Summary** and **3D Mine Model** pages, with procedural terraced terrain, risk-colored sector markers, a schematic slope section, a PCA feature-space view and displacement playback.
- **Materials Gallery** with local rock/soil images and an upload classifier trained from `data/processed/image_features.csv`.
- **ML Rigor** artifacts generated by `scripts/train_ml_rigor.py`: stratified 5-fold CV with macro-F1 mean/std, four-model comparison, FoS ablation, 8-sector/2-sector hold-out, classwise precision/recall, reliability curves and multiclass Brier score.
- **SHAP global bar, beeswarm and local waterfall** artifacts from `scripts/generate_shap_explanations.py`. This analysis uses a separate research Random Forest so it does not misrepresent the selected serving SVM's explanations. The serving API also provides local perturbation contributors and permutation importance is saved as a fallback.
- **TALUS Voice** with a deterministic local intent parser endpoint and browser speech API where supported.
- **Alert Console** with SQLite persistence, Level 0–3 prototype alerts, acknowledgement/history, duplicate suppression, browser tone demo and a clearly labeled S4 severe synthetic demo.
- **Virtual Scenarios** with six conceptual animations/checklists and **Advanced Analytics** with 5,000-sample illustrative Monte Carlo, inverse-velocity indicator and two physics response surfaces.
- Per-sector PDF report endpoint, uploaded-image classification endpoint, system health endpoint and API smoke tests.

### New API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /assistant/query` | Rule-based assistant intents and text responses |
| `GET /alerts/active`, `GET /alerts/history` | Alert lists |
| `POST /alerts/test`, `POST /alerts/{id}/ack`, `POST /alerts/{id}/resolve` | Demo alert creation and acknowledgement/resolution |
| `POST /demo/critical` | Persist a clearly labeled severe synthetic S4 scenario and create a Level 3 demo alert |
| `POST /simulate/pof` | 5,000-sample illustrative probability-of-failure simulation |
| `POST /simulate/sensitivity` | Tornado parameter swing sensitivity analysis (±20% variation) |
| `POST /simulate/recommendations` | Ranked geotechnical remediation actions with estimated FoS gains |
| `POST /physics/response-surfaces` | FoS response grids for angle × pore pressure and cohesion × friction |
| `GET /sector/{id}/ttf` | Indicative inverse-velocity extrapolation when data conditions allow |
| `GET /workers/status` | Real-time worker geofence tracking and danger zone proximity alerts |
| `GET /model/drift` | Population Stability Index (PSI) and Kolmogorov-Smirnov drift monitoring |
| `POST /image/classify` | Upload a photo and classify material from texture/color features |
| `GET /scenario/{type}` | Scenario mechanism and conservative checklist |
| `GET /report/{sector_id}.pdf` | Download sector report |
| `GET /system/health-detail` | Runtime and artifact summary |

### Reproduce ML-rigor artifacts

```powershell
python scripts/train_ml_rigor.py
python scripts/generate_shap_explanations.py
```

The training script selects the serving classifier by mean stratified 5-fold macro-F1 and saves its metadata while preserving existing metadata keys. The supplied class counts are imbalanced (Critical 554, High 405, Moderate 211, Low 30); macro-F1 is therefore reported alongside weighted-F1 and accuracy. FoS is related to the synthetic target construction, so the ablation is a dependence check, not proof that the feature is or is not leaked in real operational data.

### Verification

```powershell
pytest -q
python scripts/smoke_test.py
```

### Prototype limitations that remain

- The terrain, cross-section, and scenario animations are conceptual. They are not photogrammetric geometry, finite-element simulations or validated 3D failure mechanics.
- Browser speech recognition support varies and may rely on browser services; text input is the dependable fallback. The intent parser itself is rule-based and offline.
- The alert tone is user-triggered, and the S4 demo is synthetic. No always-on 60-second escalation daemon, real worker geofencing, emergency dispatch, SMS, email or Telegram integration is enabled.
- Monte Carlo parameter spreads are stated assumptions rather than distributions calibrated from field measurements. Inverse-velocity estimates are explicitly indicative.
- The image classifier and suggested cohesion/friction ranges are visual aids only, not geological identification or laboratory strength results.
- User roles, authentication, audit-grade authorization, formal drift gates and validated mitigation-gain estimates are not production implementations.

Read `docs/DEMO_RUNBOOK.md` for a five-minute presentation flow and `docs/VIVA_QA.md` for 30 viva questions.
