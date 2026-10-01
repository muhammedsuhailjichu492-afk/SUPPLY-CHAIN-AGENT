# AI Supply Chain & Procurement Intelligence Agent

An autonomous system that predicts demand, monitors inventory, evaluates suppliers,
and recommends procurement decisions — built to minimize stock-outs and excess
inventory across manufacturing, logistics, warehouse, and procurement operations.

**Pipeline:** Forecasting → Inventory Analysis → Supplier RAG → Risk Prediction →
Procurement Recommendation → Approval

## Architecture

```
supply-chain-agent/
├── backend/
│   ├── main.py                 FastAPI app — all REST endpoints
│   ├── database.py             SQLAlchemy engine/session (SQLite by default)
│   ├── models.py                ORM tables: Product, DemandHistory, InventoryRecord,
│   │                            Supplier, SupplierProduct, Order, DemandForecast,
│   │                            RiskAssessment, Recommendation
│   ├── schemas.py               Pydantic request/response models
│   ├── data_generator.py        Synthetic data generator (products, 2yrs demand
│   │                            history, suppliers, orders)
│   ├── forecasting/
│   │   ├── xgboost_model.py     Feature-engineered XGBoost regressor (lags,
│   │   │                        rolling stats, seasonality) — the primary,
│   │   │                        always-available forecasting model
│   │   └── lstm_model.py        PyTorch LSTM sequence model — optional; the app
│   │                            runs fine without torch installed
│   ├── inventory/
│   │   └── analyzer.py          Reorder-point / safety-stock model: days of
│   │                            cover, stockout risk, recommended order qty
│   ├── supplier_rag/
│   │   ├── embeddings.py        TF-IDF embedder (default, zero downloads) with
│   │   │                        an optional sentence-transformers backend
│   │   └── retriever.py         Retrieval over supplier profiles, scoped to
│   │                            suppliers that can actually fulfil the product
│   ├── risk/
│   │   └── risk_model.py        RandomForest supplier-risk classifier with
│   │                            explainable top-factor output
│   └── agent/
│       ├── llm_client.py        Local Ollama client (no paid API key) with a
│       │                        deterministic template fallback
│       └── orchestrator.py      Chains every stage into one agentic decision,
│                                persisted as a pending Recommendation
├── frontend/                    Plain HTML/CSS/JS dashboard (no build step)
├── train_models.py              One-shot: generate data + train every model
└── requirements.txt
```

**Why this stack:** SQLite needs no server to set up (swap `DATABASE_URL` for
Postgres in production); XGBoost is the reliable default forecaster while the
LSTM is available as a drop-in alternative; the supplier RAG uses TF-IDF so it
works with zero internet access, with sentence-transformers as an upgrade path;
and the agent's reasoning step uses a local Ollama model instead of a paid API,
falling back to a clear deterministic explanation if Ollama isn't running.

## Setup (Windows / macOS / Linux, VS Code)

1. **Open the folder in VS Code** and open a terminal (`` Ctrl+` ``).

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```
   The LSTM (`torch`) is listed but optional — if the install is slow/large on
   your machine, comment out the `torch` line in `backend/requirements.txt` and
   the app will automatically use only the XGBoost forecaster.

4. **Generate synthetic data and pretrain every model:**
   ```bash
   python train_models.py
   ```
   This creates `data/supply_chain.db` (SQLite) and trained model files under
   `models/`. Re-run any time you want a fresh dataset.

5. **Start the API (also serves the dashboard):**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

6. **Open the dashboard:** go to **http://localhost:8000/** in your browser.
   Interactive API docs are at **http://localhost:8000/docs**.

### Optional: enable the local LLM agent reasoning (Ollama)

By default the agent's recommendation text is a clear, factual template. To get
LLM-generated reasoning instead, with no paid API key:
```bash
# install from https://ollama.com, then:
ollama pull llama3.1
```
The app detects Ollama automatically — nothing else to configure. Set
`OLLAMA_MODEL` as an environment variable to use a different local model.

### Optional: switch to PostgreSQL

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/supply_chain"
pip install psycopg2-binary
python train_models.py
```

## Using the dashboard

- **Overview** — pick a product, run the full agent pipeline, and see the
  generated recommendation immediately.
- **Forecasting** — chart historical demand against an XGBoost or LSTM
  forecast for any horizon.
- **Inventory** — days of cover, stockout risk, and the reorder-point/
  safety-stock math behind the recommended order quantity.
- **Supplier RAG** — free-text search over supplier profiles ("reliable
  fast-lead-time supplier in East Asia with low defect rate").
- **Risk Prediction** — every supplier's ML-predicted risk score with its
  top contributing factors.
- **Recommendations** — the agent's pending procurement decisions; approve
  or reject each one (the human-in-the-loop step).

## Key API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/pipeline/run` | Run the full agentic pipeline for a product |
| POST | `/api/forecast` | Run just the forecasting stage |
| GET | `/api/inventory/{id}/analysis` | Inventory analysis for one product |
| GET | `/api/suppliers/search?query=...` | RAG search over supplier profiles |
| GET | `/api/risk/{supplier_id}` | Risk prediction for one supplier |
| GET | `/api/recommendations` | List recommendations (filter by `status`) |
| POST | `/api/recommendations/{id}/decision` | Approve/reject a recommendation |

Full interactive documentation: **http://localhost:8000/docs**

## Notes on the synthetic dataset

`data_generator.py` builds 12 products across 5 categories (electronics,
packaging, raw metals, fasteners, safety equipment), 2 years of daily demand
history with trend + weekly/yearly seasonality + noise, 10 suppliers with
realistic profile text, and historical purchase orders that drive the risk
model's features. Swap in real data by pointing `DemandHistory`, `Supplier`,
and `Order` at your own tables/CSV imports — every downstream model
(forecasting, RAG, risk, agent) reads from the same ORM models, so no other
code needs to change.
