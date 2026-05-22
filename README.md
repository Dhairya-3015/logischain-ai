# 🚢 LogisChain AI
## Predictive Trade Finance & Logistics Valuation System

![LogisChain AI](https://img.shields.io/badge/LogisChain-AI-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

---

## 📋 Project Overview

LogisChain AI is a **dual-domain AI system** that embeds supply chain 
intelligence directly into financial risk models for:
- Trade Finance Desks
- Working Capital Management
- Supply Chain Finance (SCF) Platforms
- Credit Risk Assessment

**Submitted to:** ZeTheta Algorithms Private Limited  
**Category:** Data Science | Supply Chain Analytics | Trade Finance | AI/ML  
**Timeline:** 15 Days  

---

## 🎯 Problem Statement

Financial institutions approve Letters of Credit without checking port 
congestion. Working capital lenders extend revolving credit without 
monitoring inventory velocity. LogisChain AI closes this intelligence 
gap by integrating operational supply chain data into financial risk models.

---

## 🏗️ System Architecture

---

## 📊 Model Performance

| Model | Metric | Our Result | Target | Status |
|-------|--------|-----------|--------|--------|
| XGBoost Credit Risk | AUC | 0.639 | > 0.771 | ⚠️ |
| GNN Supply Chain | Nodes/Edges | 210/1500 | 200+/1000+ | ✅ |
| TCN Forecasting | MAPE 30-day | 0.77% | < 12% | ✅ |
| Transformer Shipment | AUC | 0.993 | > 0.80 | ✅ |
| Trade Finance Model | Gini | 0.555 | > 0.55 | ✅ |
| CCC Prediction | AUC | 0.911 | - | ✅ |
| SC-PD Model | Risk Uplift | 65.9% | ~33% | ✅ |

---

## 🎮 LogisChain Lab — Gamified Simulation

LogisChain Lab is a **gamified simulation platform** where learners 
manage trade finance portfolios while responding to supply chain disruptions.

### Game Modes

#### Mode 1 — Trade Finance Portfolio Management
![Trade Finance Mode](docs/screenshots/trade_finance.png)

#### Mode 2 — Supply Chain Finance Pricing
![SCF Mode](docs/screenshots/supply_chain_finance.png)

#### Mode 3 — Cargo Insurance Underwriter
![Cargo Mode](docs/screenshots/cargo_insurance.png)

#### Mode 4 — Logistics Investment Analyst
![Logistics Mode](docs/screenshots/logistics_investment.png)

### 🏆 Scoring System (1000 Points)

| Dimension | Points | Description |
|-----------|--------|-------------|
| Financial Performance | 300 | Risk-adjusted return, P&L |
| Risk Management | 250 | Portfolio concentration, early warning |
| SC Intelligence Use | 200 | SC data incorporation into decisions |
| Decision Speed | 100 | Response time to disruption events |
| Learning Progression | 150 | Improvement across simulation rounds |

### 🎖️ Certification Levels

| Level | Score | Badge | Industry Equivalent |
|-------|-------|-------|---------------------|
| Novice | 0-399 | 🥉 Bronze | Trainee (0-6 months) |
| Practitioner | 400-599 | 🥈 Silver | Junior Analyst |
| Specialist | 600-749 | 🥇 Gold | Senior Analyst |
| Expert | 750-899 | 🏆 Platinum | Team Lead |
| Master | 900-1000 | 💎 Diamond | Head of Trade Finance |

---

## 🚨 Scenarios Implemented

| Scenario | SC Impact | Financial Impact | Difficulty |
|----------|-----------|-----------------|------------|
| ⚓ Port Congestion | 5-15 day delays | LC expiry risk | Medium |
| 🚢 Carrier Bankruptcy | Stranded cargo | Insurance claims | High |
| 🏗️ Suez Canal Blockage | Lane shutdown | Massive freight spike | Very High |
| ⚠️ Supplier Quality Failure | Product recalls | Trade finance default | Medium |
| 📊 Demand Whiplash | ±40% swing | Covenant breach risk | High |
| 💰 Commodity Price Shock | Cost spike | Margin compression | Medium |

---

## 📈 Key Financial Models

### Supply Chain Adjusted PD (SC-PD)

### Cash Conversion Cycle (CCC)

---

## 🗂️ Project Structure

---

## 🚀 Installation & Setup

### Prerequisites

### Step 1: Clone Repository
```bash
git clone https://github.com/ZethetaIntern/logischain-ai
cd logischain-ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Data Pipeline
```bash
python src/data/data_loader.py
python src/features/feature_engineering.py
```

### Step 5: Run Models
```bash
python src/models/xgboost_model.py
python src/models/gnn.py
python src/models/tcn.py
python src/models/transformer.py
python src/financial/financial_models.py
```

### Step 6: Launch Simulation
```bash
streamlit run demo/app.py
```

---

## 📦 Dependencies

---

## 📊 Data Sources

| Source | Type | Use |
|--------|------|-----|
| Kaggle Supply Chain Dataset | Real (100 rows) | Primary dataset |
| Synthetic Augmentation | Generated | Model training |
| Yahoo Finance | Real | Financial benchmarks |
| ICC Trade Register | Reference | PD benchmarks |
| Document Formulas | Reference | Feature computation |

---

## ⚠️ Limitations

---

## 🔬 Case Studies Referenced

| Case Study | Year | Financial Impact | LogisChain AI Application |
|------------|------|-----------------|--------------------------|
| Ever Given / Suez Blockage | 2021 | $9.6B/day | LC expiry detection |
| COVID Supply Chain | 2020-22 | $2.5T gap | Phase detection |
| Greensill Capital | 2021 | $10B losses | Concentration alerts |
| Hanjin Bankruptcy | 2016 | $14.5B stranded | Carrier health monitoring |
| Qingdao Port Fraud | 2014 | $3.6B fraud | Physical-financial cross-reference |

---

## 👥 Acknowledgements

- **ZeTheta Algorithms** — Project design and administration
- **ICC Trade Register** — Default rate benchmarks
- **Kaggle** — Supply chain dataset
- **PyTorch Geometric** — GNN implementation

---

## 📄 License

This project and all IP arising from submissions is the exclusive 
property of Zetheta Algorithms Private Limited as per the 
confidentiality terms accepted at project commencement.

---

*LogisChain AI v1.0 | ZeTheta Algorithms | 2026*

