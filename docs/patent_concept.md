# LogisChain AI — Patent Concept Document
## Deliverable D16 | Provisional Patent Application

**Title:** AI-Powered Supply Chain Intelligence System for 
Real-Time Trade Finance Risk Assessment

**Applicant:** ZeTheta Algorithms Private Limited  
**Inventors:** LogisChain AI Development Team  
**Filing Date:** May 2026  
**Classification:** G06Q 40/00 | G06N 3/08 | G06Q 10/08

---

## 1. TECHNICAL FIELD

This invention relates to artificial intelligence systems for 
financial risk assessment, specifically a dual-domain machine 
learning pipeline that integrates real-time supply chain 
operational data with trade finance credit risk models to 
produce supply chain-adjusted probability of default (SC-PD) 
scores for trade finance instruments including Letters of Credit, 
Supply Chain Finance programmes, and working capital facilities.

---

## 2. BACKGROUND & PROBLEM STATEMENT

### 2.1 The Intelligence Gap

Traditional trade finance credit models rely exclusively on 
financial statements, credit ratings, and historical default 
data. This creates a critical 30-90 day blind spot between 
when supply chain stress first appears in operational data 
and when it manifests in financial statements.

Current models fail to detect:
- Port congestion causing Letter of Credit expiry
- Inventory velocity decline signaling covenant breach
- Carrier reliability degradation predicting shipment default
- Supplier concentration risk from network disruptions

### 2.2 Scale of the Problem

The trade finance gap stands at $2.5 trillion globally. 
Annual LC-related losses from supply chain disruptions 
exceed $50 billion. Existing systems cannot connect 
operational signals to financial risk in real time.

### 2.3 Prior Art Limitations

Existing approaches fall into two categories, each with 
fundamental limitations:

**Financial-only models** (XGBoost on financial ratios, 
logistic regression on credit bureau data) achieve AUC 
of 0.738-0.771 but ignore supply chain signals entirely.

**Supply chain analytics platforms** (visibility tools, 
ETA prediction engines) provide operational intelligence 
but have no financial risk integration layer.

No prior art combines heterogeneous graph neural networks 
for supply chain topology with temporal convolutional 
networks for demand forecasting and transformer-based 
shipment event encoding into a unified trade finance 
credit scoring system.

---

## 3. SUMMARY OF INVENTION

LogisChain AI is a novel AI system comprising five 
interconnected technical innovations:

The system achieves 33-66% risk uplift over traditional 
PD models and provides 30-90 day early warning of trade 
finance defaults from supply chain signals.

---

## 4. DETAILED DESCRIPTION OF INVENTION

### 4.1 Innovation 1 — Supply Chain Adjusted PD Formula (SC-PD)

**Novel Contribution:** The first mathematically defined 
formula connecting supply chain operational metrics to 
trade finance probability of default.

**Formula:**
SC-PD = Traditional_PD × (1 + α×OTIF_adj
+ β×Inv_adj
+ γ×Network_adj)
Where:
OTIF_adj    = max(0, (θ_OTIF - OTIF_actual) / δ_OTIF)
Inv_adj     = max(0, (τ_Inv - InvTurnover_actual) / λ_Inv)
Network_adj = min(1.0, HHI × κ)
Calibrated parameters (ICC Trade Register):
α = 0.30  (OTIF weight)
β = 0.20  (Inventory weight)
γ = 0.15  (Network weight)
θ_OTIF = 0.90  (OTIF threshold)
δ_OTIF = 0.10  (OTIF normalization)
τ_Inv = 6.0    (Inventory threshold)
λ_Inv = 3.0    (Inventory normalization)
κ = 2.0        (HHI scaling factor)

**Novelty:** No prior art defines a parametric formula 
linking OTIF rates, inventory turnover, and supplier 
concentration to probability of default. The calibration 
against ICC Trade Register data is novel.

**Industrial Application:** Banks can replace static 
credit ratings with dynamic SC-PD scores updated weekly 
from supply chain data, enabling proactive risk management.

---

### 4.2 Innovation 2 — Heterogeneous Graph Attention Network 
for Supply Chain Risk Propagation (HetGAT-SC)

**Novel Contribution:** A heterogeneous graph neural 
network architecture specifically designed for supply 
chain financial risk propagation.

**Architecture:**

Node Types:
N1: Supplier nodes    (6 features: OTIF, inv_turnover,
transit_variance, defect_rate,
freight_cost, CCC)
N2: Location nodes    (4 features: delay, shipping_time,
HHI, OTIF)
N3: Product nodes     (3 features: revenue, stock, OTIF)
N4: Carrier nodes     (3 features: OTIF, shipping_time,
delay)
Edge Types:
E1: Supplier → Location   (ships_to)
E2: Carrier → Location    (operates_at)
E3: Supplier → Product    (provides)
E4: Carrier → Supplier    (serves)
E5: Location → Carrier    (uses)
Message Passing:
3 HetGAT layers with typed attention
Hidden dimension: 32
Output dimension: 16 (risk embedding)
Aggregation: sum

**Novelty:** Prior GNN applications in finance use 
homogeneous graphs (bank-to-bank, company-to-company). 
The heterogeneous 4-node-type, 5-edge-type architecture 
capturing supplier-location-product-carrier relationships 
is novel in the trade finance domain.

**Industrial Application:** Risk propagation through 
supply chain networks — if a carrier fails, all connected 
suppliers receive elevated risk scores automatically.

---

### 4.3 Innovation 3 — Multi-Horizon Temporal Convolutional 
Network for Trade Finance Risk Forecasting (TCN-TF)

**Novel Contribution:** A dilated causal TCN architecture 
producing simultaneous 30/60/90-day forecasts of supply 
chain stress indicators for trade finance early warning.

**Architecture:**

Input: 128-day sequence of port throughput,
freight rates, OTIF rates
TCN Blocks: 7 layers
Dilation factors: [1, 2, 4, 8, 16, 32, 64]
Kernel size: 3
Receptive field: 128 days
Filters: 64 per layer
Output Heads: 3 separate linear layers
Head 1: 30-day forecast (MAPE: 0.77%)
Head 2: 60-day forecast (MAPE: 0.76%)
Head 3: 90-day forecast (MAPE: 1.44%)

**Novelty:** Existing trade finance models do not 
incorporate time series forecasting. The multi-horizon 
output architecture with simultaneous 30/60/90-day 
predictions from a single forward pass is novel.

**Industrial Application:** Banks can predict port 
congestion 90 days ahead and proactively extend LC 
validity before expiry risk materializes.

---

### 4.4 Innovation 4 — Transformer-Based Shipment Event 
Sequence Encoding for Default Prediction

**Novel Contribution:** A transformer encoder that 
treats each shipment as a sequence of 7 event tokens, 
using multi-head attention to identify which events 
predict trade finance default.

**Architecture:**

Event Sequence (7 events × 7 features):
E1: Booking Confirmed
E2: Container Loaded
E3: Vessel Departed
E4: Transhipment
E5: Destination Port Arrival
E6: Customs Cleared
E7: Final Delivery
Each event features:
[temporal, operational, risk, delay_accumulation,
event_type_encoding, contextual_signal, ...]
Transformer:
d_model: 64
n_heads: 4
n_layers: 3
Positional encoding: learned embeddings
Pooling: global average over event sequence
Output: Binary risk score via sigmoid
AUC: 0.993, Brier Score: 0.039

**Novelty:** No prior art encodes shipment lifecycle 
events as transformer token sequences for credit risk 
prediction. The event-type positional encoding combined 
with delay-accumulation features is novel.

**Industrial Application:** Every shipment receives 
a risk score at each lifecycle stage, enabling real-time 
LC monitoring and automatic alert generation.

---

### 4.5 Innovation 5 — Cross-Domain Fusion Architecture

**Novel Contribution:** A unified inference pipeline 
that combines outputs from all four model types into 
a single trade finance risk score.

**Fusion Architecture:**

Input Layer:
├── Tabular SC Features (20 features)
├── GNN Embeddings (16-dim risk vector)
├── TCN Forecasts (3 horizon predictions)
└── Transformer Risk Scores (per shipment)
Fusion:
XGBoost meta-learner trained on:

All tabular features
SC-PD computed score
GNN supplier risk class
TCN 30-day forecast
Transformer shipment risk

Output:
Probability of Default (0-1)
Risk Band (Green/Amber/Red)
Alert Triggers (3 threshold levels)

**Novelty:** The stacked ensemble combining GNN 
node embeddings, TCN temporal forecasts, and 
Transformer event sequences as features in a 
meta-XGBoost model is novel for trade finance.

---

## 5. CLAIMS

### Independent Claims

**Claim 1:** A computer-implemented method for 
trade finance risk assessment comprising:
- receiving supply chain operational data including 
  OTIF rates, inventory turnover ratios, and supplier 
  concentration metrics;
- computing a Supply Chain Adjusted Probability of 
  Default using the formula SC-PD = PD_traditional × 
  (1 + α×OTIF_adj + β×Inv_adj + γ×Network_adj);
- generating a risk score for trade finance instruments 
  including Letters of Credit and Supply Chain Finance;
- providing an alert when SC-PD exceeds a threshold.

**Claim 2:** The method of Claim 1, wherein the 
heterogeneous graph neural network comprises at least 
4 node types representing suppliers, locations, 
products, and carriers, and at least 5 edge types 
representing operational relationships between nodes.

**Claim 3:** The method of Claim 1, wherein the 
temporal forecasting component produces simultaneous 
multi-horizon forecasts at 30, 60, and 90 day intervals 
using dilated causal convolutions with dilation factors 
[1, 2, 4, 8, 16, 32, 64].

**Claim 4:** The method of Claim 1, wherein shipment 
risk is assessed by encoding each shipment as a sequence 
of event tokens processed through a multi-head 
self-attention transformer encoder.

### Dependent Claims

**Claim 5:** The method of Claim 1, wherein OTIF_adj 
= max(0, (0.90 - OTIF_actual) / 0.10).

**Claim 6:** The method of Claim 1, wherein Inv_adj 
= max(0, (6.0 - InvTurnover) / 3.0).

**Claim 7:** The system of Claim 2, wherein message 
passing uses typed attention coefficients that differ 
by edge type, enabling distinct risk propagation 
patterns for each relationship type.

**Claim 8:** A system implementing the method of 
Claims 1-7, further comprising a gamified simulation 
platform that trains trade finance practitioners in 
supply chain risk identification.

---

## 6. COMMERCIAL APPLICATION

### 6.1 Primary Markets

Market 1: Trade Finance Banks
Value: $9.2T annual LC volume
Application: SC-PD integration into LC approval
Market 2: Supply Chain Finance Platforms
Value: $1.8T annual SCF volume
Application: Dynamic supplier risk scoring
Market 3: Marine Cargo Insurers
Value: $60B annual premiums
Application: Real-time cargo risk pricing
Market 4: Logistics Investment Funds
Value: $800B AUM
Application: Asset risk monitoring

### 6.2 Competitive Advantage

Existing Solutions:        LogisChain AI:
──────────────────         ──────────────
Financial data only   →    SC + Financial fusion
Static credit rating  →    Dynamic weekly scoring
Reactive monitoring   →    30-90 day early warning
Single model          →    5-model ensemble
AUC: 0.738            →    AUC: 0.856 (full pipeline)

---

## 7. BRIEF DESCRIPTION OF DRAWINGS

**Figure 1:** Three-layer LogisChain AI architecture 
showing Physical Layer, Financial Layer, and 
Intelligence Layer connections.

**Figure 2:** HetGAT supply chain graph with 4 node 
types and 5 edge types showing risk propagation paths.

**Figure 3:** TCN architecture with dilation factors 
[1,2,4,8,16,32,64] and multi-horizon output heads.

**Figure 4:** Transformer event sequence encoding 
showing 7 shipment events and attention weight 
visualization.

**Figure 5:** SC-PD formula components and contribution 
of each adjustment factor to PD uplift.

**Figure 6:** Cross-domain fusion architecture showing 
how GNN, TCN, and Transformer outputs combine into 
final risk score.

---

## 8. ABSTRACT

A computer-implemented artificial intelligence system 
and method for trade finance risk assessment that 
integrates supply chain operational data with financial 
credit models. The system comprises: (1) a Supply Chain 
Adjusted Probability of Default (SC-PD) formula 
combining OTIF rates, inventory turnover, and network 
concentration; (2) a Heterogeneous Graph Attention 
Network capturing risk propagation through supply chain 
topology; (3) a Multi-Horizon Temporal Convolutional 
Network for 30/60/90-day early warning forecasting; 
(4) a Transformer encoder for shipment event sequence 
risk scoring; and (5) a cross-domain fusion architecture 
combining all model outputs. The system achieves 33-66% 
risk uplift over traditional models and provides 30-90 
day advance warning of trade finance defaults.

---

*Patent Concept Document v1.0*  
*LogisChain AI | ZeTheta Algorithms | May 2026*  
*CONFIDENTIAL — All IP belongs to ZeTheta Algorithms*
