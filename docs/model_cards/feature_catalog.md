# LogisChain AI — Feature Catalog
## Deliverable D3 | Minimum 50 Features

**Project:** LogisChain AI - Predictive Trade Finance & Logistics Valuation  
**Organization:** ZeTheta Algorithms Private Limited  
**Total Features:** 55  
**Last Updated:** May 2026

---

## Feature Categories

---

## CATEGORY 1: Supply Chain Operational Features

### F001 — OTIF Rate
| Attribute | Value |
|-----------|-------|
| **Name** | On-Time In-Full Rate |
| **Formula** | (Orders delivered on-time AND in-full) / Total Orders |
| **Data Source** | Kaggle Supply Chain Dataset (shipping_times, lead_times, availability, order_quantities) |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0.65 - 0.98 |
| **Financial Rationale** | From document: OTIF below 85% triggers credit downgrade signal for SCF portfolio. Single strongest leading indicator of supplier financial distress |
| **Data Quality** | High - computed from two reliable columns |
| **Risk Threshold** | < 85% = HIGH RISK |

---

### F002 — Lead Time Mean
| Attribute | Value |
|-----------|-------|
| **Name** | Average Lead Time |
| **Formula** | Mean of lead_times column across all shipments |
| **Data Source** | Kaggle Supply Chain Dataset (lead_times) |
| **Update Frequency** | Weekly |
| **Expected Range** | 1 - 30 days |
| **Financial Rationale** | Higher lead times increase working capital requirements and DIO component of CCC |
| **Data Quality** | High - direct measurement |
| **Risk Threshold** | > 25 days = elevated risk |

---

### F003 — Lead Time Variability (σ_LT)
| Attribute | Value |
|-----------|-------|
| **Name** | Transit Time Variance |
| **Formula** | Standard deviation of shipping_times per supplier |
| **Data Source** | Kaggle Supply Chain Dataset (shipping_times, supplier_name) |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.5 - 8.0 days |
| **Financial Rationale** | From document: Safety Stock = 1.28 × σ_LT at 90% service level. Every +3 days σ requires working capital facility adjustment |
| **Data Quality** | High - computed from reliable data |
| **Risk Threshold** | > 3 days = adjust working capital |

---

### F004 — Safety Stock Days
| Attribute | Value |
|-----------|-------|
| **Name** | Required Safety Stock in Days |
| **Formula** | 1.28 × σ_LT (from document formula at 90% service level) |
| **Data Source** | Derived from transit_time_variance |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.64 - 10.24 days |
| **Financial Rationale** | Directly increases DIO component of CCC. Higher safety stock = more working capital needed |
| **Data Quality** | High - derived from document formula |
| **Risk Threshold** | > 5 days = working capital stress |

---

### F005 — Inventory Turnover
| Attribute | Value |
|-----------|-------|
| **Name** | Inventory Turnover Ratio |
| **Formula** | Total COGS / Average Inventory Value where COGS = manufacturing_costs × number_of_products_sold and Inventory Value = stock_levels × price |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.5 - 15x (capped at 95th percentile) |
| **Financial Rationale** | From document: decline from 8x to 5x may trigger covenant breach, accelerating loan repayment obligations |
| **Data Quality** | Medium - capped to handle outliers |
| **Risk Threshold** | < 5x = covenant breach risk |

---

### F006 — Shipment Delay Days
| Attribute | Value |
|-----------|-------|
| **Name** | Average Shipment Delay |
| **Formula** | shipping_times - lead_times (negative = early) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | -28 to +10 days |
| **Financial Rationale** | From document: Port Congestion Index > 5 day avg delay = Letter of Credit expiry risk |
| **Data Quality** | High - direct computation |
| **Risk Threshold** | > 5 days = LC expiry risk |

---

### F007 — Fill Rate
| Attribute | Value |
|-----------|-------|
| **Name** | Order Fill Rate |
| **Formula** | number_of_products_sold / order_quantities (clipped 0-1) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per order |
| **Expected Range** | 0.0 - 1.0 |
| **Financial Rationale** | Fill rate erosion signals inventory depletion indicating cash flow problems and liquidity stress |
| **Data Quality** | High |
| **Risk Threshold** | < 0.85 = liquidity stress signal |

---

### F008 — Defect Rate
| Attribute | Value |
|-----------|-------|
| **Name** | Product Defect Rate |
| **Formula** | defect_rates column (direct from dataset) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per batch |
| **Expected Range** | 0.001 - 0.05 |
| **Financial Rationale** | High defect rates signal supplier quality failure, leading to trade finance default and warranty claims |
| **Data Quality** | High |
| **Risk Threshold** | Above portfolio mean = high risk |

---

### F009 — Stock Levels
| Attribute | Value |
|-----------|-------|
| **Name** | Current Inventory Stock Level |
| **Formula** | stock_levels column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Daily |
| **Expected Range** | 0 - 100 units |
| **Financial Rationale** | Low stock levels signal potential supply disruption and working capital stress |
| **Data Quality** | High |
| **Risk Threshold** | < 10 units = stockout risk |

---

### F010 — Days Inventory Outstanding (DIO)
| Attribute | Value |
|-----------|-------|
| **Name** | Days Inventory Outstanding |
| **Formula** | lead_times (proxy for days inventory held) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Weekly |
| **Expected Range** | 1 - 30 days |
| **Financial Rationale** | From document CCC formula: DIO = (Average Inventory / COGS) × 365. Component of Cash Conversion Cycle |
| **Data Quality** | Medium - proxy measurement |
| **Risk Threshold** | > 20 days = working capital stress |

---

### F011 — Days Sales Outstanding (DSO)
| Attribute | Value |
|-----------|-------|
| **Name** | Days Sales Outstanding |
| **Formula** | shipping_times (proxy for days to collect payment) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 1 - 10 days |
| **Financial Rationale** | From document CCC formula: DSO = (AR / Revenue) × 365. Higher DSO = slower cash collection |
| **Data Quality** | Medium - proxy measurement |
| **Risk Threshold** | > 8 days = collection risk |

---

### F012 — Days Payable Outstanding (DPO)
| Attribute | Value |
|-----------|-------|
| **Name** | Days Payable Outstanding |
| **Formula** | manufacturing_lead_time (proxy for days to pay suppliers) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 1 - 30 days |
| **Financial Rationale** | From document CCC formula: DPO = (AP / COGS) × 365. Higher DPO improves working capital |
| **Data Quality** | Medium - proxy measurement |
| **Risk Threshold** | < 10 days = payable pressure |

---

### F013 — Cash Conversion Cycle (CCC)
| Attribute | Value |
|-----------|-------|
| **Name** | Cash Conversion Cycle |
| **Formula** | DIO + DSO - DPO = lead_times + shipping_times - manufacturing_lead_time |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Weekly |
| **Expected Range** | -15 to 30 days |
| **Financial Rationale** | From document: CCC is single most important metric for working capital lending. Extend +20 days = liquidity deterioration signal |
| **Data Quality** | Medium - proxy measurements |
| **Risk Threshold** | > mean + 20 days = covenant breach |

---

### F014 — Production Volume
| Attribute | Value |
|-----------|-------|
| **Name** | Manufacturing Production Volume |
| **Formula** | production_volumes column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 100 - 1000 units |
| **Financial Rationale** | Production volume decline signals demand erosion or capacity constraint, increasing credit risk |
| **Data Quality** | High |
| **Risk Threshold** | > 20% decline = distress signal |

---

### F015 — Manufacturing Lead Time
| Attribute | Value |
|-----------|-------|
| **Name** | Time to Manufacture Product |
| **Formula** | manufacturing_lead_time column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per production run |
| **Expected Range** | 1 - 30 days |
| **Financial Rationale** | Longer manufacturing lead times increase pipeline inventory and working capital requirements |
| **Data Quality** | High |
| **Risk Threshold** | > 20 days = high working capital need |

---

## CATEGORY 2: Financial & Credit Risk Features

### F016 — Traditional Probability of Default (PD)
| Attribute | Value |
|-----------|-------|
| **Name** | Baseline Probability of Default |
| **Formula** | ICC Trade Register benchmark = 2.8% for trade finance |
| **Data Source** | ICC Trade Register (document reference) |
| **Update Frequency** | Annual |
| **Expected Range** | 0.003 - 0.05 |
| **Financial Rationale** | From document: baseline PD from ICC Trade Register. Used as anchor for SC-PD computation |
| **Data Quality** | High - industry benchmark |
| **Risk Threshold** | > 4% = elevated risk |

---

### F017 — Supply Chain Adjusted PD (SC-PD)
| Attribute | Value |
|-----------|-------|
| **Name** | Supply Chain Adjusted Probability of Default |
| **Formula** | Traditional PD × (1 + 0.3×OTIF_adj + 0.2×Inv_adj + 0.15×Network_adj) |
| **Data Source** | Derived - document Section A5.4 exact formula |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.028 - 0.115 |
| **Financial Rationale** | Core LogisChain AI innovation. Augments traditional PD with supply chain health indicators. 33-66% risk uplift over traditional PD |
| **Data Quality** | High - document formula |
| **Risk Threshold** | > Traditional PD × 1.3 = high risk |

---

### F018 — OTIF Adjustment Factor
| Attribute | Value |
|-----------|-------|
| **Name** | OTIF-Based PD Adjustment |
| **Formula** | max(0, (90% - OTIF_actual) / 10%) |
| **Data Source** | Derived from OTIF rate |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.0 - 2.5 |
| **Financial Rationale** | From document exact formula. Penalizes suppliers with OTIF below 90% threshold |
| **Data Quality** | High |
| **Risk Threshold** | > 0.5 = significant PD uplift |

---

### F019 — Inventory Health Adjustment
| Attribute | Value |
|-----------|-------|
| **Name** | Inventory Turnover PD Adjustment |
| **Formula** | max(0, (6.0 - InvTurnover_actual) / 3.0) |
| **Data Source** | Derived from inventory turnover |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.0 - 1.67 |
| **Financial Rationale** | From document exact formula. Penalizes suppliers with inventory turnover below 6x |
| **Data Quality** | High |
| **Risk Threshold** | > 0.4 = inventory stress |

---

### F020 — Network Resilience Adjustment
| Attribute | Value |
|-----------|-------|
| **Name** | Supply Network Resilience Factor |
| **Formula** | min(1.0, supplier_concentration_hhi × 2) |
| **Data Source** | Derived from HHI |
| **Update Frequency** | Quarterly |
| **Expected Range** | 0.0 - 1.0 |
| **Financial Rationale** | From document: 1.0 - min(1.0, AltSupplierCount/3). Higher concentration = lower resilience = higher PD |
| **Data Quality** | Medium - HHI proxy |
| **Risk Threshold** | > 0.6 = concentration risk |

---

### F021 — Revenue Generated
| Attribute | Value |
|-----------|-------|
| **Name** | Total Revenue per Transaction |
| **Formula** | revenue_generated column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per transaction |
| **Expected Range** | $100 - $10,000 |
| **Financial Rationale** | Revenue size determines exposure at default (EAD) and working capital facility sizing |
| **Data Quality** | High |
| **Risk Threshold** | N/A - used for sizing |

---

### F022 — Manufacturing Cost
| Attribute | Value |
|-----------|-------|
| **Name** | Cost of Goods Sold Proxy |
| **Formula** | manufacturing_costs × number_of_products_sold |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per production run |
| **Expected Range** | $100 - $100,000 |
| **Financial Rationale** | COGS determines gross margin and debt service capacity. Used in inventory turnover computation |
| **Data Quality** | High |
| **Risk Threshold** | > 80% of revenue = margin stress |

---

### F023 — Freight Cost per Unit
| Attribute | Value |
|-----------|-------|
| **Name** | Transportation Cost per Unit Shipped |
| **Formula** | shipping_costs / number_of_products_sold |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | $0.001 - $0.50 |
| **Financial Rationale** | From document: freight cost volatility directly affects COGS, operating margins, and credit risk metrics |
| **Data Quality** | High |
| **Risk Threshold** | High volatility = credit risk signal |

---

### F024 — Inventory Value
| Attribute | Value |
|-----------|-------|
| **Name** | Total Inventory Value |
| **Formula** | stock_levels × price |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Daily |
| **Expected Range** | $0 - $10,000 |
| **Financial Rationale** | Inventory value determines collateral quality for warehouse receipt finance and working capital loans |
| **Data Quality** | High |
| **Risk Threshold** | Declining trend = liquidation risk |

---

### F025 — Price per Unit
| Attribute | Value |
|-----------|-------|
| **Name** | Product Unit Price |
| **Formula** | price column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per transaction |
| **Expected Range** | $1.70 - $99.17 |
| **Financial Rationale** | Unit price determines cargo value and insurance premium calculation basis |
| **Data Quality** | High |
| **Risk Threshold** | N/A - used for valuation |

---

### F026 — CCC Stress Flag
| Attribute | Value |
|-----------|-------|
| **Name** | Cash Conversion Cycle Stress Indicator |
| **Formula** | 1 if CCC > median CCC else 0 |
| **Data Source** | Derived from CCC |
| **Update Frequency** | Weekly |
| **Expected Range** | 0 or 1 |
| **Financial Rationale** | Binary signal for working capital stress. Used as target variable for CCC prediction model |
| **Data Quality** | High |
| **Risk Threshold** | = 1 means stress |

---

### F027 — Default Risk Flag
| Attribute | Value |
|-----------|-------|
| **Name** | Trade Finance Default Indicator |
| **Formula** | 1 if inspection_results == Fail else 0 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per inspection |
| **Expected Range** | 0 or 1 |
| **Financial Rationale** | Primary target variable for credit risk models. Proxy for trade finance default in absence of actual default data |
| **Data Quality** | Medium - proxy variable |
| **Risk Threshold** | = 1 means default |

---

## CATEGORY 3: Network & Concentration Features

### F028 — Supplier Concentration HHI (Global)
| Attribute | Value |
|-----------|-------|
| **Name** | Herfindahl-Hirschman Index - Portfolio Level |
| **Formula** | Σ(supplier_revenue_share²) across all suppliers |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.15 - 0.65 |
| **Financial Rationale** | From document: supplier concentration > 60% single source = counterparty concentration risk. Require credit insurance |
| **Data Quality** | High |
| **Risk Threshold** | > 0.25 = moderate concentration |

---

### F029 — Supplier Concentration HHI (Per Location)
| Attribute | Value |
|-----------|-------|
| **Name** | Location-Level Supplier Concentration |
| **Formula** | Σ(supplier_revenue_share²) within each location |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.20 - 0.80 |
| **Financial Rationale** | Location-level concentration captures geographic counterparty risk more accurately than portfolio-level HHI |
| **Data Quality** | High |
| **Risk Threshold** | > 0.40 = location concentration risk |

---

### F030 — Carrier Reliability Score
| Attribute | Value |
|-----------|-------|
| **Name** | Shipping Carrier OTIF Rate |
| **Formula** | Mean OTIF rate per shipping carrier |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.65 - 0.95 |
| **Financial Rationale** | Unreliable carriers increase delay risk, triggering LC expiry and cargo insurance claims |
| **Data Quality** | High |
| **Risk Threshold** | < 0.85 = unreliable carrier |

---

### F031 — Transport Mode OTIF
| Attribute | Value |
|-----------|-------|
| **Name** | OTIF Rate by Transportation Mode |
| **Formula** | Mean OTIF per transportation mode (Road/Air/Sea/Rail) |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.65 - 0.95 |
| **Financial Rationale** | From document: each mode has distinct cost-speed-reliability tradeoffs. Modal risk affects trade finance pricing |
| **Data Quality** | High |
| **Risk Threshold** | < 0.85 = modal risk |

---

### F032 — Transport Mode Average Delay
| Attribute | Value |
|-----------|-------|
| **Name** | Average Delay by Transportation Mode |
| **Formula** | Mean shipment_delay_days per transportation mode |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Weekly |
| **Expected Range** | -20 to +10 days |
| **Financial Rationale** | Modal delay patterns inform LC validity period setting and cargo insurance duration |
| **Data Quality** | High |
| **Risk Threshold** | > 5 days = LC risk |

---

### F033 — Supplier Count per Location
| Attribute | Value |
|-----------|-------|
| **Name** | Number of Unique Suppliers per Location |
| **Formula** | Count of unique supplier_names per location |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 1 - 5 |
| **Financial Rationale** | Fewer suppliers per location = higher geographic concentration risk = higher trade finance exposure |
| **Data Quality** | High |
| **Risk Threshold** | < 2 = geographic concentration |

---

### F034 — Route Risk Score
| Attribute | Value |
|-----------|-------|
| **Name** | Average Defect Rate by Trade Route |
| **Formula** | Mean defect_rates per route (A/B/C) |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.01 - 0.05 |
| **Financial Rationale** | High-risk routes have elevated damage probability affecting marine cargo insurance pricing |
| **Data Quality** | High |
| **Risk Threshold** | Above portfolio mean = risky route |

---

### F035 — On Time Rate
| Attribute | Value |
|-----------|-------|
| **Name** | Percentage of Shipments Delivered On Time |
| **Formula** | shipping_times <= lead_times (binary, averaged) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0.70 - 0.95 |
| **Financial Rationale** | On-time component of OTIF. Directly determines LC expiry risk |
| **Data Quality** | High |
| **Risk Threshold** | < 0.83 = delay risk |

---

## CATEGORY 4: Transportation & Logistics Features

### F036 — Shipping Time
| Attribute | Value |
|-----------|-------|
| **Name** | Actual Transit Time |
| **Formula** | shipping_times column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 1 - 10 days |
| **Financial Rationale** | Actual transit time vs promised determines delay and LC validity buffer requirement |
| **Data Quality** | High |
| **Risk Threshold** | > lead_times = delayed |

---

### F037 — Lead Time
| Attribute | Value |
|-----------|-------|
| **Name** | Promised Delivery Lead Time |
| **Formula** | lead_times column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per order |
| **Expected Range** | 1 - 30 days |
| **Financial Rationale** | Promised lead time determines LC tenor and working capital facility duration |
| **Data Quality** | High |
| **Risk Threshold** | > 25 days = high financing need |

---

### F038 — Carrier Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Shipping Carrier (Encoded) |
| **Formula** | LabelEncoder(shipping_carriers): A=0, B=1, C=2 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0, 1, 2 |
| **Financial Rationale** | Carrier identity predicts reliability and delay risk patterns |
| **Data Quality** | High |
| **Risk Threshold** | Carrier A encoded risk = highest |

---

### F039 — Location Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Delivery Location (Encoded) |
| **Formula** | LabelEncoder(location): Mumbai=0, Delhi=1, etc. |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0 - 4 |
| **Financial Rationale** | Location determines geographic risk, port connectivity and logistics infrastructure quality |
| **Data Quality** | High |
| **Risk Threshold** | N/A - categorical |

---

### F040 — Product Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Product Type (Encoded) |
| **Formula** | LabelEncoder(product_type): cosmetics=0, haircare=1, skincare=2 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per transaction |
| **Expected Range** | 0, 1, 2 |
| **Financial Rationale** | Product type determines cargo value sensitivity, insurance premium category and demand volatility |
| **Data Quality** | High |
| **Risk Threshold** | N/A - categorical |

---

### F041 — Route Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Trade Route (Encoded) |
| **Formula** | LabelEncoder(routes): Route A=0, Route B=1, Route C=2 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0, 1, 2 |
| **Financial Rationale** | Trade route determines geopolitical risk exposure and alternative routing options |
| **Data Quality** | High |
| **Risk Threshold** | N/A - categorical |

---

### F042 — Transport Mode Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Transportation Mode (Encoded) |
| **Formula** | LabelEncoder(transportation_modes): Air=0, Road=1, Rail=2, Sea=3 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0 - 3 |
| **Financial Rationale** | From document: each mode has distinct cost-speed-reliability tradeoffs affecting insurance and finance pricing |
| **Data Quality** | High |
| **Risk Threshold** | N/A - categorical |

---

### F043 — Shipping Cost
| Attribute | Value |
|-----------|-------|
| **Name** | Total Shipping Cost |
| **Formula** | shipping_costs column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per shipment |
| **Expected Range** | $1 - $10,000 |
| **Financial Rationale** | Shipping cost volatility signals freight market stress, affecting trade finance pricing spread |
| **Data Quality** | High |
| **Risk Threshold** | > 200% spike in 60 days = COGS inflation |

---

### F044 — Order Quantity
| Attribute | Value |
|-----------|-------|
| **Name** | Units Ordered per Transaction |
| **Formula** | order_quantities column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per order |
| **Expected Range** | 1 - 100 units |
| **Financial Rationale** | Large order quantities increase LC amount and exposure at default (EAD) |
| **Data Quality** | High |
| **Risk Threshold** | > 80 units = large exposure |

---

### F045 — Number of Products Sold
| Attribute | Value |
|-----------|-------|
| **Name** | Actual Units Sold |
| **Formula** | number_of_products_sold column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per transaction |
| **Expected Range** | 1 - 1000 units |
| **Financial Rationale** | Sales volume determines COGS computation and inventory turnover calculation |
| **Data Quality** | High |
| **Risk Threshold** | Declining trend = demand erosion |

---

## CATEGORY 5: Cross-Domain Fusion Features

### F046 — Working Capital Velocity Index (WCVI)
| Attribute | Value |
|-----------|-------|
| **Name** | Working Capital Velocity Index |
| **Formula** | (Inventory Velocity Z-Score + Receivables Velocity Z-Score - Payables Velocity Z-Score) / 3 |
| **Data Source** | Derived - document Section A5.4 Fusion Feature 2 |
| **Update Frequency** | Weekly |
| **Expected Range** | -3.0 to +3.0 |
| **Financial Rationale** | From document: declining WCVI signals CCC extension before it appears in financial statements. 30-60 day early warning |
| **Data Quality** | Medium - proxy computation |
| **Risk Threshold** | Declining trend = CCC stress warning |

---

### F047 — Trade Route Financial Stress Index (TRFSI)
| Attribute | Value |
|-----------|-------|
| **Name** | Trade Route Financial Stress Index |
| **Formula** | w1×PortCongestion + w2×FreightVolatility + w3×LCRejectionRate + w4×PaymentDelay |
| **Data Source** | Derived - document Section A5.4 Fusion Feature 3 |
| **Update Frequency** | Weekly |
| **Expected Range** | 0.0 - 1.0 |
| **Financial Rationale** | From document: TRFSI spike on trade lane precedes increased trade finance losses by 30-45 days |
| **Data Quality** | Medium - proxy weights |
| **Risk Threshold** | > 0.6 = elevated lane risk |

---

### F048 — GNN Supplier Risk Embedding
| Attribute | Value |
|-----------|-------|
| **Name** | Graph Neural Network Risk Embedding |
| **Formula** | 16-dimensional embedding from HetGAT message passing (3 layers) |
| **Data Source** | Derived from GNN model (src/models/gnn.py) |
| **Update Frequency** | Weekly model inference |
| **Expected Range** | Continuous vector [-1, 1]^16 |
| **Financial Rationale** | Captures network context risk beyond individual supplier data. Risk propagation through supply chain graph |
| **Data Quality** | High - neural network output |
| **Risk Threshold** | Risk class = 1 from GNN classifier |

---

### F049 — TCN Demand Forecast (30-day)
| Attribute | Value |
|-----------|-------|
| **Name** | 30-Day Port Throughput Forecast |
| **Formula** | TCN model output (30-day horizon) from dilated causal convolutions |
| **Data Source** | Derived from TCN model (src/models/tcn.py) |
| **Update Frequency** | Daily model inference |
| **Expected Range** | 80,000 - 180,000 TEU/day |
| **Financial Rationale** | Port throughput forecast predicts congestion, LC expiry risk and working capital requirements 30 days ahead |
| **Data Quality** | High - MAPE 0.77% |
| **Risk Threshold** | > 150,000 TEU = congestion risk |

---

### F050 — Transformer Shipment Risk Score
| Attribute | Value |
|-----------|-------|
| **Name** | Shipment-Level Risk Score from Transformer |
| **Formula** | Sigmoid output of Transformer encoder over 7-event sequence |
| **Data Source** | Derived from Transformer model (src/models/transformer.py) |
| **Update Frequency** | Per shipment |
| **Expected Range** | 0.0 - 1.0 |
| **Financial Rationale** | Individual shipment risk score combining all event-level signals. Feeds into LC pricing and cargo insurance |
| **Data Quality** | High - AUC 0.993 |
| **Risk Threshold** | > 0.5 = high risk shipment |

---

### F051 — Inspection Result
| Attribute | Value |
|-----------|-------|
| **Name** | Quality Inspection Outcome |
| **Formula** | inspection_results column: Pass/Fail/Pending |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per inspection |
| **Expected Range** | Categorical: Pass, Fail, Pending |
| **Financial Rationale** | Primary proxy for trade finance default. Fail = delivery/quality problem = default risk |
| **Data Quality** | High |
| **Risk Threshold** | Fail = default |

---

### F052 — Availability
| Attribute | Value |
|-----------|-------|
| **Name** | Product Availability Level |
| **Formula** | availability column (direct) |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Daily |
| **Expected Range** | 1 - 100 |
| **Financial Rationale** | Low availability signals inventory depletion and potential SCF invoice rejection |
| **Data Quality** | High |
| **Risk Threshold** | < order_quantities = not in full |

---

### F053 — Total COGS
| Attribute | Value |
|-----------|-------|
| **Name** | Total Cost of Goods Sold |
| **Formula** | manufacturing_costs × number_of_products_sold |
| **Data Source** | Derived from Kaggle dataset |
| **Update Frequency** | Per production run |
| **Expected Range** | $100 - $100,000 |
| **Financial Rationale** | COGS is denominator in inventory turnover calculation and determines gross margin |
| **Data Quality** | High |
| **Risk Threshold** | > 85% of revenue = margin squeeze |

---

### F054 — Inventory Turnover Capped
| Attribute | Value |
|-----------|-------|
| **Name** | Outlier-Capped Inventory Turnover |
| **Formula** | inventory_turnover clipped at 95th percentile to remove extreme outliers |
| **Data Source** | Derived from inventory_turnover |
| **Update Frequency** | Monthly |
| **Expected Range** | 0.14 - 36x |
| **Financial Rationale** | Capped version used in ML models to prevent extreme outliers from dominating model training |
| **Data Quality** | High - outlier corrected |
| **Risk Threshold** | < 5x = covenant breach risk |

---

### F055 — Customer Demographics Encoded
| Attribute | Value |
|-----------|-------|
| **Name** | Customer Demographic Segment |
| **Formula** | LabelEncoder(customer_demographics): Female=0, Male=1, Non-binary=2, Unknown=3 |
| **Data Source** | Kaggle Supply Chain Dataset |
| **Update Frequency** | Per customer |
| **Expected Range** | 0 - 3 |
| **Financial Rationale** | Demographic segments have different demand volatility patterns affecting inventory and working capital needs |
| **Data Quality** | Medium - 30% Unknown values |
| **Risk Threshold** | N/A - segmentation variable |

---

## Feature Summary Statistics

| Category | Features | Avg Data Quality | Key Risk Threshold |
|----------|----------|------------------|--------------------|
| SC Operational | F001-F015 | High | OTIF < 85% |
| Financial | F016-F027 | High | SC-PD > 4.5% |
| Network | F028-F035 | High | HHI > 0.25 |
| Transportation | F036-F045 | High | Delay > 5 days |
| Cross-Domain | F046-F055 | Medium-High | Risk Score > 0.5 |

---

## Data Quality Framework

From document Section A4.3:

| Dimension | Assessment | Strategy |
|-----------|------------|----------|
| Completeness | 100% (no nulls after cleaning) | Dropna applied |
| Timeliness | Static dataset - no real-time | Synthetic augmentation |
| Accuracy | High for operational features | Cross-validation |
| Consistency | Single source - high consistency | Feature scaling |
| Manipulation Resistance | Low (no fraud detection) | Document limitation |

---

*Feature Catalog v1.0 | LogisChain AI | ZeTheta Algorithms*