import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ── PAGE CONFIG ──────────────────────────
st.set_page_config(
    page_title="LogisChain Lab",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: red;
        font-weight: bold;
    }
    .risk-low {
        color: green;
        font-weight: bold;
    }
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/features/features.csv")
        return df
    except:
        # Generate sample data if not found
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'supplier_name': np.random.choice(
                [f'Supplier {i}' for i in range(1,6)], n
            ),
            'otif': np.random.uniform(0.65, 0.95, n),
            'shipment_delay_days': np.random.uniform(-20, 10, n),
            'inventory_turnover_capped': np.random.uniform(3, 12, n),
            'ccc': np.random.uniform(3, 25, n),
            'defect_rates': np.random.uniform(0.01, 0.05, n),
            'freight_cost_per_unit': np.random.uniform(0.01, 0.5, n),
            'sc_pd': np.random.uniform(0.02, 0.12, n),
            'location': np.random.choice(
                ['Mumbai', 'Delhi', 'Chennai',
                 'Kolkata', 'Bangalore'], n
            ),
            'product_type': np.random.choice(
                ['cosmetics', 'haircare', 'skincare'], n
            ),
            'shipping_carriers': np.random.choice(
                ['Carrier A', 'Carrier B', 'Carrier C'], n
            ),
            'inspection_results': np.random.choice(
                ['Pass', 'Fail', 'Pending'], n
            ),
        })
        return df

df = load_data()

# ── SESSION STATE ────────────────────────
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 500
if 'npl_ratio' not in st.session_state:
    st.session_state.npl_ratio = 2.1
if 'turn' not in st.session_state:
    st.session_state.turn = 1
if 'decisions' not in st.session_state:
    st.session_state.decisions = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'game_mode' not in st.session_state:
    st.session_state.game_mode = None
if 'scenario_active' not in st.session_state:
    st.session_state.scenario_active = None

# ── SCORING SYSTEM ────────────────────────
def compute_score():
    """
    From document Section B2.1
    1000 point scoring framework
    """
    # Financial Performance (300 pts)
    fp_score = min(300, int(
        300 * (1 - st.session_state.npl_ratio / 5)
    ))

    # Risk Management (250 pts)
    rm_score = min(250, int(
        250 * (1 - len([d for d in st.session_state.decisions
                        if d.get('outcome') == 'bad']) /
               max(1, len(st.session_state.decisions)))
    ))

    # SC Intelligence Use (200 pts)
    sc_score = min(200, int(
        len([d for d in st.session_state.decisions
             if d.get('used_sc_data')]) * 20
    ))

    # Decision Speed (100 pts)
    ds_score = max(0, 100 - st.session_state.turn * 5)

    # Learning Progression (150 pts)
    lp_score = min(150, st.session_state.turn * 10)

    total = fp_score + rm_score + sc_score + ds_score + lp_score

    return {
        'total': total,
        'financial_performance': fp_score,
        'risk_management': rm_score,
        'sc_intelligence': sc_score,
        'decision_speed': ds_score,
        'learning_progression': lp_score
    }

def get_certification(score):
    """From document Section B5"""
    if score >= 900:
        return "💎 MASTER", "Head of Trade Finance (5+ years)"
    elif score >= 750:
        return "🏆 EXPERT", "Trade Finance Team Lead (3-5 years)"
    elif score >= 600:
        return "🥇 SPECIALIST", "Senior Trade Finance Analyst"
    elif score >= 400:
        return "🥈 PRACTITIONER", "Junior Trade Finance Analyst"
    else:
        return "🥉 NOVICE", "Trade Finance Trainee"

# ── SCENARIOS ────────────────────────────
SCENARIOS = {
    "Port Congestion": {
        "description": "Major port hub experiencing 5-15 day delays",
        "sc_impact": "5-15 day delays at major hub",
        "financial_impact": "LC expiry, inventory depletion, freight spike",
        "difficulty": "Medium",
        "icon": "⚓",
        "otif_change": -0.15,
        "delay_days": 10,
        "freight_spike": 0.3,
        "options": [
            "Extend all affected LCs by 30 days",
            "Do nothing - wait and see",
            "Require additional collateral"
        ],
        "correct_option": 0,
        "score_impact": [85, -30, 20]
    },
    "Carrier Bankruptcy": {
        "description": "Major carrier files for bankruptcy, cargo stranded",
        "sc_impact": "Stranded cargo, alternative routing needed",
        "financial_impact": "Cargo insurance claims, SCF disruption",
        "difficulty": "High",
        "icon": "🚢",
        "otif_change": -0.25,
        "delay_days": 20,
        "freight_spike": 0.5,
        "options": [
            "Immediately reroute via alternative carrier",
            "Wait for bankruptcy proceedings",
            "File insurance claims and wait"
        ],
        "correct_option": 0,
        "score_impact": [90, -50, 30]
    },
    "Suez Canal Blockage": {
        "description": "Mega vessel blocks Suez Canal for 6-14 days",
        "sc_impact": "Complete lane shutdown, 400+ vessels blocked",
        "financial_impact": "Massive freight increase, LC expiry risk",
        "difficulty": "Very High",
        "icon": "🏗️",
        "otif_change": -0.30,
        "delay_days": 14,
        "freight_spike": 1.2,
        "options": [
            "Extend LCs + offer working capital support",
            "Reroute via Cape of Good Hope immediately",
            "Do nothing - canal will reopen soon"
        ],
        "correct_option": 0,
        "score_impact": [100, 60, -40]
    },
    "Supplier Quality Failure": {
        "description": "Major supplier reports product recalls",
        "sc_impact": "Production halt, product recalls",
        "financial_impact": "Trade finance default, warranty claims",
        "difficulty": "Medium",
        "icon": "⚠️",
        "otif_change": -0.20,
        "delay_days": 7,
        "freight_spike": 0.1,
        "options": [
            "Reduce exposure limits for affected supplier",
            "Increase monitoring frequency only",
            "Require credit insurance immediately"
        ],
        "correct_option": 2,
        "score_impact": [70, 30, 85]
    },
    "Demand Whiplash": {
        "description": "Rapid demand swing ±40% (Bullwhip Effect)",
        "sc_impact": "Inventory obsolescence or stockout",
        "financial_impact": "Covenant breach risk, working capital stress",
        "difficulty": "High",
        "icon": "📊",
        "otif_change": -0.10,
        "delay_days": 5,
        "freight_spike": 0.2,
        "options": [
            "Adjust working capital facility size upward",
            "Tighten credit limits across portfolio",
            "Increase SCF programme utilisation"
        ],
        "correct_option": 0,
        "score_impact": [80, -20, 50]
    },
    "Commodity Price Shock": {
        "description": "Input cost spike +50-200%",
        "sc_impact": "Input cost spike affecting all suppliers",
        "financial_impact": "Margin compression, collateral revaluation",
        "difficulty": "Medium",
        "icon": "💰",
        "otif_change": -0.05,
        "delay_days": 3,
        "freight_spike": 0.4,
        "options": [
            "Revalue collateral and adjust credit limits",
            "Increase trade finance pricing spread",
            "Do nothing - temporary price shock"
        ],
        "correct_option": 1,
        "score_impact": [60, 80, -30]
    }
}

# ── SIDEBAR ───────────────────────────────
with st.sidebar:
    # st.image("https://via.placeholder.com/200x80?text=LogisChain+Lab",
    #      width=200)

    st.markdown("<h2 style='text-align:center;color:#1f77b4;'>LogisChain Lab</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # Game Mode Selection
    st.subheader("🎮 Game Mode")
    game_mode = st.selectbox(
        "Select Mode",
        ["Trade Finance Portfolio",
         "Supply Chain Finance Pricing",
         "Cargo Insurance Underwriter",
         "Logistics Investment Analyst"]
    )
    st.session_state.game_mode = game_mode

    st.markdown("---")

    # Score Display
    scores = compute_score()
    st.subheader("📊 Your Score")
    st.metric("Total Score", f"{scores['total']}/1000")

    # Certification
    cert, desc = get_certification(scores['total'])
    st.info(f"**{cert}**\n{desc}")

    st.markdown("---")

    # Score Breakdown
    st.subheader("Score Breakdown")
    st.progress(scores['financial_performance']/300,
                text=f"Financial: {scores['financial_performance']}/300")
    st.progress(scores['risk_management']/250,
                text=f"Risk Mgmt: {scores['risk_management']}/250")
    st.progress(scores['sc_intelligence']/200,
                text=f"SC Intel: {scores['sc_intelligence']}/200")
    st.progress(scores['decision_speed']/100,
                text=f"Speed: {scores['decision_speed']}/100")
    st.progress(scores['learning_progression']/150,
                text=f"Learning: {scores['learning_progression']}/150")

    st.markdown("---")

    # Turn Counter
    st.metric("Current Turn", f"Turn {st.session_state.turn}/52")
    st.caption("1 turn = 1 simulated week")

    # Reset Button
    if st.button("🔄 Reset Game"):
        for key in ['score', 'portfolio_value', 'npl_ratio',
                    'turn', 'decisions', 'alerts',
                    'scenario_active']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── MAIN CONTENT ──────────────────────────
st.markdown(
    '<p class="main-header">🚢 LogisChain Lab</p>',
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center'>AI-Powered Trade Finance & Supply Chain Simulation</p>",
    unsafe_allow_html=True
)

# ── TABS ──────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🎯 Scenarios",
    "📈 Portfolio",
    "🤖 AI Intelligence",
    "📋 Leaderboard"
])

# ════════════════════════════════════════
# TAB 1: DASHBOARD
# ════════════════════════════════════════
with tab1:
    st.header(f"🏦 {game_mode}")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${st.session_state.portfolio_value}M",
            delta=f"+${np.random.randint(1,5)}M"
        )
    with col2:
        npl_color = "normal" if st.session_state.npl_ratio < 3 else "inverse"
        st.metric(
            "NPL Ratio",
            f"{st.session_state.npl_ratio:.1f}%",
            delta=f"{np.random.uniform(-0.2, 0.3):.1f}%",
            delta_color=npl_color
        )
    with col3:
        otif = df['otif'].mean()
        st.metric(
            "Portfolio OTIF",
            f"{otif:.1%}",
            delta=f"{np.random.uniform(-0.02, 0.02):.1%}"
        )
    with col4:
        st.metric(
            "Active Alerts",
            len(st.session_state.alerts),
            delta=np.random.randint(0, 3)
        )

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Supplier Risk Dashboard")

        # SC-PD by supplier
        if 'sc_pd' in df.columns:
            supplier_pd = df.groupby('supplier_name')['sc_pd'].mean()
        else:
            supplier_pd = pd.Series({
                'Supplier 1': 0.043,
                'Supplier 2': 0.045,
                'Supplier 3': 0.050,
                'Supplier 4': 0.038,
                'Supplier 5': 0.058,
            })

        colors = ['red' if x > 0.045 else 'green'
                  for x in supplier_pd.values]

        fig = go.Figure(go.Bar(
            x=supplier_pd.index,
            y=supplier_pd.values * 100,
            marker_color=colors,
            text=[f'{x*100:.2f}%' for x in supplier_pd.values],
            textposition='outside'
        ))
        fig.add_hline(y=2.8, line_dash="dash",
                      line_color="orange",
                      annotation_text="Baseline PD 2.8%")
        fig.update_layout(
            title="SC-Adjusted PD by Supplier",
            yaxis_title="Probability of Default (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 OTIF Trend")

        # Simulate OTIF over time
        turns = list(range(1, st.session_state.turn + 2))
        otif_trend = [0.88 + np.random.uniform(-0.05, 0.05)
                      for _ in turns]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=turns, y=otif_trend,
            mode='lines+markers',
            name='OTIF Rate',
            line=dict(color='blue')
        ))
        fig2.add_hline(y=0.85, line_dash="dash",
                       line_color="red",
                       annotation_text="85% Threshold")
        fig2.update_layout(
            title="OTIF Rate Over Time",
            xaxis_title="Turn (Week)",
            yaxis_title="OTIF Rate",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Supply Chain Alerts
    st.subheader("🚨 Supply Chain Intelligence Alerts")

    # Generate dynamic alerts
    alerts = []
    otif_val = df['otif'].mean()
    if otif_val < 0.85:
        alerts.append({
            'type': '🔴 CRITICAL',
            'message': f'OTIF Rate {otif_val:.1%} below 85% threshold',
            'action': 'Credit downgrade signal for SCF portfolio',
            'metric': 'OTIF'
        })

    avg_delay = df['shipment_delay_days'].mean()
    if avg_delay > 5:
        alerts.append({
            'type': '🟡 WARNING',
            'message': f'Avg delay {avg_delay:.1f} days exceeds 5-day threshold',
            'action': 'Extend LC validity, increase reserve allocation',
            'metric': 'Port Congestion'
        })

    if 'sc_pd' in df.columns:
        high_risk = (df['sc_pd'] > 0.045).sum()
        if high_risk > 0:
            alerts.append({
                'type': '🟡 WARNING',
                'message': f'{high_risk} suppliers with SC-PD above threshold',
                'action': 'Reduce SCF exposure, tighten credit limits',
                'metric': 'Credit Risk'
            })

    if len(alerts) == 0:
        st.success("✅ No critical alerts - portfolio healthy!")
    else:
        for alert in alerts:
            with st.expander(
                f"{alert['type']}: {alert['metric']}"
            ):
                st.write(f"**Signal:** {alert['message']}")
                st.write(f"**Action:** {alert['action']}")
                if st.button(
                    f"✅ Acknowledge & Act",
                    key=f"alert_{alert['metric']}"
                ):
                    st.session_state.decisions.append({
                        'turn': st.session_state.turn,
                        'action': alert['action'],
                        'used_sc_data': True,
                        'outcome': 'good'
                    })
                    st.success("Decision recorded! +SC Intelligence points")

# ════════════════════════════════════════
# TAB 2: SCENARIOS
# ════════════════════════════════════════
# ... (rest of the file unchanged) ...

# ── FOOTER ────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>LogisChain Lab | "
    "Powered by LogisChain AI | "
    "Dharya Agarwal</p>",
    unsafe_allow_html=True
)
