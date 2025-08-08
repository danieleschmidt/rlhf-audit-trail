"""Streamlit-based dashboard for RLHF audit trail monitoring."""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .core import AuditableRLHF
from .config import PrivacyConfig, ComplianceConfig
from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)


class DashboardData:
    """Mock data generator for dashboard demonstration."""
    
    @staticmethod
    def generate_training_metrics(days: int = 7) -> pd.DataFrame:
        """Generate mock training metrics data."""
        import random
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'loss': 2.5 * np.exp(-i/100) + random.uniform(-0.1, 0.1),
                'reward': 0.5 + 0.4 * (1 - np.exp(-i/50)) + random.uniform(-0.05, 0.05),
                'learning_rate': 1e-4 * np.exp(-i/200),
                'gradient_norm': 1.0 + random.uniform(-0.2, 0.2),
                'privacy_epsilon': min(10.0, i * 0.01 + random.uniform(-0.001, 0.001))
            })
            
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_compliance_data() -> Dict[str, Any]:
        """Generate mock compliance data."""
        return {
            'eu_ai_act': {
                'score': 95.2,
                'status': 'Compliant',
                'last_check': datetime.now(),
                'requirements': {
                    'risk_management': {'status': 'Pass', 'score': 98},
                    'data_governance': {'status': 'Pass', 'score': 94},
                    'transparency': {'status': 'Pass', 'score': 96},
                    'human_oversight': {'status': 'Pass', 'score': 92},
                    'accuracy_robustness': {'status': 'Pass', 'score': 97}
                }
            },
            'nist_framework': {
                'score': 87.5,
                'status': 'Largely Compliant',
                'last_check': datetime.now(),
                'requirements': {
                    'govern': {'status': 'Pass', 'score': 90},
                    'map': {'status': 'Pass', 'score': 88},
                    'measure': {'status': 'Warning', 'score': 85},
                    'manage': {'status': 'Pass', 'score': 89}
                }
            }
        }
    
    @staticmethod
    def generate_privacy_data() -> Dict[str, Any]:
        """Generate mock privacy data."""
        return {
            'total_epsilon': 8.5,
            'epsilon_budget': 10.0,
            'remaining_budget': 1.5,
            'annotator_count': 45,
            'privacy_violations': 0,
            'anonymization_rate': 100.0,
            'budget_usage': [
                {'date': datetime.now() - timedelta(days=6), 'epsilon': 1.2},
                {'date': datetime.now() - timedelta(days=5), 'epsilon': 2.1},
                {'date': datetime.now() - timedelta(days=4), 'epsilon': 3.3},
                {'date': datetime.now() - timedelta(days=3), 'epsilon': 4.8},
                {'date': datetime.now() - timedelta(days=2), 'epsilon': 6.2},
                {'date': datetime.now() - timedelta(days=1), 'epsilon': 7.5},
                {'date': datetime.now(), 'epsilon': 8.5}
            ]
        }
    
    @staticmethod
    def generate_audit_logs() -> List[Dict[str, Any]]:
        """Generate mock audit log data."""
        import random
        
        event_types = ['annotation', 'policy_update', 'checkpoint', 'privacy_check', 'compliance_check']
        logs = []
        
        for i in range(100):
            logs.append({
                'timestamp': datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                'event_type': random.choice(event_types),
                'session_id': f"session_{random.randint(1, 5)}",
                'event_data': {
                    'batch_size': random.randint(16, 64),
                    'loss': round(random.uniform(0.5, 2.5), 3),
                    'reward': round(random.uniform(0.2, 0.9), 3)
                },
                'verified': True,
                'signature_valid': True
            })
            
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)


def create_training_metrics_chart(df: pd.DataFrame) -> go.Figure:
    """Create training metrics chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Over Time', 'Reward Over Time', 
                       'Learning Rate', 'Privacy Epsilon Usage'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['loss'], name='Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Reward
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['reward'], name='Reward', line=dict(color='green')),
        row=1, col=2
    )
    
    # Learning Rate
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['learning_rate'], name='LR', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Privacy Epsilon
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['privacy_epsilon'], name='Epsilon', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="RLHF Training Metrics",
        showlegend=False
    )
    
    return fig


def create_compliance_chart(compliance_data: Dict[str, Any]) -> go.Figure:
    """Create compliance status chart."""
    frameworks = list(compliance_data.keys())
    scores = [compliance_data[fw]['score'] for fw in frameworks]
    colors = ['green' if score >= 90 else 'orange' if score >= 80 else 'red' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(x=frameworks, y=scores, marker_color=colors, text=scores, textposition='auto')
    ])
    
    fig.update_layout(
        title="Compliance Framework Scores",
        xaxis_title="Framework",
        yaxis_title="Compliance Score (%)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_privacy_budget_chart(privacy_data: Dict[str, Any]) -> go.Figure:
    """Create privacy budget usage chart."""
    budget_df = pd.DataFrame(privacy_data['budget_usage'])
    
    fig = go.Figure()
    
    # Budget usage over time
    fig.add_trace(go.Scatter(
        x=budget_df['date'], 
        y=budget_df['epsilon'],
        mode='lines+markers',
        name='Epsilon Used',
        line=dict(color='blue')
    ))
    
    # Budget limit line
    fig.add_hline(
        y=privacy_data['epsilon_budget'], 
        line_dash="dash", 
        line_color="red",
        annotation_text="Budget Limit"
    )
    
    fig.update_layout(
        title="Privacy Budget Usage Over Time",
        xaxis_title="Date",
        yaxis_title="Epsilon",
        yaxis=dict(range=[0, privacy_data['epsilon_budget'] * 1.1])
    )
    
    return fig


def main_dashboard():
    """Main dashboard application."""
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit not available. Install with: pip install streamlit plotly")
        return
        
    st.set_page_config(
        page_title="RLHF Audit Trail Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RLHF Audit Trail Dashboard")
    st.markdown("Real-time monitoring for verifiable RLHF with EU AI Act compliance")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Overview", 
        "Training Metrics", 
        "Compliance Status", 
        "Privacy Monitor", 
        "Audit Logs",
        "Model Cards"
    ])
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Session selector
    st.sidebar.subheader("Training Session")
    sessions = ["session_1", "session_2", "session_3"]
    selected_session = st.sidebar.selectbox("Select Session", sessions)
    
    # Page routing
    if page == "Overview":
        show_overview_page()
    elif page == "Training Metrics":
        show_training_metrics_page()
    elif page == "Compliance Status":
        show_compliance_page()
    elif page == "Privacy Monitor":
        show_privacy_page()
    elif page == "Audit Logs":
        show_audit_logs_page()
    elif page == "Model Cards":
        show_model_cards_page()


def show_overview_page():
    """Show overview dashboard page."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Sessions",
            value="3",
            delta="1 new today"
        )
    
    with col2:
        st.metric(
            label="Compliance Score",
            value="95.2%",
            delta="2.1%"
        )
    
    with col3:
        st.metric(
            label="Privacy Budget Used",
            value="85%",
            delta="-5%"
        )
    
    with col4:
        st.metric(
            label="Audit Events",
            value="1,247",
            delta="89 today"
        )
    
    # System status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🟢 Audit Trail System: Online")
        st.success("🟢 Privacy Engine: Active")
        st.success("🟢 Compliance Monitor: Running")
        
    with col2:
        st.info("📊 Last Health Check: 2 minutes ago")
        st.info("🔒 Last Security Scan: 1 hour ago")
        st.info("📋 Last Compliance Check: 30 minutes ago")
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_logs = DashboardData.generate_audit_logs()[:10]
    
    for log in recent_logs:
        with st.expander(f"{log['event_type'].title()} - {log['timestamp'].strftime('%H:%M:%S')}"):
            st.json(log['event_data'])


def show_training_metrics_page():
    """Show training metrics page."""
    st.header("Training Metrics")
    
    # Generate and display metrics
    df = DashboardData.generate_training_metrics()
    fig = create_training_metrics_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Current metrics table
    st.subheader("Current Metrics")
    latest_metrics = df.iloc[-1]
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Current Loss", f"{latest_metrics['loss']:.3f}")
        st.metric("Current Reward", f"{latest_metrics['reward']:.3f}")
        
    with metrics_col2:
        st.metric("Learning Rate", f"{latest_metrics['learning_rate']:.2e}")
        st.metric("Gradient Norm", f"{latest_metrics['gradient_norm']:.3f}")
    
    # Detailed metrics table
    st.subheader("Detailed Metrics (Last 24 Hours)")
    recent_df = df.tail(24)
    st.dataframe(recent_df, use_container_width=True)


def show_compliance_page():
    """Show compliance status page."""
    st.header("Compliance Status")
    
    compliance_data = DashboardData.generate_compliance_data()
    
    # Overall compliance chart
    fig = create_compliance_chart(compliance_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed compliance status
    for framework, data in compliance_data.items():
        st.subheader(f"{framework.upper().replace('_', ' ')} Compliance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "success" if data['status'] == 'Compliant' else "warning"
            getattr(st, status_color)(f"Status: {data['status']}")
            
        with col2:
            st.metric("Score", f"{data['score']:.1f}%")
            
        with col3:
            st.info(f"Last Check: {data['last_check'].strftime('%Y-%m-%d %H:%M')}")
        
        # Requirements breakdown
        st.write("**Requirements Breakdown:**")
        req_df = pd.DataFrame([
            {"Requirement": req, "Status": info['status'], "Score": info['score']}
            for req, info in data['requirements'].items()
        ])
        st.dataframe(req_df, use_container_width=True)


def show_privacy_page():
    """Show privacy monitoring page."""
    st.header("Privacy Monitor")
    
    privacy_data = DashboardData.generate_privacy_data()
    
    # Privacy budget overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Privacy Budget Used",
            f"{privacy_data['total_epsilon']:.1f} / {privacy_data['epsilon_budget']:.1f}",
            f"{(privacy_data['total_epsilon']/privacy_data['epsilon_budget']*100):.1f}%"
        )
    
    with col2:
        st.metric(
            "Remaining Budget",
            f"{privacy_data['remaining_budget']:.1f}",
            f"{(privacy_data['remaining_budget']/privacy_data['epsilon_budget']*100):.1f}%"
        )
    
    with col3:
        st.metric(
            "Annotator Count",
            privacy_data['annotator_count'],
            "0 violations"
        )
    
    # Privacy budget usage chart
    fig = create_privacy_budget_chart(privacy_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Privacy settings
    st.subheader("Privacy Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info("**Differential Privacy Settings**")
        st.write(f"• Epsilon Budget: {privacy_data['epsilon_budget']}")
        st.write(f"• Delta: 1e-5")
        st.write(f"• Clipping Norm: 1.0")
        
    with config_col2:
        st.success("**Privacy Status**")
        st.write(f"• Anonymization Rate: {privacy_data['anonymization_rate']:.1f}%")
        st.write(f"• Privacy Violations: {privacy_data['privacy_violations']}")
        st.write("• Noise Mechanism: Gaussian")


def show_audit_logs_page():
    """Show audit logs page."""
    st.header("Audit Logs")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        event_filter = st.selectbox("Event Type", 
                                   ["All", "annotation", "policy_update", "checkpoint", "compliance_check"])
    
    with col2:
        time_filter = st.selectbox("Time Range", 
                                  ["Last Hour", "Last Day", "Last Week", "All"])
    
    with col3:
        session_filter = st.selectbox("Session", 
                                     ["All", "session_1", "session_2", "session_3"])
    
    # Audit logs table
    logs = DashboardData.generate_audit_logs()
    
    # Apply filters (mock filtering)
    if event_filter != "All":
        logs = [log for log in logs if log['event_type'] == event_filter]
    
    # Display logs
    logs_df = pd.DataFrame([{
        'Timestamp': log['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'Event Type': log['event_type'],
        'Session': log['session_id'],
        'Verified': '✅' if log['verified'] else '❌',
        'Signature Valid': '✅' if log['signature_valid'] else '❌'
    } for log in logs[:50]])
    
    st.dataframe(logs_df, use_container_width=True)
    
    # Log details
    if st.checkbox("Show detailed event data"):
        selected_log_idx = st.selectbox("Select log entry", range(min(10, len(logs))))
        if selected_log_idx is not None and selected_log_idx < len(logs):
            st.json(logs[selected_log_idx])


def show_model_cards_page():
    """Show model cards page."""
    st.header("Model Cards")
    
    st.write("**Available Model Cards:**")
    
    # Mock model card data
    model_cards = [
        {
            "name": "RLHF Safety Model v2.1",
            "created": "2025-01-15",
            "framework": "EU AI Act",
            "status": "Compliant"
        },
        {
            "name": "Helpfulness Optimizer v1.3", 
            "created": "2025-01-10",
            "framework": "NIST Framework",
            "status": "Under Review"
        }
    ]
    
    for card in model_cards:
        with st.expander(f"📋 {card['name']} ({card['created']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Framework:** {card['framework']}")
                st.write(f"**Status:** {card['status']}")
                st.write(f"**Created:** {card['created']}")
                
            with col2:
                if st.button(f"Generate Report", key=f"report_{card['name']}"):
                    st.success("Model card report generated successfully!")
                if st.button(f"Download PDF", key=f"pdf_{card['name']}"):
                    st.info("PDF download would start here")
    
    # Model card generator
    st.subheader("Generate New Model Card")
    
    with st.form("model_card_form"):
        model_name = st.text_input("Model Name", "New RLHF Model")
        include_provenance = st.checkbox("Include Provenance", True)
        include_privacy = st.checkbox("Include Privacy Analysis", True)
        framework = st.selectbox("Compliance Framework", ["EU AI Act", "NIST Framework", "Both"])
        
        if st.form_submit_button("Generate Model Card"):
            with st.spinner("Generating model card..."):
                time.sleep(2)  # Simulate processing
                st.success(f"Model card for '{model_name}' generated successfully!")


def run_dashboard(host: str = "localhost", port: int = 8501, debug: bool = False):
    """Run the Streamlit dashboard."""
    if not STREAMLIT_AVAILABLE:
        raise RuntimeError("Streamlit not available. Install with: pip install streamlit plotly")
    
    # This function would be called from the command line
    # Streamlit handles the actual server startup
    main_dashboard()


if __name__ == "__main__":
    run_dashboard()