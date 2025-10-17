# pages/03_Liquidity_ALM.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sacco_core.config import ConfigManager
from sacco_core.rbac import RBACManager
from sacco_core.audit import AuditLogger
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Liquidity & ALM",
    page_icon="💧",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class LiquidityALMPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "03_Liquidity_ALM.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "liquidity_alm_page"
        )
        return True
    
    def render_liquidity_dashboard(self):
        """Render liquidity dashboard"""
        st.subheader("Liquidity Position")
        
        # Generate sample liquidity data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        liquidity_data = pd.DataFrame({
            'date': dates,
            'cash_equivalents': np.random.uniform(40, 60, len(dates)),
            'investments': np.random.uniform(20, 40, len(dates)),
            'total_deposits': np.random.uniform(220, 280, len(dates)),
            'loan_disbursements': np.random.uniform(15, 25, len(dates))
        })
        
        liquidity_data['liquidity_ratio'] = (
            liquidity_data['cash_equivalents'] / liquidity_data['total_deposits'] * 100
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_ratio = liquidity_data['liquidity_ratio'].iloc[-1]
            st.metric(
                "Liquidity Ratio", 
                f"{latest_ratio:.1f}%",
                delta=f"{(latest_ratio - self.config.limits.liquidity_ratio_min * 100):+.1f}% vs min",
                delta_color="inverse" if latest_ratio < self.config.limits.liquidity_ratio_min * 100 else "normal"
            )
        
        with col2:
            st.metric("Cash & Equivalents", f"KES {liquidity_data['cash_equivalents'].iloc[-1]:.1f}M")
        
        with col3:
            st.metric("Total Deposits", f"KES {liquidity_data['total_deposits'].iloc[-1]:.1f}M")
        
        with col4:
            st.metric("Loan-to-Deposit Ratio", "78.5%")
        
        # Liquidity trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=liquidity_data['date'],
            y=liquidity_data['liquidity_ratio'],
            name='Liquidity Ratio',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_hline(
            y=self.config.limits.liquidity_ratio_min * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Minimum Threshold"
        )
        
        fig.update_layout(
            title="Liquidity Ratio Trend",
            xaxis_title="Date",
            yaxis_title="Liquidity Ratio (%)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alm_gap_analysis(self):
        """Render ALM gap analysis"""
        st.subheader("Asset-Liability Maturity Gap Analysis")
        
        # Sample maturity buckets
        buckets = ['O/N', '1-7D', '8-30D', '31-90D', '91-180D', '181-365D', '1-3Y', '3Y+']
        
        gap_data = pd.DataFrame({
            'Bucket': buckets,
            'Assets': [15, 25, 45, 60, 40, 35, 20, 10],
            'Liabilities': [20, 35, 55, 50, 30, 25, 15, 5],
            'Rate_Sensitive': [80, 75, 70, 65, 60, 55, 40, 20]
        })
        
        gap_data['Gap'] = gap_data['Assets'] - gap_data['Liabilities']
        gap_data['Cumulative_Gap'] = gap_data['Gap'].cumsum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                gap_data,
                x='Bucket',
                y=['Assets', 'Liabilities'],
                title="Maturity Buckets - Assets vs Liabilities",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=gap_data['Bucket'],
                y=gap_data['Gap'],
                name='Periodic Gap',
                marker_color=['red' if x < 0 else 'green' for x in gap_data['Gap']]
            ))
            
            fig.add_trace(go.Scatter(
                x=gap_data['Bucket'],
                y=gap_data['Cumulative_Gap'],
                name='Cumulative Gap',
                line=dict(color='orange', width=3)
            ))
            
            fig.update_layout(
                title="Maturity Gap Analysis",
                xaxis_title="Maturity Bucket",
                yaxis_title="Gap (KES M)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the liquidity ALM page"""
        st.title("💧 Liquidity & Asset-Liability Management")
        
        self.render_liquidity_dashboard()
        st.markdown("---")
        self.render_alm_gap_analysis()

if __name__ == "__main__":
    page = LiquidityALMPage()
    page.run()