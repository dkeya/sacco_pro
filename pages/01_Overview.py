# pages/01_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sacco_core.config import ConfigManager
from sacco_core.rbac import RBACManager
from sacco_core.audit import AuditLogger

# Page configuration
st.set_page_config(
    page_title="SACCO Overview",
    page_icon="📊",
    layout="wide"
)

class OverviewPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        
        # Check access
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        """Check if user has access to this page"""
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "01_Overview.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        # Log page access
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "overview_page"
        )
        
        return True
    
    def render_kpi_cards(self):
        """Render KPI cards at the top"""
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Loan Portfolio", 
                "KES 245.8M", 
                "3.2%",
                help="Total outstanding loan portfolio"
            )
        
        with col2:
            st.metric(
                "PAR > 30 Days", 
                "4.2%", 
                "-0.5%",
                delta_color="inverse",
                help="Portfolio at Risk over 30 days"
            )
        
        with col3:
            st.metric(
                "Liquidity Ratio", 
                "18.5%", 
                "1.2%",
                help="Cash and equivalents to total deposits"
            )
        
        with col4:
            st.metric(
                "Membership", 
                "1,250", 
                "5.0%",
                help="Total active members"
            )
        
        with col5:
            st.metric(
                "Capital Adequacy", 
                "22.1%", 
                "0.8%",
                help="Capital to risk-weighted assets"
            )
    
    def render_portfolio_trends(self):
        """Render portfolio trend charts"""
        st.subheader("Portfolio Trends")
        
        # Generate sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        portfolio_data = pd.DataFrame({
            'date': dates,
            'total_loans': np.random.normal(240, 10, len(dates)).cumsum(),
            'par_30': np.random.uniform(3.5, 5.5, len(dates)),
            'liquidity_ratio': np.random.uniform(16, 20, len(dates)),
            'members': np.random.normal(1200, 50, len(dates)).cumsum()
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan portfolio trend
            fig = px.line(
                portfolio_data, 
                x='date', 
                y='total_loans',
                title="Loan Portfolio Growth",
                labels={'total_loans': 'Portfolio (KES M)', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # PAR trend
            fig = px.line(
                portfolio_data, 
                x='date', 
                y='par_30',
                title="PAR 30+ Days Trend",
                labels={'par_30': 'PAR %', 'date': 'Date'}
            )
            fig.add_hline(
                y=self.config.limits.par30_trigger_max * 100, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Threshold"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_heatmap(self):
        """Render risk heatmap"""
        st.subheader("Risk Exposure Heatmap")
        
        # Sample risk data by product
        risk_data = pd.DataFrame({
            'Product': ['Personal Loans', 'Business Loans', 'Asset Finance', 'Emergency', 'School Fees'],
            'Exposure (KES M)': [85.2, 92.5, 45.3, 12.8, 9.9],
            'PAR %': [3.2, 5.8, 2.1, 8.5, 1.2],
            'Growth %': [12.5, 8.2, 15.3, 25.1, 18.7]
        })
        
        fig = px.treemap(
            risk_data,
            path=['Product'],
            values='Exposure (KES M)',
            color='PAR %',
            color_continuous_scale='RdYlGn_r',
            title="Portfolio Exposure by Product (Color = PAR %)",
            hover_data=['Growth %']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_alerts(self):
        """Render recent alerts panel"""
        st.subheader("Recent Alerts & Notifications")
        
        alerts_data = {
            'Date': ['2024-01-15 14:30', '2024-01-14 09:15', '2024-01-13 16:45', '2024-01-12 11:20'],
            'Alert Type': ['PAR Warning', 'Concentration', 'Liquidity', 'Employer Limit'],
            'Severity': ['Medium', 'High', 'Low', 'Medium'],
            'Description': [
                'PAR30 approaching threshold at 4.2%',
                'Top employer exposure at 28% of portfolio',
                'Liquidity ratio improved to 18.5%',
                'Employer XYZ exposure exceeds 25% limit'
            ],
            'Status': ['Active', 'Active', 'Resolved', 'Active']
        }
        
        alerts_df = pd.DataFrame(alerts_data)
        
        # Color code by severity
        def color_severity(severity):
            if severity == 'High':
                return 'color: red; font-weight: bold'
            elif severity == 'Medium':
                return 'color: orange'
            else:
                return 'color: green'
        
        styled_df = alerts_df.style.applymap(
            color_severity, 
            subset=['Severity']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def run(self):
        """Run the overview page"""
        st.title("SACCO Performance Overview")
        
        # Render all components
        self.render_kpi_cards()
        st.markdown("---")
        self.render_portfolio_trends()
        st.markdown("---")
        self.render_risk_heatmap()
        st.markdown("---")
        self.render_recent_alerts()

# Run the page
if __name__ == "__main__":
    page = OverviewPage()
    page.run()