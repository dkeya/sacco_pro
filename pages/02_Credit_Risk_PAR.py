# pages/02_Credit_Risk_PAR.py
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
    page_title="Credit Risk & PAR Analysis",
    page_icon="📈",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class CreditRiskPage:
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
            "02_Credit_Risk_PAR.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "credit_risk_page"
        )
        return True
    
    def generate_sample_par_data(self):
        """Generate sample PAR ladder data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        
        data = []
        for date in dates:
            data.append({
                'month_end': date,
                'current': np.random.uniform(180, 220),
                'dpd1_30': np.random.uniform(8, 15),
                'dpd31_60': np.random.uniform(3, 8),
                'dpd61_90': np.random.uniform(1, 4),
                'dpd91_180': np.random.uniform(0.5, 2),
                'dpd180_plus': np.random.uniform(0.1, 1),
                'written_off': np.random.uniform(2, 5)
            })
        
        return pd.DataFrame(data)
    
    def render_par_ladder(self):
        """Render PAR ladder analysis"""
        st.subheader("PAR Ladder Analysis")
        
        par_data = self.generate_sample_par_data()
        
        # Calculate percentages
        total_loans = par_data[[
            'current', 'dpd1_30', 'dpd31_60', 'dpd61_90', 
            'dpd91_180', 'dpd180_plus', 'written_off'
        ]].sum(axis=1)
        
        for col in ['current', 'dpd1_30', 'dpd31_60', 'dpd61_90', 'dpd91_180', 'dpd180_plus', 'written_off']:
            par_data[f'{col}_pct'] = (par_data[col] / total_loans) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PAR trend chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=par_data['month_end'], 
                y=par_data['dpd1_30_pct'],
                name='1-30 DPD',
                stackgroup='one'
            ))
            fig.add_trace(go.Scatter(
                x=par_data['month_end'], 
                y=par_data['dpd31_60_pct'],
                name='31-60 DPD',
                stackgroup='one'
            ))
            fig.add_trace(go.Scatter(
                x=par_data['month_end'], 
                y=par_data['dpd61_90_pct'],
                name='61-90 DPD',
                stackgroup='one'
            ))
            fig.add_trace(go.Scatter(
                x=par_data['month_end'], 
                y=par_data['dpd91_180_pct'],
                name='91-180 DPD',
                stackgroup='one'
            ))
            fig.add_trace(go.Scatter(
                x=par_data['month_end'], 
                y=par_data['dpd180_plus_pct'],
                name='180+ DPD',
                stackgroup='one'
            ))
            
            fig.update_layout(
                title="PAR Ladder Trend (% of Portfolio)",
                xaxis_title="Month",
                yaxis_title="% of Portfolio",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Current PAR breakdown
            latest = par_data.iloc[-1]
            
            par_categories = ['Current', '1-30 DPD', '31-60 DPD', '61-90 DPD', 
                             '91-180 DPD', '180+ DPD', 'Written Off']
            par_values = [
                latest['current_pct'],
                latest['dpd1_30_pct'], 
                latest['dpd31_60_pct'],
                latest['dpd61_90_pct'],
                latest['dpd91_180_pct'],
                latest['dpd180_plus_pct'],
                latest.get('written_off_pct', 0)
            ]
            
            fig = px.pie(
                values=par_values, 
                names=par_categories,
                title="Current Portfolio Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # PAR summary metrics
        st.subheader("PAR Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            par30 = (latest['dpd31_60_pct'] + latest['dpd61_90_pct'] + 
                    latest['dpd91_180_pct'] + latest['dpd180_plus_pct'])
            st.metric("PAR 30+", f"{par30:.1f}%", 
                     delta=f"{(par30 - self.config.limits.par30_trigger_max * 100):+.1f}% vs threshold",
                     delta_color="inverse" if par30 > self.config.limits.par30_trigger_max * 100 else "normal")
        
        with col2:
            par90 = latest['dpd91_180_pct'] + latest['dpd180_plus_pct']
            st.metric("PAR 90+", f"{par90:.1f}%")
        
        with col3:
            total_par = 100 - latest['current_pct']
            st.metric("Total PAR", f"{total_par:.1f}%")
        
        with col4:
            npl_ratio = latest['dpd91_180_pct'] + latest['dpd180_plus_pct'] + latest.get('written_off_pct', 0)
            st.metric("NPL Ratio", f"{npl_ratio:.1f}%")
    
    def render_risk_concentration(self):
        """Render risk concentration analysis"""
        st.subheader("Risk Concentration Analysis")
        
        # Sample concentration data
        concentration_data = pd.DataFrame({
            'Segment': ['Employer A', 'Employer B', 'Employer C', 'Employer D', 'Others'],
            'Exposure (KES M)': [68.5, 45.2, 32.1, 28.7, 71.3],
            'PAR30 %': [2.1, 5.8, 8.2, 3.5, 4.1],
            'Members': [185, 120, 85, 75, 785]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                concentration_data,
                x='Segment',
                y='Exposure (KES M)',
                color='PAR30 %',
                title="Exposure by Employer Segment",
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Employer concentration risk
            concentration_data['Share %'] = (
                concentration_data['Exposure (KES M)'] / 
                concentration_data['Exposure (KES M)'].sum() * 100
            )
            
            fig = px.pie(
                concentration_data,
                values='Exposure (KES M)',
                names='Segment',
                title="Employer Concentration"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Check for limit breaches
            max_share = concentration_data['Share %'].max()
            if max_share > self.config.limits.single_employer_share_max * 100:
                st.warning(
                    f"⚠️ Maximum employer concentration ({max_share:.1f}%) "
                    f"exceeds limit ({self.config.limits.single_employer_share_max * 100:.1f}%)"
                )
    
    def run(self):
        """Run the credit risk page"""
        st.title("🎯 Credit Risk & PAR Analysis")
        
        self.render_par_ladder()
        st.markdown("---")
        self.render_risk_concentration()

if __name__ == "__main__":
    page = CreditRiskPage()
    page.run()