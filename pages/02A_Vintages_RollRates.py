# pages/02A_Vintages_RollRates.py
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

st.set_page_config(
    page_title="Vintages & Roll Rates",
    page_icon="📊",
    layout="wide"
)

class VintagesRollRatesPage:
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
            "02A_Vintages_RollRates.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "vintages_roll_rates_page"
        )
        return True
    
    def run(self):
        st.title("Vintages & Roll Rates Analysis")
        st.info("This module will implement TRUE vintages & roll-rates using monthly state snapshots")
        
        # Placeholder for vintages analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cohort Vintage Analysis")
            st.write("Monthly cohorts with MoB curves will be displayed here")
            
            # Sample vintage data
            months_on_books = list(range(1, 13))
            default_rates = [0.5, 1.2, 2.1, 3.0, 3.8, 4.5, 5.1, 5.6, 6.0, 6.3, 6.5, 6.6]
            
            fig = px.line(
                x=months_on_books, 
                y=default_rates,
                title="Cumulative Default Rate by Months on Books",
                labels={'x': 'Months on Books', 'y': 'Cumulative Default Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Roll Rate Analysis")
            st.write("T→T+1 transition matrices will be displayed here")
            
            # Sample roll rate matrix
            states = ['Current', '1-30 DPD', '31-60 DPD', '61-90 DPD', '90+ DPD']
            transition_matrix = np.array([
                [0.92, 0.06, 0.01, 0.005, 0.005],
                [0.35, 0.45, 0.12, 0.05, 0.03],
                [0.15, 0.25, 0.35, 0.15, 0.10],
                [0.08, 0.12, 0.20, 0.40, 0.20],
                [0.02, 0.05, 0.08, 0.15, 0.70]
            ])
            
            fig = px.imshow(
                transition_matrix,
                x=states,
                y=states,
                title="Roll Rate Transition Matrix (T → T+1)",
                color_continuous_scale='Blues',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # File upload for monthly states
        st.subheader("Upload Monthly Loan States")
        uploaded_file = st.file_uploader(
            "Upload loan_states_monthly.csv", 
            type=['csv'],
            help="File should contain monthly snapshots of loan states"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} records")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    page = VintagesRollRatesPage()
    page.run()