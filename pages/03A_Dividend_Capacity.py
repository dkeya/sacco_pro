# pages/03A_Dividend_Capacity.py
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
from sacco_core.analytics.dividend import DividendCalculator
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Dividend Capacity Analysis",
    page_icon="💰",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class DividendCapacityPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.dividend_calc = DividendCalculator()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "03A_Dividend_Capacity.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "dividend_capacity_page"
        )
        return True
    
    def render_dividend_calculator(self):
        """Render the main dividend capacity calculator"""
        st.subheader("Dividend Capacity Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Input parameters
            st.markdown("#### Financial Parameters")
            total_shares = st.number_input(
                "Total Share Capital (KES)", 
                min_value=0.0,
                value=15000000.0,
                step=100000.0,
                help="Total member share capital"
            )
            
            net_surplus = st.number_input(
                "Net Surplus for the Year (KES)",
                min_value=0.0,
                value=3500000.0,
                step=100000.0,
                help="Net surplus after all expenses and provisions"
            )
            
            statutory_reserve = st.number_input(
                "Statutory Reserve Transfer (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Percentage of surplus to transfer to statutory reserve"
            )
        
        with col2:
            st.markdown("#### Policy Gates")
            current_liquidity = st.slider(
                "Current Liquidity Ratio (%)",
                min_value=0.0,
                max_value=50.0,
                value=18.5,
                step=0.1,
                help="Current liquidity ratio"
            )
            
            current_par30 = st.slider(
                "Current PAR 30+ (%)",
                min_value=0.0,
                max_value=30.0,
                value=4.2,
                step=0.1,
                help="Current portfolio at risk over 30 days"
            )
            
            ecl_provision = st.number_input(
                "ECL Provision Required (KES)",
                min_value=0.0,
                value=850000.0,
                step=10000.0,
                help="Expected credit loss provision required"
            )
        
        with col3:
            st.markdown("#### Dividend Policy")
            st.info(f"**Maximum Dividend**: {self.config.dividend.max_recommendation * 100}%")
            st.info(f"**PAR30 Gate**: {self.config.dividend.par30_gate * 100}%")
            st.info(f"**Liquidity Gate**: {self.config.dividend.liquidity_gate * 100}%")
            
            # Calculate dividend capacity
            if st.button("Calculate Dividend Capacity", type="primary"):
                result = self.dividend_calc.calculate_dividend_capacity(
                    total_shares=total_shares,
                    net_surplus=net_surplus,
                    statutory_reserve_pct=statutory_reserve,
                    current_liquidity=current_liquidity / 100,
                    current_par30=current_par30 / 100,
                    ecl_provision=ecl_provision,
                    config=self.config
                )
                
                st.session_state.dividend_result = result
        
        # Display results
        if 'dividend_result' in st.session_state:
            self.render_dividend_results(st.session_state.dividend_result)
    
    def render_dividend_results(self, result):
        """Render dividend calculation results"""
        st.markdown("---")
        st.subheader("Dividend Capacity Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Recommended Dividend Rate",
                f"{result['recommended_dividend_pct']:.2f}%",
                delta=f"{(result['recommended_dividend_pct'] - result['previous_dividend_pct']):+.2f}%" if result['previous_dividend_pct'] else None
            )
        
        with col2:
            st.metric(
                "Dividend Amount",
                f"KES {result['dividend_amount']:,.0f}",
                help="Total dividend to be paid out"
            )
        
        with col3:
            status_color = "🟢" if result['policy_gates_passed'] else "🔴"
            st.metric(
                "Policy Status",
                f"{status_color} {'PASSED' if result['policy_gates_passed'] else 'FAILED'}",
                help="All policy gates must pass for dividend declaration"
            )
        
        with col4:
            st.metric(
                "Available for Distribution",
                f"KES {result['available_for_distribution']:,.0f}",
                help="Funds available after all provisions and reserves"
            )
        
        # Policy gates check
        st.markdown("#### Policy Gates Check")
        gates_data = []
        
        for gate in result['policy_gates']:
            status = "✅ PASS" if gate['passed'] else "❌ FAIL"
            gates_data.append({
                'Policy Gate': gate['name'],
                'Current Value': gate['current_value'],
                'Threshold': gate['threshold'],
                'Status': status,
                'Description': gate['description']
            })
        
        gates_df = pd.DataFrame(gates_data)
        st.dataframe(gates_df, use_container_width=True)
        
        # Dividend breakdown
        st.markdown("#### Surplus Distribution Breakdown")
        
        distribution_data = {
            'Category': [
                'Net Surplus',
                'Statutory Reserve',
                'ECL Provision', 
                'Other Reserves',
                'Available for Dividend',
                'Recommended Dividend'
            ],
            'Amount (KES)': [
                result['net_surplus'],
                result['statutory_reserve_amount'],
                result['ecl_provision'],
                result['other_reserves'],
                result['available_for_distribution'],
                result['dividend_amount']
            ]
        }
        
        fig = px.bar(
            distribution_data,
            x='Category',
            y='Amount (KES)',
            title="Surplus Distribution Breakdown",
            color='Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical dividend trend
        self.render_historical_trend()
    
    def render_historical_trend(self):
        """Render historical dividend trends"""
        st.markdown("---")
        st.subheader("Historical Dividend Trends")
        
        # Generate sample historical data
        years = [2020, 2021, 2022, 2023, 2024]
        historical_data = pd.DataFrame({
            'Year': years,
            'Dividend_Rate': [5.2, 5.8, 6.1, 6.5, 6.8],  # Sample data
            'Net_Surplus': [2.1, 2.8, 3.2, 3.5, 3.8],
            'Liquidity_Ratio': [16.5, 17.2, 17.8, 18.2, 18.5],
            'PAR30': [5.8, 5.2, 4.8, 4.5, 4.2]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data['Year'],
                y=historical_data['Dividend_Rate'],
                name='Dividend Rate (%)',
                line=dict(color='green', width=4)
            ))
            fig.update_layout(
                title="Historical Dividend Rates",
                xaxis_title="Year",
                yaxis_title="Dividend Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data['Year'],
                y=historical_data['Net_Surplus'],
                name='Net Surplus (KES M)',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=historical_data['Year'],
                y=historical_data['PAR30'],
                name='PAR30 (%)',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            fig.update_layout(
                title="Surplus vs PAR30 Trend",
                xaxis_title="Year",
                yaxis_title="Net Surplus (KES M)",
                yaxis2=dict(
                    title="PAR30 (%)",
                    overlaying='y',
                    side='right'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_dividend_policy(self):
        """Render dividend policy information"""
        st.markdown("---")
        st.subheader("Dividend Policy Framework")
        
        policy_info = """
        ### SACCO Dividend Declaration Policy
        
        **1. Statutory Requirements:**
        - Minimum 20% of net surplus to statutory reserve
        - Adequate provisions for bad and doubtful debts
        - Compliance with SASRA capital adequacy ratios
        
        **2. Prudential Gates:**
        - Liquidity ratio must exceed 10%
        - PAR30 must be below 15%
        - Capital adequacy ratio above regulatory minimum
        
        **3. Distribution Limits:**
        - Maximum dividend rate: 7% of share capital
        - Must maintain adequate operational reserves
        - Consideration of future capital requirements
        
        **4. Board Discretion:**
        - Board may recommend lower dividend if future investments planned
        - Consideration of member expectations and market conditions
        - Alignment with long-term strategic objectives
        """
        
        st.markdown(policy_info)
    
    def run(self):
        """Run the dividend capacity page"""
        st.title("💰 Dividend Capacity Analysis")
        
        st.markdown("""
        This module calculates the SACCO's capacity to declare dividends based on financial performance, 
        regulatory requirements, and prudential guidelines.
        """)
        
        self.render_dividend_calculator()
        self.render_dividend_policy()

if __name__ == "__main__":
    page = DividendCapacityPage()
    page.run()