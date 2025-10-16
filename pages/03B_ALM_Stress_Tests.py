# pages/03B_ALM_Stress_Tests.py
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
from sacco_core.analytics.alm_stress import ALMStressTester

st.set_page_config(
    page_title="ALM Stress Tests",
    page_icon="🌊",
    layout="wide"
)

class ALMStressTestPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.stress_tester = ALMStressTester()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "03B_ALM_Stress_Tests.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "alm_stress_test_page"
        )
        return True
    
    def render_stress_scenarios(self):
        """Render stress scenario configuration"""
        st.subheader("💡 Stress Scenario Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏛️ Regulatory Stress Scenarios")
            
            # Deposit run-off scenarios
            deposit_runoff_mild = st.slider(
                "Mild Deposit Run-off (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Deposit withdrawal in mild stress scenario"
            )
            
            deposit_runoff_severe = st.slider(
                "Severe Deposit Run-off (%)", 
                min_value=0.0,
                max_value=80.0,
                value=25.0,
                step=1.0,
                help="Deposit withdrawal in severe stress scenario"
            )
            
            liquidity_shock = st.slider(
                "Liquidity Shock (BPS)",
                min_value=0,
                max_value=500,
                value=100,
                step=10,
                help="Interest rate shock in basis points"
            )
        
        with col2:
            st.markdown("#### 📈 Market Risk Scenarios")
            
            # Interest rate shocks
            rate_shock_mild = st.slider(
                "Mild Rate Shock (BPS)",
                min_value=0,
                max_value=300,
                value=50,
                step=10,
                help="Mild interest rate increase scenario"
            )
            
            rate_shock_severe = st.slider(
                "Severe Rate Shock (BPS)",
                min_value=0, 
                max_value=500,
                value=200,
                step=10,
                help="Severe interest rate increase scenario"
            )
            
            yield_curve_shift = st.selectbox(
                "Yield Curve Shift",
                ["Parallel Up", "Parallel Down", "Steepening", "Flattening"],
                help="Type of yield curve movement"
            )
        
        with col3:
            st.markdown("#### 🏢 Credit Risk Scenarios")
            
            # Credit quality deterioration
            pd_increase_mild = st.slider(
                "Mild PD Increase (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Probability of default increase in mild scenario"
            )
            
            pd_increase_severe = st.slider(
                "Severe PD Increase (%)",
                min_value=0.0,
                max_value=50.0, 
                value=15.0,
                step=0.5,
                help="Probability of default increase in severe scenario"
            )
            
            collateral_haircut = st.slider(
                "Collateral Haircut (%)",
                min_value=0.0,
                max_value=40.0,
                value=10.0,
                step=1.0,
                help="Reduction in collateral values"
            )
        
        # Run stress tests
        if st.button("🚀 Run Stress Tests", type="primary"):
            scenarios = {
                'mild_deposit_runoff': deposit_runoff_mild / 100,
                'severe_deposit_runoff': deposit_runoff_severe / 100,
                'liquidity_shock_bps': liquidity_shock,
                'mild_rate_shock_bps': rate_shock_mild,
                'severe_rate_shock_bps': rate_shock_severe,
                'yield_curve_shift': yield_curve_shift,
                'mild_pd_increase': pd_increase_mild / 100,
                'severe_pd_increase': pd_increase_severe / 100,
                'collateral_haircut': collateral_haircut / 100
            }
            
            results = self.stress_tester.run_all_stress_tests(scenarios)
            st.session_state.stress_results = results
    
    def render_stress_results(self):
        """Render stress test results"""
        if 'stress_results' not in st.session_state:
            return
        
        results = st.session_state.stress_results
        
        st.markdown("---")
        st.subheader("📊 Stress Test Results Summary")
        
        # Key metrics comparison
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            baseline_liquidity = results['baseline']['liquidity_ratio'] * 100
            stressed_liquidity = results['mild_deposit_runoff']['liquidity_ratio'] * 100
            st.metric(
                "Liquidity Ratio",
                f"{stressed_liquidity:.1f}%",
                delta=f"{(stressed_liquidity - baseline_liquidity):.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            baseline_capital = results['baseline']['capital_adequacy'] * 100
            stressed_capital = results['severe_deposit_runoff']['capital_adequacy'] * 100
            st.metric(
                "Capital Adequacy", 
                f"{stressed_capital:.1f}%",
                delta=f"{(stressed_capital - baseline_capital):.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            baseline_gap = results['baseline']['cumulative_gap']
            stressed_gap = results['mild_rate_shock']['cumulative_gap']
            st.metric(
                "Funding Gap",
                f"KES {stressed_gap:,.0f}",
                delta=f"KES {(stressed_gap - baseline_gap):,.0f}",
                delta_color="inverse"
            )
        
        with col4:
            baseline_nii = results['baseline']['net_interest_income']
            stressed_nii = results['severe_rate_shock']['net_interest_income']
            st.metric(
                "Net Interest Income",
                f"KES {stressed_nii:,.0f}",
                delta=f"KES {(stressed_nii - baseline_nii):,.0f}",
                delta_color="inverse"
            )
        
        # Scenario comparison chart
        self.render_scenario_comparison(results)
        
        # Detailed results by scenario
        self.render_detailed_scenarios(results)
        
        # Gap analysis under stress
        self.render_stress_gap_analysis(results)
    
    def render_scenario_comparison(self, results):
        """Render scenario comparison visualization"""
        st.markdown("#### 📈 Scenario Impact Comparison")
        
        scenarios = list(results.keys())
        metrics = ['liquidity_ratio', 'capital_adequacy', 'net_interest_income', 'cumulative_gap']
        
        comparison_data = []
        for scenario in scenarios:
            for metric in metrics:
                value = results[scenario][metric]
                if metric in ['liquidity_ratio', 'capital_adequacy']:
                    value = value * 100  # Convert to percentage
                
                comparison_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': value
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create faceted bar chart
        fig = px.bar(
            comparison_df,
            x='Scenario',
            y='Value',
            color='Scenario',
            facet_col='Metric',
            facet_col_wrap=2,
            title="Key Metrics Across Stress Scenarios",
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_scenarios(self, results):
        """Render detailed scenario results"""
        st.markdown("#### 🔍 Detailed Scenario Analysis")
        
        # Create results table
        scenarios_data = []
        for scenario_name, scenario_data in results.items():
            scenarios_data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Liquidity Ratio': f"{scenario_data['liquidity_ratio'] * 100:.1f}%",
                'Capital Adequacy': f"{scenario_data['capital_adequacy'] * 100:.1f}%", 
                'Net Interest Income': f"KES {scenario_data['net_interest_income']:,.0f}",
                'Funding Gap': f"KES {scenario_data['cumulative_gap']:,.0f}",
                'ECL Increase': f"KES {scenario_data['ecl_increase']:,.0f}",
                'Status': scenario_data['status']
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        
        # Color code status
        def color_status(status):
            if status == 'Pass':
                return 'background-color: #90EE90'
            elif status == 'Watch':
                return 'background-color: #FFE4B5' 
            else:
                return 'background-color: #FFB6C1'
        
        styled_df = scenarios_df.style.applymap(
            color_status, subset=['Status']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def render_stress_gap_analysis(self, results):
        """Render gap analysis under stress"""
        st.markdown("#### ⚠️ Maturity Gap Analysis Under Stress")
        
        # Sample maturity buckets data under different scenarios
        buckets = ['O/N', '1-7D', '8-30D', '31-90D', '91-180D', '181-365D', '1-3Y', '3Y+']
        
        gap_data = pd.DataFrame({
            'Bucket': buckets,
            'Baseline_Gap': [15, 25, 45, 60, 40, 35, 20, 10],
            'Mild_Stress_Gap': [10, 18, 35, 45, 30, 25, 15, 8],
            'Severe_Stress_Gap': [5, 10, 20, 30, 20, 15, 10, 5]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=gap_data['Bucket'],
            y=gap_data['Baseline_Gap'],
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Mild Stress',
            x=gap_data['Bucket'], 
            y=gap_data['Mild_Stress_Gap'],
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            name='Severe Stress',
            x=gap_data['Bucket'],
            y=gap_data['Severe_Stress_Gap'], 
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Maturity Gap Analysis: Baseline vs Stress Scenarios",
            xaxis_title="Maturity Bucket",
            yaxis_title="Gap (KES M)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Liquidity coverage ratio under stress
        self.render_liquidity_coverage(results)
    
    def render_liquidity_coverage(self, results):
        """Render liquidity coverage analysis"""
        st.markdown("#### 💧 Liquidity Coverage Ratio (LCR) Under Stress")
        
        # Sample LCR data
        time_horizons = ['1D', '7D', '30D', '90D']
        lcr_data = pd.DataFrame({
            'Time_Horizon': time_horizons,
            'Baseline_LCR': [125, 115, 105, 95],
            'Mild_Stress_LCR': [110, 95, 85, 75], 
            'Severe_Stress_LCR': [85, 70, 60, 50]
        })
        
        fig = px.line(
            lcr_data,
            x='Time_Horizon',
            y=['Baseline_LCR', 'Mild_Stress_LCR', 'Severe_Stress_LCR'],
            title="Liquidity Coverage Ratio Across Time Horizons",
            labels={'value': 'LCR (%)', 'variable': 'Scenario'}
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Regulatory Minimum")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_mitigation_strategies(self):
        """Render stress mitigation strategies"""
        st.markdown("---")
        st.subheader("🛡️ Stress Mitigation Strategies")
        
        mitigation_strategies = """
        ### Recommended Mitigation Actions
        
        **1. Liquidity Stress:**
        - Maintain higher liquid asset buffers
        - Diversify funding sources
        - Establish contingent funding lines
        - Develop asset sale programs
        
        **2. Interest Rate Risk:**
        - Implement interest rate hedging strategies
        - Balance fixed vs floating rate exposures
        - Regular gap analysis and monitoring
        - Stress testing of NII sensitivity
        
        **3. Credit Risk:**
        - Strengthen underwriting standards
        - Increase collateral requirements
        - Enhance collection strategies
        - Build adequate provisioning buffers
        
        **4. Operational Response:**
        - Develop contingency funding plan
        - Establish crisis management team
        - Regular stress testing exercises
        - Board-level risk oversight
        """
        
        st.markdown(mitigation_strategies)
        
        # Action plan template
        with st.expander("📋 Stress Response Action Plan Template"):
            st.text_area(
                "Immediate Actions (0-7 days):",
                "1. Activate contingency funding plan\n2. Contact key depositors\n3. Review large exposure limits\n4. Emergency board meeting",
                height=100
            )
            
            st.text_area(
                "Short-term Actions (1-4 weeks):",
                "1. Implement deposit retention strategies\n2. Review loan disbursement policies\n3. Enhance liquidity monitoring\n4. Stakeholder communication",
                height=100
            )
            
            st.text_area(
                "Long-term Actions (1-6 months):",
                "1. Strategic capital planning\n2. Funding diversification initiative\n3. Risk management framework enhancement\n4. Regulatory engagement",
                height=100
            )
    
    def run(self):
        """Run the ALM stress test page"""
        st.title("🌊 ALM Stress Testing")
        
        st.markdown("""
        Comprehensive Asset-Liability Management stress testing to assess the SACCO's resilience 
        under various adverse scenarios including liquidity shocks, interest rate changes, 
        and credit quality deterioration.
        """)
        
        self.render_stress_scenarios()
        self.render_stress_results()
        self.render_mitigation_strategies()

if __name__ == "__main__":
    page = ALMStressTestPage()
    page.run()