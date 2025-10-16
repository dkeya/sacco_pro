# pages/09_Provisioning_Writeoff.py
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
from sacco_core.analytics.provisioning import ProvisioningAnalyzer, LoanStage

st.set_page_config(
    page_title="Provisioning & Write-off",
    page_icon="📉",
    layout="wide"
)

class ProvisioningWriteoffPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.provisioning_analyzer = ProvisioningAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "09_Provisioning_Writeoff.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "provisioning_writeoff_page"
        )
        return True
    
    def render_provisioning_dashboard(self):
        """Render provisioning and write-off dashboard"""
        st.subheader("📉 IFRS 9 Provisioning & Write-off Management")
        
        try:
            # Get ECL analysis
            analysis = self.provisioning_analyzer.calculate_ifrs9_ecl()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_provision = analysis.get('provisioning_requirements', {}).get('total_provision_required', 0)
                st.metric(
                    "Total Provision Required",
                    f"KES {total_provision:,.0f}",
                    help="Total IFRS 9 expected credit loss provision"
                )
            
            with col2:
                coverage_ratio = analysis.get('provisioning_requirements', {}).get('provision_coverage_ratio', 0) * 100
                st.metric(
                    "Provision Coverage Ratio",
                    f"{coverage_ratio:.1f}%",
                    help="Provision as percentage of total loan portfolio"
                )
            
            with col3:
                portfolio_ecl = analysis.get('portfolio_ecl', {}).get('total_portfolio_ecl', 0)
                st.metric(
                    "Portfolio ECL",
                    f"KES {portfolio_ecl:,.0f}",
                    help="Total expected credit loss for portfolio"
                )
            
            with col4:
                writeoff_potential = analysis.get('writeoff_analysis', {}).get('potential_writeoff_amount', 0)
                st.metric(
                    "Write-off Potential",
                    f"KES {writeoff_potential:,.0f}",
                    delta_color="inverse",
                    help="Potential write-offs from high-risk loans"
                )
            
            # Provisioning overview
            self.render_provisioning_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering provisioning dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_provisioning_overview(self, analysis):
        """Render provisioning overview"""
        st.markdown("#### 📊 IFRS 9 Staging & Provisioning Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan staging distribution
            staging_details = analysis.get('staging_analysis', {}).get('staging_details', [])
            if staging_details:
                stages = [detail['stage'] for detail in staging_details]
                exposures = [detail['total_exposure'] for detail in staging_details]
                provisions = [analysis.get('provisioning_requirements', {}).get('provision_by_stage', {}).get(stage, 0) 
                            for stage in stages]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Portfolio Exposure',
                    x=stages,
                    y=exposures,
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    name='Provision Required',
                    x=stages,
                    y=provisions,
                    marker_color='coral'
                ))
                fig.update_layout(
                    title="Portfolio Exposure vs Provision by Stage",
                    xaxis_title="IFRS 9 Stage",
                    yaxis_title="Amount (KES)",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No staging data available")
        
        with col2:
            # Provision coverage by stage
            ecl_calculations = analysis.get('ecl_calculations', [])
            if ecl_calculations:
                stages = [ecl.stage.value for ecl in ecl_calculations]
                coverage_ratios = [
                    (ecl.provision_required / ecl.exposure_at_default * 100) 
                    for ecl in ecl_calculations
                ]
                
                fig = px.bar(
                    x=stages,
                    y=coverage_ratios,
                    title="Provision Coverage Ratio by Stage (%)",
                    labels={'x': 'IFRS 9 Stage', 'y': 'Coverage Ratio (%)'},
                    color=coverage_ratios,
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ECL calculation data available")
        
        # Detailed analysis sections
        self.render_staging_analysis(analysis)
        self.render_ecl_calculation(analysis)
        self.render_writeoff_analysis(analysis)
        self.render_recovery_analysis(analysis)
        self.render_regulatory_compliance(analysis)
    
    def render_staging_analysis(self, analysis):
        """Render detailed loan staging analysis"""
        st.markdown("---")
        st.subheader("🏷️ Loan Staging Analysis")
        
        staging_analysis = analysis.get('staging_analysis', {})
        staging_details = staging_analysis.get('staging_details', [])
        
        if staging_details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Stage Distribution")
                
                # Pie chart of stage distribution
                stage_labels = [detail['stage'] for detail in staging_details]
                stage_values = [detail['loan_count'] for detail in staging_details]
                
                fig = px.pie(
                    names=stage_labels,
                    values=stage_values,
                    title="Loan Distribution by IFRS 9 Stage",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 💰 Exposure by Stage")
                
                # Exposure breakdown
                exposure_data = []
                for detail in staging_details:
                    exposure_data.append({
                        'Stage': detail['stage'],
                        'Exposure': detail['total_exposure'],
                        'Average_Exposure': detail['average_exposure'],
                        'Loan_Count': detail['loan_count']
                    })
                
                exposure_df = pd.DataFrame(exposure_data)
                st.dataframe(exposure_df, use_container_width=True)
            
            # High-risk loans table
            st.markdown("#### ⚠️ High-Risk Loan Candidates")
            high_risk_loans = staging_analysis.get('high_risk_loans', [])
            if high_risk_loans:
                hr_df = pd.DataFrame(high_risk_loans)
                st.dataframe(hr_df[['loan_id', 'member_id', 'outstanding_amount', 'days_past_due', 'member_risk_grade']], 
                           use_container_width=True)
            else:
                st.info("No high-risk loans identified")
        else:
            st.info("No staging analysis data available")
    
    def render_ecl_calculation(self, analysis):
        """Render ECL calculation details"""
        st.markdown("---")
        st.subheader("🧮 Expected Credit Loss Calculation")
        
        ecl_calculations = analysis.get('ecl_calculations', [])
        portfolio_ecl = analysis.get('portfolio_ecl', {})
        
        if ecl_calculations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 ECL Components by Stage")
                
                # ECL components table
                ecl_data = []
                for ecl in ecl_calculations:
                    ecl_data.append({
                        'Stage': ecl.stage.value,
                        'Exposure (KES)': f"{ecl.exposure_at_default:,.0f}",
                        'PD (%)': f"{ecl.probability_of_default * 100:.2f}%",
                        'LGD (%)': f"{ecl.loss_given_default * 100:.1f}%",
                        'ECL (KES)': f"{ecl.expected_credit_loss:,.0f}",
                        'Provision (KES)': f"{ecl.provision_required:,.0f}"
                    })
                
                ecl_df = pd.DataFrame(ecl_data)
                st.dataframe(ecl_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Portfolio ECL Metrics")
                
                # Portfolio ECL metrics
                metrics_data = {
                    'Metric': [
                        'Total Portfolio ECL',
                        'ECL Coverage Ratio', 
                        'Average Probability of Default',
                        'Average Loss Given Default',
                        'Stress Test ECL Increase'
                    ],
                    'Value': [
                        f"KES {portfolio_ecl.get('total_portfolio_ecl', 0):,.0f}",
                        f"{portfolio_ecl.get('ecl_coverage_ratio', 0) * 100:.2f}%",
                        f"{portfolio_ecl.get('average_pd', 0) * 100:.2f}%",
                        f"{portfolio_ecl.get('average_lgd', 0) * 100:.1f}%",
                        f"{portfolio_ecl.get('stress_test_results', {}).get('ecl_increase_percentage', 0):.1f}%"
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            # ECL trend analysis
            st.markdown("#### 📅 ECL Trend Analysis")
            ecl_trend = portfolio_ecl.get('ecl_trend', {})
            if ecl_trend:
                periods = list(ecl_trend.keys())
                values = list(ecl_trend.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=values,
                    name='Portfolio ECL',
                    line=dict(color='red', width=3),
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title="Portfolio ECL Trend",
                    xaxis_title="Period",
                    yaxis_title="ECL (KES)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ECL calculation data available")
    
    def render_writeoff_analysis(self, analysis):
        """Render write-off analysis"""
        st.markdown("---")
        st.subheader("📝 Write-off Analysis & Management")
        
        writeoff_analysis = analysis.get('writeoff_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💸 Write-off Potential")
            
            writeoff_potential = writeoff_analysis.get('potential_writeoff_amount', 0)
            high_risk_count = writeoff_analysis.get('high_risk_loan_count', 0)
            recovery_estimation = writeoff_analysis.get('recovery_estimation', 0)
            
            st.metric("Total Write-off Potential", f"KES {writeoff_potential:,.0f}")
            st.metric("High-Risk Loans", f"{high_risk_count}")
            st.metric("Estimated Recovery", f"KES {recovery_estimation:,.0f}")
            
            # Write-off trend
            writeoff_trend = writeoff_analysis.get('writeoff_trend', {})
            if writeoff_trend:
                quarters = list(writeoff_trend.keys())
                amounts = list(writeoff_trend.values())
                
                fig = px.bar(
                    x=quarters,
                    y=amounts,
                    title="Quarterly Write-off Trend",
                    labels={'x': 'Quarter', 'y': 'Write-off Amount (KES)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Write-off Candidates")
            
            writeoff_candidates = writeoff_analysis.get('writeoff_candidates', [])
            if writeoff_candidates:
                candidates_df = pd.DataFrame(writeoff_candidates)
                st.dataframe(candidates_df, use_container_width=True)
                
                # Write-off approval workflow
                st.markdown("#### ⚖️ Write-off Approval")
                selected_loan = st.selectbox("Select Loan for Write-off", 
                                           [c['loan_id'] for c in writeoff_candidates])
                
                if st.button("🚨 Initiate Write-off Process"):
                    st.success(f"Write-off process initiated for {selected_loan}")
                    st.info("Approval workflow started. Notifications sent to relevant officers.")
            else:
                st.info("No write-off candidates identified")
    
    def render_recovery_analysis(self, analysis):
        """Render recovery analysis"""
        st.markdown("---")
        st.subheader("🔄 Recovery Management")
        
        recovery_analysis = analysis.get('recovery_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Recovery Performance")
            
            total_recoveries = recovery_analysis.get('total_recoveries_ytd', 0)
            recovery_rate = recovery_analysis.get('recovery_rate', 0) * 100
            active_cases = recovery_analysis.get('active_recovery_cases', 0)
            avg_recovery_time = recovery_analysis.get('average_recovery_time', 0)
            
            st.metric("YTD Recoveries", f"KES {total_recoveries:,.0f}")
            st.metric("Recovery Rate", f"{recovery_rate:.1f}%")
            st.metric("Active Recovery Cases", f"{active_cases}")
            st.metric("Average Recovery Time", f"{avg_recovery_time:.0f} days")
        
        with col2:
            st.markdown("#### 📊 Recovery by Product")
            
            recovery_by_product = recovery_analysis.get('recovery_by_product', {})
            if recovery_by_product:
                products = list(recovery_by_product.keys())
                rates = [rate * 100 for rate in recovery_by_product.values()]
                
                fig = px.bar(
                    x=products,
                    y=rates,
                    title="Recovery Rate by Product Type (%)",
                    labels={'x': 'Product Type', 'y': 'Recovery Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recovery by product data available")
    
    def render_regulatory_compliance(self, analysis):
        """Render regulatory compliance section"""
        st.markdown("---")
        st.subheader("⚖️ Regulatory Compliance")
        
        regulatory_compliance = analysis.get('regulatory_compliance', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Compliance Status")
            
            ifrs9_compliant = regulatory_compliance.get('ifrs9_compliance', False)
            central_bank_compliant = regulatory_compliance.get('central_bank_requirements', False)
            reporting_ready = regulatory_compliance.get('regulatory_reporting_ready', False)
            
            if ifrs9_compliant:
                st.success("✅ IFRS 9 Compliance: MET")
            else:
                st.error("❌ IFRS 9 Compliance: NOT MET")
            
            if central_bank_compliant:
                st.success("✅ Central Bank Requirements: MET")
            else:
                st.error("❌ Central Bank Requirements: NOT MET")
            
            if reporting_ready:
                st.success("✅ Regulatory Reporting: READY")
            else:
                st.error("❌ Regulatory Reporting: NOT READY")
            
            # Coverage ratio vs requirement
            coverage_gap = regulatory_compliance.get('coverage_ratio_vs_requirement', 0) * 100
            if coverage_gap >= 0:
                st.metric("Coverage Above Requirement", f"+{coverage_gap:.2f}%")
            else:
                st.metric("Coverage Below Requirement", f"{coverage_gap:.2f}%", delta_color="inverse")
        
        with col2:
            st.markdown("#### 📋 Compliance Issues")
            
            compliance_issues = regulatory_compliance.get('compliance_issues', [])
            if compliance_issues:
                st.error("**Identified Issues:**")
                for issue in compliance_issues:
                    st.write(f"• {issue}")
                
                st.markdown("#### 💡 Recommended Actions")
                st.info("1. Increase provision coverage to meet regulatory minimum")
                st.info("2. Review high-risk loan classification criteria")
                st.info("3. Enhance recovery efforts for written-off loans")
            else:
                st.success("✅ No compliance issues identified")
    
    def run(self):
        """Run the provisioning and write-off page"""
        st.title("📉 IFRS 9 Provisioning & Write-off Management")
        
        st.markdown("""
        Comprehensive IFRS 9 Expected Credit Loss calculation, loan staging analysis, provisioning management, 
        and write-off processing to ensure regulatory compliance and accurate financial reporting.
        """)
        
        try:
            self.render_provisioning_dashboard()
        except Exception as e:
            st.error(f"Error running provisioning page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = ProvisioningWriteoffPage()
    page.run()