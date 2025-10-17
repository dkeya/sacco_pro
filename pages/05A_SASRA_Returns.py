# pages/05A_SASRA_Returns.py
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
from sacco_core.analytics.sasra import SASRAReturnsGenerator
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="SASRA Returns",
    page_icon="📋",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class SASRAReturnsPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.sasra_generator = SASRAReturnsGenerator()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "05A_SASRA_Returns.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "sasra_returns_page"
        )
        return True
    
    def render_sasra_dashboard(self):
        """Render SASRA returns dashboard"""
        st.subheader("📊 SASRA Returns Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Reporting Period",
                "Q1 2024",
                help="Current SASRA reporting quarter"
            )
        
        with col2:
            st.metric(
                "Next Submission Date", 
                "2024-03-31",
                "45 days",
                delta_color="inverse",
                help="Days until next submission deadline"
            )
        
        with col3:
            st.metric(
                "Last Submission Status",
                "Submitted",
                help="Status of last quarterly submission"
            )
        
        with col4:
            st.metric(
                "Compliance Score",
                "92%",
                "3%",
                help="SASRA regulatory compliance rating"
            )
        
        # SASRA rating history
        self.render_sasra_rating_history()
    
    def render_sasra_rating_history(self):
        """Render SASRA rating history"""
        st.markdown("#### 📈 SASRA Supervisory Rating History")
        
        rating_data = pd.DataFrame({
            'Period': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
            'CAMEL Rating': [3, 2, 2, 2, 2],
            'Composite Rating': [2, 2, 2, 2, 2],
            'Capital Adequacy': [2, 2, 2, 2, 1],
            'Asset Quality': [3, 2, 2, 2, 2],
            'Management': [2, 2, 2, 2, 2],
            'Earnings': [2, 2, 2, 2, 2],
            'Liquidity': [2, 2, 2, 2, 2]
        })
        
        # Rating scale: 1=Strong, 2=Satisfactory, 3=Fair, 4=Marginal, 5=Unsatisfactory
        fig = go.Figure()
        
        rating_components = ['Capital Adequacy', 'Asset Quality', 'Management', 'Earnings', 'Liquidity']
        colors = ['#2E8B57', '#3CB371', '#90EE90', '#F0E68C', '#FFA07A']
        
        for i, component in enumerate(rating_components):
            fig.add_trace(go.Scatter(
                x=rating_data['Period'],
                y=rating_data[component],
                name=component,
                line=dict(color=colors[i], width=4),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="SASRA CAMELS Rating Components Over Time",
            xaxis_title="Reporting Period",
            yaxis_title="Rating (1=Strong, 5=Unsatisfactory)",
            yaxis=dict(autorange='reversed'),  # Lower numbers are better
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_returns_generator(self):
        """Render SASRA returns generator"""
        st.markdown("---")
        st.subheader("🔄 SASRA Returns Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Returns Configuration")
            
            reporting_period = st.selectbox(
                "Reporting Period",
                ["Q1 2024", "Q4 2023", "Q3 2023", "Q2 2023", "Q1 2023"],
                help="Select the reporting period"
            )
            
            return_type = st.selectbox(
                "Return Type",
                ["Quarterly Prudential Returns", "Annual Returns", "Monthly Liquidity Returns"],
                help="Type of SASRA return to generate"
            )
            
            include_reconciliations = st.checkbox(
                "Include Reconciliation Reports",
                value=True,
                help="Generate supporting reconciliation reports"
            )
            
            validation_checks = st.checkbox(
                "Run Validation Checks",
                value=True,
                help="Perform data validation before generation"
            )
        
        with col2:
            st.markdown("#### 📊 Data Sources")
            
            st.info("**Automated Data Mapping**")
            st.success("✓ GL Accounts → SASRA Lines")
            st.success("✓ Loan Portfolio → Asset Quality")
            st.success("✓ Deposit Ledger → Liability Structure")
            st.success("✓ Member Data → Large Exposures")
            
            if st.button("🚀 Generate SASRA Returns", type="primary"):
                with st.spinner("Generating SASRA returns..."):
                    returns_data = self.sasra_generator.generate_returns(
                        period=reporting_period,
                        return_type=return_type,
                        include_reconciliations=include_reconciliations,
                        run_validation=validation_checks
                    )
                    
                    st.session_state.sasra_returns = returns_data
                    st.success("SASRA returns generated successfully!")
        
        # Display generated returns
        if 'sasra_returns' in st.session_state:
            self.render_generated_returns(st.session_state.sasra_returns)
    
    def render_generated_returns(self, returns_data):
        """Render generated SASRA returns"""
        st.markdown("---")
        st.subheader("📋 Generated SASRA Returns")
        
        # Main returns summary
        st.markdown("#### 📈 Prudential Returns Summary")
        
        summary_data = returns_data.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Assets",
                f"KES {summary_data.get('total_assets', 0):,.0f}",
                help="Total assets as per SASRA format"
            )
        
        with col2:
            st.metric(
                "Total Liabilities",
                f"KES {summary_data.get('total_liabilities', 0):,.0f}",
                help="Total liabilities as per SASRA format"
            )
        
        with col3:
            st.metric(
                "Capital Adequacy",
                f"{summary_data.get('capital_adequacy', 0)*100:.1f}%",
                help="Core capital to total assets"
            )
        
        with col4:
            st.metric(
                "Liquidity Ratio",
                f"{summary_data.get('liquidity_ratio', 0)*100:.1f}%",
                help="Liquid assets to total deposits"
            )
        
        # Detailed returns sections
        self.render_detailed_returns(returns_data)
        
        # Export options
        self.render_export_options(returns_data)
    
    def render_detailed_returns(self, returns_data):
        """Render detailed SASRA returns sections"""
        st.markdown("#### 📊 Detailed Returns Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Balance Sheet", "Asset Quality", "Large Exposures", "Capital Adequacy", "Liquidity"
        ])
        
        with tab1:
            self.render_balance_sheet_returns(returns_data.get('balance_sheet', {}))
        
        with tab2:
            self.render_asset_quality_returns(returns_data.get('asset_quality', {}))
        
        with tab3:
            self.render_large_exposures(returns_data.get('large_exposures', {}))
        
        with tab4:
            self.render_capital_adequacy(returns_data.get('capital_adequacy', {}))
        
        with tab5:
            self.render_liquidity_analysis(returns_data.get('liquidity', {}))
    
    def render_balance_sheet_returns(self, balance_sheet_data):
        """Render balance sheet returns"""
        st.markdown("##### 💰 Balance Sheet Structure")
        
        # Assets breakdown
        assets_data = balance_sheet_data.get('assets', {})
        liabilities_data = balance_sheet_data.get('liabilities', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Assets Composition**")
            assets_df = pd.DataFrame({
                'Category': list(assets_data.keys()),
                'Amount (KES)': list(assets_data.values())
            })
            st.dataframe(assets_df, use_container_width=True)
            
            # Assets pie chart
            if assets_data:
                fig = px.pie(
                    assets_df,
                    values='Amount (KES)',
                    names='Category',
                    title="Assets Composition"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Liabilities Composition**")
            liabilities_df = pd.DataFrame({
                'Category': list(liabilities_data.keys()),
                'Amount (KES)': list(liabilities_data.values())
            })
            st.dataframe(liabilities_df, use_container_width=True)
            
            # Liabilities pie chart
            if liabilities_data:
                fig = px.pie(
                    liabilities_df,
                    values='Amount (KES)',
                    names='Category',
                    title="Liabilities Composition"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_asset_quality_returns(self, asset_quality_data):
        """Render asset quality returns"""
        st.markdown("##### 📉 Asset Quality Analysis")
        
        # PAR ladder
        par_data = asset_quality_data.get('par_ladder', {})
        
        if par_data:
            par_df = pd.DataFrame({
                'Days Past Due': list(par_data.keys()),
                'Amount (KES)': list(par_data.values()),
                'Percentage': [f"{(v/sum(par_data.values()))*100:.1f}%" for v in par_data.values()]
            })
            
            st.dataframe(par_df, use_container_width=True)
            
            # PAR ladder chart
            fig = px.bar(
                par_df,
                x='Days Past Due',
                y='Amount (KES)',
                title="Portfolio at Risk (PAR) Ladder",
                color='Days Past Due'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # NPL analysis
        npl_ratio = asset_quality_data.get('npl_ratio', 0)
        provisioning_cover = asset_quality_data.get('provisioning_cover', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "NPL Ratio",
                f"{npl_ratio*100:.2f}%",
                help="Non-performing loans to total loans"
            )
        
        with col2:
            st.metric(
                "Provisioning Coverage",
                f"{provisioning_cover*100:.1f}%",
                help="Loan loss provisions to NPLs"
            )
    
    def render_large_exposures(self, large_exposures_data):
        """Render large exposures analysis"""
        st.markdown("##### 🎯 Large Exposures & Concentration Risk")
        
        # Top employer exposures
        employer_exposures = large_exposures_data.get('employer_exposures', [])
        
        if employer_exposures:
            employer_df = pd.DataFrame(employer_exposures)
            st.dataframe(employer_df, use_container_width=True)
            
            # Large exposures chart
            fig = px.bar(
                employer_df.head(10),
                x='employer_name',
                y='exposure_amount',
                title="Top 10 Employer Exposures",
                labels={'exposure_amount': 'Exposure (KES)', 'employer_name': 'Employer'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Concentration ratios
        concentration_ratios = large_exposures_data.get('concentration_ratios', {})
        
        if concentration_ratios:
            st.markdown("**Concentration Ratios**")
            for ratio_name, ratio_value in concentration_ratios.items():
                st.metric(
                    ratio_name,
                    f"{ratio_value*100:.2f}%",
                    help=f"{ratio_name} concentration ratio"
                )
    
    def render_capital_adequacy(self, capital_data):
        """Render capital adequacy analysis"""
        st.markdown("##### 🏛️ Capital Adequacy Analysis")
        
        # Capital components
        capital_components = capital_data.get('components', {})
        
        if capital_components:
            capital_df = pd.DataFrame({
                'Component': list(capital_components.keys()),
                'Amount (KES)': list(capital_components.values())
            })
            
            st.dataframe(capital_df, use_container_width=True)
            
            # Capital structure chart
            fig = px.pie(
                capital_df,
                values='Amount (KES)',
                names='Component',
                title="Capital Structure"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Capital ratios
        capital_ratios = capital_data.get('ratios', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            core_capital = capital_ratios.get('core_capital_ratio', 0)
            st.metric(
                "Core Capital Ratio",
                f"{core_capital*100:.2f}%",
                delta=f"{(core_capital-0.10)*100:.2f}% vs min",
                help="Minimum requirement: 10%"
            )
        
        with col2:
            total_capital = capital_ratios.get('total_capital_ratio', 0)
            st.metric(
                "Total Capital Ratio",
                f"{total_capital*100:.2f}%",
                help="Total capital to risk-weighted assets"
            )
        
        with col3:
            statutory_reserve = capital_ratios.get('statutory_reserve_ratio', 0)
            st.metric(
                "Statutory Reserve",
                f"{statutory_reserve*100:.2f}%",
                help="Statutory reserve to total deposits"
            )
    
    def render_liquidity_analysis(self, liquidity_data):
        """Render liquidity analysis"""
        st.markdown("##### 💧 Liquidity Analysis")
        
        # Liquidity ratios
        liquidity_ratios = liquidity_data.get('ratios', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            liquidity_ratio = liquidity_ratios.get('liquidity_ratio', 0)
            st.metric(
                "Liquidity Ratio",
                f"{liquidity_ratio*100:.2f}%",
                help="Minimum requirement: 15%"
            )
        
        with col2:
            lcr = liquidity_ratios.get('lcr', 0)
            st.metric(
                "Liquidity Coverage Ratio",
                f"{lcr*100:.1f}%",
                help="High-quality liquid assets to net cash outflows"
            )
        
        with col3:
            nsfr = liquidity_ratios.get('nsfr', 0)
            st.metric(
                "Net Stable Funding Ratio",
                f"{nsfr*100:.1f}%",
                help="Available stable funding to required stable funding"
            )
        
        # Liquidity maturity ladder
        maturity_ladder = liquidity_data.get('maturity_ladder', {})
        
        if maturity_ladder:
            maturity_df = pd.DataFrame({
                'Time Bucket': list(maturity_ladder.keys()),
                'Net Gap (KES)': list(maturity_ladder.values())
            })
            
            fig = px.bar(
                maturity_df,
                x='Time Bucket',
                y='Net Gap (KES)',
                title="Liquidity Maturity Gap Analysis",
                color='Net Gap (KES)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_export_options(self, returns_data):
        """Render export options for SASRA returns"""
        st.markdown("---")
        st.subheader("📤 Export SASRA Returns")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export to Excel", use_container_width=True):
                # Generate Excel file
                excel_file = self.sasra_generator.export_to_excel(returns_data)
                st.success("Excel file generated successfully!")
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_file,
                    file_name=f"SASRA_Returns_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
        
        with col2:
            if st.button("📄 Export to PDF", use_container_width=True):
                # Generate PDF report
                pdf_file = self.sasra_generator.export_to_pdf(returns_data)
                st.success("PDF report generated successfully!")
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_file,
                    file_name=f"SASRA_Returns_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            if st.button("🔍 Pre-submission Check", use_container_width=True):
                validation_results = self.sasra_generator.validate_returns(returns_data)
                self.render_validation_results(validation_results)
    
    def render_validation_results(self, validation_results):
        """Render pre-submission validation results"""
        st.markdown("#### ✅ Pre-submission Validation Results")
        
        checks_passed = validation_results.get('checks_passed', 0)
        total_checks = validation_results.get('total_checks', 0)
        issues = validation_results.get('issues', [])
        
        st.metric(
            "Validation Score",
            f"{checks_passed}/{total_checks}",
            f"{(checks_passed/total_checks)*100:.1f}%"
        )
        
        if issues:
            st.error("**Critical Issues Found:**")
            for issue in issues:
                st.write(f"❌ {issue}")
        else:
            st.success("✅ All validation checks passed! Returns are ready for submission.")
    
    def render_submission_tracker(self):
        """Render SASRA submission tracker"""
        st.markdown("---")
        st.subheader("📅 SASRA Submission Tracker")
        
        submission_data = pd.DataFrame({
            'Period': ['Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022'],
            'Submission Date': ['2024-01-15', '2023-10-16', '2023-07-17', '2023-04-15', '2023-01-16'],
            'Status': ['Submitted', 'Submitted', 'Submitted', 'Submitted', 'Submitted'],
            'SASRA Ack': ['Received', 'Received', 'Received', 'Received', 'Received'],
            'Follow-up Required': ['No', 'No', 'No', 'No', 'No'],
            'Remarks': ['On time', 'On time', '2 days late', 'On time', 'On time']
        })
        
        st.dataframe(submission_data, use_container_width=True)
        
        # Submission performance
        on_time_rate = (len(submission_data[submission_data['Remarks'] == 'On time']) / len(submission_data)) * 100
        st.metric("On-time Submission Rate", f"{on_time_rate:.1f}%")
    
    def render_compliance_monitoring(self):
        """Render SASRA compliance monitoring"""
        st.markdown("---")
        st.subheader("⚖️ SASRA Compliance Monitoring")
        
        compliance_areas = [
            {
                'Area': 'Capital Adequacy',
                'Requirement': 'Minimum 10% core capital',
                'Current': '12.5%',
                'Status': 'Compliant',
                'Trend': 'Stable'
            },
            {
                'Area': 'Liquidity Ratio', 
                'Requirement': 'Minimum 15% liquidity ratio',
                'Current': '18.2%',
                'Status': 'Compliant',
                'Trend': 'Improving'
            },
            {
                'Area': 'Single Employer Limit',
                'Requirement': 'Maximum 25% exposure',
                'Current': '22.8%', 
                'Status': 'Compliant',
                'Trend': 'Watch'
            },
            {
                'Area': 'NPL Ratio',
                'Requirement': 'Maximum 8% NPL ratio',
                'Current': '4.2%',
                'Status': 'Compliant', 
                'Trend': 'Improving'
            },
            {
                'Area': 'Provisioning Coverage',
                'Requirement': 'Adequate loan loss provisions',
                'Current': '85%',
                'Status': 'Compliant',
                'Trend': 'Stable'
            }
        ]
        
        compliance_df = pd.DataFrame(compliance_areas)
        
        # Color code status
        def color_compliance_status(status):
            if status == 'Compliant':
                return 'color: green; font-weight: bold'
            else:
                return 'color: red; font-weight: bold'
        
        styled_compliance = compliance_df.style.applymap(
            color_compliance_status, subset=['Status']
        )
        
        st.dataframe(styled_compliance, use_container_width=True)
    
    def run(self):
        """Run the SASRA returns page"""
        st.title("📋 SASRA Regulatory Returns")
        
        st.markdown("""
        Automated generation of SASRA prudential returns, compliance monitoring, and regulatory reporting 
        for Deposit-Taking SACCOs in Kenya. Ensure timely and accurate submissions to maintain regulatory compliance.
        """)
        
        self.render_sasra_dashboard()
        self.render_returns_generator()
        self.render_submission_tracker()
        self.render_compliance_monitoring()

if __name__ == "__main__":
    page = SASRAReturnsPage()
    page.run()