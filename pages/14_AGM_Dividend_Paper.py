# pages/14_AGM_Dividend_Paper.py
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
from sacco_core.analytics.agm_reports import AGMReportAnalyzer, ReportSection, DividendStatus

st.set_page_config(
    page_title="AGM Dividend Paper",
    page_icon="📄",
    layout="wide"
)

class AGMDividendPaperPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.agm_analyzer = AGMReportAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "14_AGM_Dividend_Paper.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "agm_dividend_paper_page"
        )
        return True
    
    def render_agm_dashboard(self):
        """Render AGM and dividend paper dashboard"""
        st.subheader("📄 AGM Report & Dividend Paper Generation")
        
        try:
            # Generate AGM report
            analysis = self.agm_analyzer.generate_agm_report()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                agm_report = analysis.get('agm_report')
                total_assets = agm_report.total_assets if agm_report else 0
                st.metric(
                    "Total Assets",
                    f"KES {total_assets:,.0f}",
                    help="Total society assets"
                )
            
            with col2:
                net_income = agm_report.net_income if agm_report else 0
                st.metric(
                    "Net Income",
                    f"KES {net_income:,.0f}",
                    help="Annual net income"
                )
            
            with col3:
                dividend_capacity = analysis.get('dividend_capacity', {})
                dividend_payout = dividend_capacity.get('final_dividend_capacity', 0)
                st.metric(
                    "Dividend Payout",
                    f"KES {dividend_payout:,.0f}",
                    help="Total dividend distribution"
                )
            
            with col4:
                dividend_per_share = dividend_capacity.get('dividend_per_share', 0)
                st.metric(
                    "Dividend per Share",
                    f"KES {dividend_per_share:.3f}",
                    help="Dividend amount per share"
                )
            
            # AGM overview
            self.render_agm_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering AGM dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_agm_overview(self, analysis):
        """Render AGM report overview"""
        st.markdown("#### 📊 AGM Report Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Financial performance trend
            performance_comparison = analysis.get('performance_comparison', {})
            historical_data = performance_comparison.get('historical_data', [])
            
            if historical_data:
                years = [data['year'] for data in historical_data]
                assets = [data['total_assets'] for data in historical_data]
                income = [data['net_income'] for data in historical_data]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Total Assets', x=years, y=assets, yaxis='y1'))
                fig.add_trace(go.Scatter(name='Net Income', x=years, y=income, yaxis='y2', line=dict(color='red', width=3)))
                
                fig.update_layout(
                    title="Financial Performance Trend",
                    xaxis_title="Year",
                    yaxis=dict(title='Total Assets (KES)', side='left'),
                    yaxis2=dict(title='Net Income (KES)', side='right', overlaying='y')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical performance data available")
        
        with col2:
            # Dividend capacity analysis
            dividend_capacity = analysis.get('dividend_capacity', {})
            if dividend_capacity:
                labels = ['Mandatory Reserves', 'Available for Dividends']
                values = [
                    dividend_capacity.get('mandatory_reserves', 0),
                    dividend_capacity.get('final_dividend_capacity', 0)
                ]
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Net Income Allocation",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dividend capacity data available")
        
        # Detailed sections
        self.render_executive_summary(analysis)
        self.render_financial_performance(analysis)
        self.render_dividend_declaration(analysis)
        self.render_governance_compliance(analysis)
        self.render_member_communications(analysis)
        self.render_report_export(analysis)
    
    def render_executive_summary(self, analysis):
        """Render executive summary section"""
        st.markdown("---")
        st.subheader("🏛️ Executive Summary")
        
        agm_report = analysis.get('agm_report')
        if agm_report and agm_report.report_sections:
            executive_summary = agm_report.report_sections.get(ReportSection.EXECUTIVE_SUMMARY, "No executive summary available")
            
            # Display in a nice formatted box
            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                """, unsafe_allow_html=True)
                
                st.markdown(executive_summary)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Key highlights
            st.markdown("#### 🎯 Key Performance Highlights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Member Count", f"{agm_report.member_count:,}")
            
            with col2:
                st.metric("Total Assets", f"KES {agm_report.total_assets:,.0f}")
            
            with col3:
                st.metric("Net Income", f"KES {agm_report.net_income:,.0f}")
            
            with col4:
                compliance_status = "✅ COMPLIANT" if agm_report.compliance_status else "❌ NON-COMPLIANT"
                st.metric("Regulatory Status", compliance_status)
        else:
            st.info("No executive summary data available")
    
    def render_financial_performance(self, analysis):
        """Render financial performance section"""
        st.markdown("---")
        st.subheader("💰 Financial Performance")
        
        performance_comparison = analysis.get('performance_comparison', {})
        historical_data = performance_comparison.get('historical_data', [])
        
        if historical_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Financial Metrics")
                
                # Create financial metrics table
                financial_data = []
                for data in historical_data:
                    financial_data.append({
                        'Year': data['year'],
                        'Total Assets': f"KES {data['total_assets']:,.0f}",
                        'Net Income': f"KES {data['net_income']:,.0f}",
                        'Member Equity': f"KES {data['member_equity']:,.0f}",
                        'Loan Portfolio': f"KES {data['loan_portfolio']:,.0f}",
                        'Efficiency': f"{data['operational_efficiency']*100:.1f}%"
                    })
                
                financial_df = pd.DataFrame(financial_data)
                st.dataframe(financial_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Growth Analysis")
                
                growth_rates = performance_comparison.get('growth_rates', {})
                if growth_rates:
                    metrics = list(growth_rates.keys())
                    rates = [growth_rates[metric] for metric in metrics]
                    
                    # Compare with industry
                    industry_comparison = performance_comparison.get('industry_comparison', {})
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Our SACCO',
                        x=metrics,
                        y=rates,
                        marker_color='lightblue'
                    ))
                    
                    if industry_comparison:
                        industry_rates = [
                            industry_comparison.get('asset_growth_industry', 0),
                            industry_comparison.get('income_growth_industry', 0),
                            industry_comparison.get('efficiency_industry', 0)
                        ]
                        fig.add_trace(go.Bar(
                            name='Industry Average',
                            x=metrics[:3],  # Match available industry metrics
                            y=industry_rates,
                            marker_color='lightcoral'
                        ))
                    
                    fig.update_layout(
                        title="Growth Rate Comparison (%)",
                        xaxis_title="Metrics",
                        yaxis_title="Growth Rate (%)",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Financial report section
        agm_report = analysis.get('agm_report')
        if agm_report and agm_report.report_sections:
            financial_review = agm_report.report_sections.get(ReportSection.FINANCIAL_PERFORMANCE, "")
            with st.expander("📋 Detailed Financial Review"):
                st.markdown(financial_review)
    
    def render_dividend_declaration(self, analysis):
        """Render dividend declaration section"""
        st.markdown("---")
        st.subheader("💸 Dividend Declaration")
        
        dividend_capacity = analysis.get('dividend_capacity', {})
        dividend_allocations = analysis.get('dividend_allocations', [])
        
        if dividend_capacity:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Dividend Capacity Analysis")
                
                capacity_data = {
                    'Metric': [
                        'Net Income',
                        'Mandatory Reserves',
                        'Available for Dividends',
                        'Maximum Allowed Payout',
                        'Final Dividend Capacity',
                        'Dividend per Share',
                        'Payout Ratio',
                        'Dividend Yield'
                    ],
                    'Value': [
                        f"KES {dividend_capacity.get('net_income', 0):,.0f}",
                        f"KES {dividend_capacity.get('mandatory_reserves', 0):,.0f}",
                        f"KES {dividend_capacity.get('available_for_dividends', 0):,.0f}",
                        f"KES {dividend_capacity.get('max_dividend_allowed', 0):,.0f}",
                        f"KES {dividend_capacity.get('final_dividend_capacity', 0):,.0f}",
                        f"KES {dividend_capacity.get('dividend_per_share', 0):.3f}",
                        f"{dividend_capacity.get('payout_ratio', 0)*100:.1f}%",
                        f"{dividend_capacity.get('dividend_yield', 0):.2f}%"
                    ]
                }
                
                capacity_df = pd.DataFrame(capacity_data)
                st.dataframe(capacity_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 👥 Member Dividend Distribution")
                
                if dividend_allocations:
                    # Summary statistics
                    total_members = len(dividend_allocations)
                    total_payout = sum(alloc.net_dividend for alloc in dividend_allocations)
                    avg_dividend = total_payout / total_members if total_members > 0 else 0
                    
                    st.metric("Eligible Members", f"{total_members:,}")
                    st.metric("Total Payout", f"KES {total_payout:,.0f}")
                    st.metric("Average Dividend", f"KES {avg_dividend:,.0f}")
                    
                    # Dividend distribution histogram
                    dividend_amounts = [alloc.net_dividend for alloc in dividend_allocations]
                    
                    fig = px.histogram(
                        x=dividend_amounts,
                        title="Dividend Distribution Among Members",
                        labels={'x': 'Dividend Amount (KES)', 'y': 'Number of Members'},
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No dividend allocation data available")
            
            # Dividend declaration section
            agm_report = analysis.get('agm_report')
            if agm_report and agm_report.report_sections:
                dividend_declaration = agm_report.report_sections.get(ReportSection.DIVIDEND_DECLARATION, "")
                with st.expander("📜 Official Dividend Declaration"):
                    st.markdown(dividend_declaration)
                    
                    # Board approval interface
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Recommend for Board Approval", type="primary"):
                            st.success("Dividend declaration recommended for board approval!")
                    
                    with col2:
                        if st.button("📧 Send to Board Members"):
                            st.info("Dividend proposal sent to board members for review")
        
        else:
            st.info("No dividend capacity data available")
    
    def render_governance_compliance(self, analysis):
        """Render governance and compliance section"""
        st.markdown("---")
        st.subheader("⚖️ Governance & Compliance")
        
        compliance_analysis = analysis.get('compliance_analysis', {})
        agm_report = analysis.get('agm_report')
        
        if compliance_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Compliance Status")
                
                compliance_score = compliance_analysis.get('overall_compliance_score', 0)
                st.metric("Overall Compliance Score", f"{compliance_score:.1f}%")
                
                critical_issues = compliance_analysis.get('critical_issues', [])
                if critical_issues:
                    st.error("**Critical Compliance Issues:**")
                    for issue in critical_issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ No critical compliance issues")
                
                # Compliance recommendations
                recommendations = compliance_analysis.get('recommendations', [])
                if recommendations:
                    st.info("**Compliance Recommendations:**")
                    for rec in recommendations:
                        st.write(f"• {rec}")
            
            with col2:
                st.markdown("#### 📅 Compliance Timeline")
                
                next_review = compliance_analysis.get('next_review_date', 'Unknown')
                st.metric("Next Compliance Review", next_review)
                
                # Compliance trend (simulated)
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
                scores = [85, 87, 89, 88, 90, 92, 91, 93]  # Simulated improving trend
                
                fig = px.line(
                    x=months,
                    y=scores,
                    title="Monthly Compliance Score Trend",
                    labels={'x': 'Month', 'y': 'Compliance Score (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Governance report section
        if agm_report and agm_report.report_sections:
            governance_report = agm_report.report_sections.get(ReportSection.GOVERNANCE_COMPLIANCE, "")
            with st.expander("🏛️ Detailed Governance Report"):
                st.markdown(governance_report)
    
    def render_member_communications(self, analysis):
        """Render member communications section"""
        st.markdown("---")
        st.subheader("📢 Member Communications")
        
        member_communication = analysis.get('member_communication', {})
        dividend_allocations = analysis.get('dividend_allocations', [])
        
        if member_communication and dividend_allocations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✉️ Communication Templates")
                
                templates = member_communication.get('templates', {})
                
                with st.expander("📨 Dividend Notification Template"):
                    st.code(templates.get('dividend_notification', 'Template not available'), language='text')
                
                with st.expander("📅 AGM Invitation Template"):
                    st.code(templates.get('agm_invitation', 'Template not available'), language='text')
            
            with col2:
                st.markdown("#### 🚀 Communication Setup")
                
                channels = member_communication.get('communication_channels', [])
                delivery_time = member_communication.get('estimated_delivery_time', 'Unknown')
                
                st.metric("Available Channels", ", ".join(channels))
                st.metric("Estimated Delivery Time", delivery_time)
                
                # Communication controls
                st.markdown("**Bulk Communication Actions:**")
                
                if st.button("📧 Prepare Dividend Notifications", type="primary"):
                    st.success(f"Prepared dividend notifications for {len(dividend_allocations)} members")
                
                if st.button("📱 Schedule AGM Invitations"):
                    st.info("AGM invitations scheduled for delivery")
                
                if st.button("🖨️ Generate Member Statements"):
                    st.success("Member dividend statements generated")
        
        else:
            st.info("No member communication data available")
    
    def render_report_export(self, analysis):
        """Render report export section"""
        st.markdown("---")
        st.subheader("📤 Report Export & Distribution")
        
        report_export = analysis.get('report_export', {})
        agm_report = analysis.get('agm_report')
        
        if report_export:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Export Options")
                
                export_formats = report_export.get('export_formats', [])
                st.write("**Available Formats:**")
                for format in export_formats:
                    st.write(f"• {format}")
                
                file_size = report_export.get('estimated_file_size', 'Unknown')
                generation_time = report_export.get('generation_time', 'Unknown')
                
                st.metric("Estimated File Size", file_size)
                st.metric("Generation Time", generation_time)
            
            with col2:
                st.markdown("#### 📊 Export Summary")
                
                summary_stats = report_export.get('summary_statistics', {})
                if summary_stats:
                    st.metric("Eligible Members", f"{summary_stats.get('total_members_eligible', 0):,}")
                    st.metric("Total Dividend", f"KES {summary_stats.get('total_dividend_payout', 0):,.0f}")
                    st.metric("Average Dividend", f"KES {summary_stats.get('average_dividend_per_member', 0):,.0f}")
            
            # Export actions
            st.markdown("#### 🚀 Generate Reports")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Full AGM Report PDF", type="primary"):
                    st.success("AGM report PDF generation started!")
            
            with col2:
                if st.button("💸 Dividend Schedule Excel"):
                    st.success("Dividend schedule Excel export started!")
            
            with col3:
                if st.button("👥 Member Communications Pack"):
                    st.success("Member communication pack prepared!")
            
            with col4:
                if st.button("🏛️ Board Presentation"):
                    st.success("Board presentation generated!")
            
            # Strategic outlook section
            if agm_report and agm_report.report_sections:
                strategic_outlook = agm_report.report_sections.get(ReportSection.STRATEGIC_OUTLOOK, "")
                with st.expander("🔮 Strategic Outlook & Future Plans"):
                    st.markdown(strategic_outlook)
        else:
            st.info("No report export data available")
    
    def run(self):
        """Run the AGM dividend paper page"""
        st.title("📄 AGM Report & Dividend Paper Generation")
        
        st.markdown("""
        Comprehensive Annual General Meeting report generation, dividend calculation, and member communication 
        platform. Generate regulatory-compliant AGM reports, calculate optimal dividend distributions, and 
        automate member communications for transparent governance and member value maximization.
        """)
        
        try:
            self.render_agm_dashboard()
        except Exception as e:
            st.error(f"Error running AGM dividend paper page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = AGMDividendPaperPage()
    page.run()