# pages/08_Operations_TAT.py
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
from sacco_core.analytics.operations_tat import OperationsTATAnalyzer

st.set_page_config(
    page_title="Operations TAT",
    page_icon="⏱️",
    layout="wide"
)

class OperationsTATPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.tat_analyzer = OperationsTATAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "08_Operations_TAT.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "operations_tat_page"
        )
        return True
    
    def render_tat_dashboard(self):
        """Render operations TAT dashboard"""
        st.subheader("⏱️ Operations Turnaround Time Dashboard")
        
        try:
            # Get TAT analysis
            analysis = self.tat_analyzer.analyze_operations_tat()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overall_compliance = analysis.get('sla_compliance', {}).get('overall_compliance_rate', 0) * 100
                st.metric(
                    "Overall SLA Compliance",
                    f"{overall_compliance:.1f}%",
                    help="Percentage of operations meeting SLA targets"
                )
            
            with col2:
                avg_tat = analysis.get('overall_performance', {}).get('average_tat_all_operations', 0)
                st.metric(
                    "Average TAT (All Operations)",
                    f"{avg_tat:.1f} hours",
                    help="Average turnaround time across all operations"
                )
            
            with col3:
                critical_breaches = analysis.get('sla_compliance', {}).get('critical_sla_breaches', 0)
                st.metric(
                    "Critical SLA Breaches",
                    f"{critical_breaches}",
                    delta_color="inverse" if critical_breaches > 0 else "normal",
                    help="Number of critical SLA breaches this month"
                )
            
            with col4:
                performance_trend = analysis.get('overall_performance', {}).get('performance_trend', 'Unknown')
                trend_icon = "📈" if performance_trend == 'Improving' else "📉" if performance_trend == 'Declining' else "➡️"
                st.metric(
                    "Performance Trend",
                    f"{trend_icon} {performance_trend}",
                    help="Overall TAT performance trend"
                )
            
            # TAT overview
            self.render_tat_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering TAT dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_tat_overview(self, analysis):
        """Render TAT overview across operations"""
        st.markdown("#### 📊 Turnaround Time Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan operations TAT
            loan_tat = analysis.get('loan_operations', {}).get('application_approval_tat', {})
            service_tat = analysis.get('service_operations', {}).get('overall_service_tat', {})
            transaction_tat = analysis.get('transaction_operations', {}).get('overall_transaction_tat', {})
            
            tat_data = {
                'Operation': ['Loan Approval', 'Service Requests', 'Transactions'],
                'Average_TAT': [
                    loan_tat.get('average', 0),
                    service_tat.get('average', 0), 
                    transaction_tat.get('average', 0)
                ],
                'SLA_Compliance': [
                    loan_tat.get('sla_compliance_rate', 0) * 100,
                    service_tat.get('sla_compliance_rate', 0) * 100,
                    transaction_tat.get('sla_compliance_rate', 0) * 100
                ]
            }
            
            tat_df = pd.DataFrame(tat_data)
            
            fig = px.bar(
                tat_df,
                x='Operation',
                y='Average_TAT',
                title="Average Turnaround Time by Operation (Hours)",
                color='SLA_Compliance',
                color_continuous_scale='RdYlGn',
                labels={'Average_TAT': 'Average TAT (Hours)', 'SLA_Compliance': 'SLA Compliance %'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # SLA compliance trend
            monthly_trend = analysis.get('sla_compliance', {}).get('monthly_trend', {})
            if monthly_trend:
                months = list(monthly_trend.keys())
                compliance_rates = [rate * 100 for rate in monthly_trend.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months,
                    y=compliance_rates,
                    name='SLA Compliance',
                    line=dict(color='green', width=3),
                    mode='lines+markers'
                ))
                fig.add_hline(
                    y=90,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Target (90%)"
                )
                fig.update_layout(
                    title="Monthly SLA Compliance Trend",
                    xaxis_title="Month",
                    yaxis_title="SLA Compliance Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trend data available")
        
        # Detailed analysis sections
        self.render_loan_operations_analysis(analysis)
        self.render_service_operations_analysis(analysis) 
        self.render_transaction_analysis(analysis)
    
    def render_loan_operations_analysis(self, analysis):
        """Render detailed loan operations TAT analysis"""
        st.markdown("---")
        st.subheader("🏦 Loan Operations TAT")
        
        loan_analysis = analysis.get('loan_operations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Loan Process TAT")
            
            # Application to approval TAT
            approval_tat = loan_analysis.get('application_approval_tat', {})
            disbursement_tat = loan_analysis.get('approval_disbursement_tat', {})
            
            process_data = {
                'Process_Stage': ['Application to Approval', 'Approval to Disbursement'],
                'Average_TAT': [
                    approval_tat.get('average', 0),
                    disbursement_tat.get('average', 0)
                ],
                'SLA_Compliance': [
                    approval_tat.get('sla_compliance_rate', 0) * 100,
                    disbursement_tat.get('sla_compliance_rate', 0) * 100
                ]
            }
            
            process_df = pd.DataFrame(process_data)
            
            fig = px.bar(
                process_df,
                x='Process_Stage',
                y='Average_TAT',
                title="Loan Process Stage TAT (Hours)",
                color='SLA_Compliance',
                color_continuous_scale='RdYlGn',
                labels={'Average_TAT': 'Average TAT (Hours)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Performance by Product")
            
            product_tat = loan_analysis.get('by_product', {})
            if product_tat and 'mean' in product_tat:
                products = list(product_tat['mean'].keys())
                avg_tat = list(product_tat['mean'].values())
                
                fig = px.bar(
                    x=products,
                    y=avg_tat,
                    title="Average TAT by Loan Product (Hours)",
                    labels={'x': 'Product Type', 'y': 'Average TAT (Hours)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No product-wise TAT data available")
        
        # Branch performance
        self.render_branch_performance(loan_analysis)
    
    def render_branch_performance(self, loan_analysis):
        """Render branch performance analysis"""
        st.markdown("#### 🌍 Branch Performance Comparison")
        
        branch_tat = loan_analysis.get('by_branch', {})
        if branch_tat and 'mean' in branch_tat:
            branches = list(branch_tat['mean'].keys())
            avg_tat = list(branch_tat['mean'].values())
            volumes = list(branch_tat['count'].values())
            
            fig = px.scatter(
                x=avg_tat,
                y=volumes,
                size=volumes,
                color=branches,
                hover_name=branches,
                title="Branch Performance: TAT vs Volume",
                labels={'x': 'Average TAT (Hours)', 'y': 'Loan Volume'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No branch performance data available")
    
    def render_service_operations_analysis(self, analysis):
        """Render service operations TAT analysis"""
        st.markdown("---")
        st.subheader("📞 Service Operations TAT")
        
        service_analysis = analysis.get('service_operations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 TAT by Request Type")
            
            request_type_tat = service_analysis.get('by_request_type', {})
            if request_type_tat and 'mean' in request_type_tat:
                request_types = list(request_type_tat['mean'].keys())
                avg_tat = list(request_type_tat['mean'].values())
                
                fig = px.bar(
                    x=request_types,
                    y=avg_tat,
                    title="Average TAT by Request Type (Hours)",
                    labels={'x': 'Request Type', 'y': 'Average TAT (Hours)'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No request type TAT data available")
        
        with col2:
            st.markdown("#### 🚨 Priority-wise Performance")
            
            priority_tat = service_analysis.get('by_priority', {})
            if priority_tat and 'mean' in priority_tat:
                priorities = list(priority_tat['mean'].keys())
                avg_tat = list(priority_tat['mean'].values())
                
                fig = px.bar(
                    x=priorities,
                    y=avg_tat,
                    title="Average TAT by Priority Level (Hours)",
                    labels={'x': 'Priority', 'y': 'Average TAT (Hours)'},
                    color=priorities,
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No priority-wise TAT data available")
    
    def render_transaction_analysis(self, analysis):
        """Render transaction processing TAT analysis"""
        st.markdown("---")
        st.subheader("💳 Transaction Processing TAT")
        
        transaction_analysis = analysis.get('transaction_operations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔄 TAT by Transaction Type")
            
            transaction_type_tat = transaction_analysis.get('by_transaction_type', {})
            if transaction_type_tat and 'mean' in transaction_type_tat:
                transaction_types = list(transaction_type_tat['mean'].keys())
                avg_tat = list(transaction_type_tat['mean'].values())
                
                fig = px.bar(
                    x=transaction_types,
                    y=avg_tat,
                    title="Average TAT by Transaction Type (Hours)",
                    labels={'x': 'Transaction Type', 'y': 'Average TAT (Hours)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transaction type TAT data available")
        
        with col2:
            st.markdown("#### 📱 Channel Performance")
            
            channel_tat = transaction_analysis.get('by_channel', {})
            if channel_tat and 'mean' in channel_tat:
                channels = list(channel_tat['mean'].keys())
                avg_tat = list(channel_tat['mean'].values())
                
                fig = px.pie(
                    names=channels,
                    values=avg_tat,
                    title="TAT Distribution by Processing Channel",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No channel performance data available")
    
    def render_bottleneck_analysis(self):
        """Render process bottleneck analysis"""
        st.markdown("---")
        st.subheader("🔍 Process Bottleneck Analysis")
        
        try:
            analysis = self.tat_analyzer.analyze_operations_tat()
            bottlenecks = analysis.get('bottleneck_analysis', [])
            
            if bottlenecks:
                for bottleneck in bottlenecks:
                    with st.expander(f"🚧 {bottleneck.get('process', 'Unknown Process')} - {bottleneck.get('bottleneck_stage', 'Unknown Stage')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Average Delay",
                                f"{bottleneck.get('average_delay_hours', 0):.1f} hours"
                            )
                        
                        with col2:
                            impact = bottleneck.get('impact_level', 'Medium')
                            st.metric("Impact Level", impact)
                        
                        with col3:
                            st.write("**Recommendation:**")
                            st.info(bottleneck.get('recommendation', 'No recommendation available'))
            else:
                st.success("✅ No significant bottlenecks identified")
                
        except Exception as e:
            st.error(f"Error rendering bottleneck analysis: {str(e)}")
    
    def render_improvement_recommendations(self):
        """Render improvement recommendations"""
        st.markdown("---")
        st.subheader("🎯 Improvement Recommendations")
        
        try:
            analysis = self.tat_analyzer.analyze_operations_tat()
            improvement_areas = analysis.get('overall_performance', {}).get('key_improvement_areas', [])
            recommendations = analysis.get('sla_compliance', {}).get('improvement_recommendations', [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Key Improvement Areas")
                if improvement_areas:
                    for area in improvement_areas:
                        st.warning(f"⚠️ {area}")
                else:
                    st.info("No specific improvement areas identified")
            
            with col2:
                st.markdown("#### 💡 Recommended Actions")
                if recommendations:
                    for i, recommendation in enumerate(recommendations, 1):
                        st.success(f"{i}. {recommendation}")
                else:
                    st.info("No specific recommendations available")
                    
        except Exception as e:
            st.error(f"Error rendering improvement recommendations: {str(e)}")
    
    def run(self):
        """Run the operations TAT page"""
        st.title("⏱️ Operations Turnaround Time Monitoring")
        
        st.markdown("""
        Comprehensive monitoring and analysis of turnaround times across all SACCO operations to ensure 
        efficient service delivery, identify bottlenecks, and maintain optimal operational performance.
        """)
        
        try:
            self.render_tat_dashboard()
            self.render_bottleneck_analysis()
            self.render_improvement_recommendations()
        except Exception as e:
            st.error(f"Error running operations TAT page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = OperationsTATPage()
    page.run()