# pages/13A_Collections_Funnel_SMS.py
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
from sacco_core.analytics.sms_automation import SMSAutomationAnalyzer, SMSStage, MessageTemplate, DeliveryStatus

st.set_page_config(
    page_title="Collections SMS Automation",
    page_icon="📱",
    layout="wide"
)

class CollectionsSMSAutomationPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.sms_analyzer = SMSAutomationAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "13A_Collections_Funnel_SMS.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "collections_sms_automation_page"
        )
        return True
    
    def render_sms_dashboard(self):
        """Render SMS automation dashboard"""
        st.subheader("📱 Collections SMS Automation & Funnel Management")
        
        try:
            # Get SMS automation analysis
            analysis = self.sms_analyzer.analyze_sms_automation()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cost_analysis = analysis.get('cost_analysis', {})
                total_cost = cost_analysis.get('total_cost', 0)
                st.metric(
                    "Total SMS Cost",
                    f"KES {total_cost:,.0f}",
                    help="Total cost of SMS campaigns"
                )
            
            with col2:
                total_messages = cost_analysis.get('total_messages', 0)
                st.metric(
                    "Messages Sent",
                    f"{total_messages}",
                    help="Total SMS messages sent"
                )
            
            with col3:
                total_roi = cost_analysis.get('total_roi', 0) * 100
                st.metric(
                    "Overall ROI",
                    f"{total_roi:.1f}%",
                    help="Return on investment from SMS campaigns"
                )
            
            with col4:
                compliance_analysis = analysis.get('compliance_analysis', {})
                compliance_rate = compliance_analysis.get('compliance_rate', 0) * 100
                st.metric(
                    "Compliance Rate",
                    f"{compliance_rate:.1f}%",
                    help="Percentage of compliant messages"
                )
            
            # SMS automation overview
            self.render_sms_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering SMS dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_sms_overview(self, analysis):
        """Render SMS automation overview"""
        st.markdown("#### 📊 SMS Funnel Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Funnel stage performance
            funnel_analysis = analysis.get('funnel_analysis', {})
            if funnel_analysis:
                stages = [funnel.stage.value for funnel in funnel_analysis.values()]
                response_rates = [funnel.response_rate * 100 for funnel in funnel_analysis.values()]
                
                fig = px.bar(
                    x=stages,
                    y=response_rates,
                    title="Response Rate by Funnel Stage (%)",
                    labels={'x': 'Funnel Stage', 'y': 'Response Rate (%)'},
                    color=response_rates,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No funnel analysis data available")
        
        with col2:
            # Campaign performance
            campaign_performance = analysis.get('campaign_performance', {})
            if campaign_performance:
                campaigns = [data['campaign_name'] for data in campaign_performance.values()]
                roi_values = [data.get('roi', 0) * 100 for data in campaign_performance.values()]
                
                fig = px.bar(
                    x=campaigns,
                    y=roi_values,
                    title="Campaign ROI Performance (%)",
                    labels={'x': 'Campaign', 'y': 'ROI (%)'},
                    color=roi_values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No campaign performance data available")
        
        # Detailed analysis sections
        self.render_funnel_analysis(analysis)
        self.render_campaign_management(analysis)
        self.render_response_optimization(analysis)
        self.render_compliance_monitoring(analysis)
        self.render_cost_analysis(analysis)
    
    def render_funnel_analysis(self, analysis):
        """Render detailed funnel analysis"""
        st.markdown("---")
        st.subheader("🔄 SMS Conversation Funnel Analysis")
        
        funnel_analysis = analysis.get('funnel_analysis', {})
        
        if funnel_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Funnel Metrics by Stage")
                
                funnel_data = []
                for stage, funnel in funnel_analysis.items():
                    funnel_data.append({
                        'Stage': stage.value,
                        'Messages Sent': funnel.messages_sent,
                        'Messages Delivered': funnel.messages_delivered,
                        'Responses Received': funnel.responses_received,
                        'Response Rate': f"{funnel.response_rate * 100:.1f}%",
                        'Conversion Rate': f"{funnel.conversion_rate * 100:.1f}%",
                        'Avg Response Time': f"{funnel.average_response_time:.1f}h"
                    })
                
                funnel_df = pd.DataFrame(funnel_data)
                st.dataframe(funnel_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Funnel Visualization")
                
                # Create funnel chart
                stages = [funnel.stage.value for funnel in funnel_analysis.values()]
                messages = [funnel.messages_sent for funnel in funnel_analysis.values()]
                responses = [funnel.responses_received for funnel in funnel_analysis.values()]
                conversions = [int(funnel.responses_received * funnel.conversion_rate) for funnel in funnel_analysis.values()]
                
                fig = go.Figure(go.Funnelarea(
                    values=messages,
                    text=stages,
                    title={"position": "top center", "text": "SMS Funnel Performance"}
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Funnel optimization recommendations
            st.markdown("#### 🎯 Funnel Optimization Opportunities")
            recommendations = analysis.get('recommendations', [])
            funnel_recs = [rec for rec in recommendations if rec['type'] == 'Funnel Optimization']
            
            if funnel_recs:
                for rec in funnel_recs:
                    with st.expander(f"🔧 {rec['description']} (Priority: {rec['priority']})"):
                        st.write(f"**Recommended Action:** {rec['action']}")
                        st.write(f"**Expected Impact:** {rec['expected_impact']}")
                        
                        if st.button("Implement Optimization", key=f"funnel_opt_{rec['description'][:10]}"):
                            st.success(f"Optimization implementation started")
            else:
                st.success("✅ Funnel performance is optimal")
        else:
            st.info("No funnel analysis data available")
    
    def render_campaign_management(self, analysis):
        """Render campaign management interface"""
        st.markdown("---")
        st.subheader("🎯 SMS Campaign Management")
        
        campaign_performance = analysis.get('campaign_performance', {})
        
        if campaign_performance:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Active Campaigns")
                
                active_campaigns = [camp for camp, data in campaign_performance.items() if data.get('status') == 'Active']
                
                if active_campaigns:
                    for campaign_id in active_campaigns[:3]:  # Show top 3 active campaigns
                        data = campaign_performance[campaign_id]
                        with st.expander(f"📧 {data['campaign_name']}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Response Rate", f"{data.get('response_rate', 0) * 100:.1f}%")
                            with col2:
                                st.metric("Conversion Rate", f"{data.get('conversion_rate', 0) * 100:.1f}%")
                            with col3:
                                st.metric("ROI", f"{data.get('roi', 0) * 100:.1f}%")
                            
                            # Campaign controls
                            if st.button("Pause Campaign", key=f"pause_{campaign_id}"):
                                st.warning(f"Campaign {data['campaign_name']} paused")
                            if st.button("View Details", key=f"details_{campaign_id}"):
                                st.info(f"Showing details for {data['campaign_name']}")
                else:
                    st.info("No active campaigns")
            
            with col2:
                st.markdown("#### 🚀 Create New Campaign")
                
                with st.form("new_campaign_form"):
                    campaign_name = st.text_input("Campaign Name")
                    target_segment = st.selectbox("Target Segment", 
                                                ['Delinquent 1-30', 'Delinquent 31-60', 'Delinquent 61-90', 'Legal'])
                    message_template = st.selectbox("Message Template", [t.value for t in MessageTemplate])
                    scheduled_time = st.date_input("Schedule Date")
                    
                    if st.form_submit_button("Create Campaign"):
                        st.success(f"Campaign '{campaign_name}' created successfully!")
        
        # Campaign performance details
        st.markdown("#### 📊 Campaign Performance Details")
        if campaign_performance:
            performance_data = []
            for campaign_id, data in campaign_performance.items():
                performance_data.append({
                    'Campaign ID': campaign_id,
                    'Campaign Name': data['campaign_name'],
                    'Messages Sent': data['messages_sent'],
                    'Response Rate': f"{data.get('response_rate', 0) * 100:.1f}%",
                    'Conversion Rate': f"{data.get('conversion_rate', 0) * 100:.1f}%",
                    'Total Cost': f"KES {data.get('total_cost', 0):,.0f}",
                    'ROI': f"{data.get('roi', 0) * 100:.1f}%",
                    'Status': data.get('status', 'Unknown')
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
        else:
            st.info("No campaign performance data available")
    
    def render_response_optimization(self, analysis):
        """Render response optimization analysis"""
        st.markdown("---")
        st.subheader("📈 Response Rate Optimization")
        
        response_optimization = analysis.get('response_optimization', {})
        
        if response_optimization:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Template Performance")
                
                template_performance = response_optimization.get('template_performance', {})
                if template_performance:
                    templates = list(template_performance.keys())
                    response_rates = [rate * 100 for rate in template_performance.values()]
                    
                    fig = px.bar(
                        x=templates,
                        y=response_rates,
                        title="Response Rate by Template Type (%)",
                        labels={'x': 'Template Type', 'y': 'Response Rate (%)'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### ⏰ Send Time Optimization")
                
                time_performance = response_optimization.get('time_performance', {})
                if time_performance:
                    times = list(time_performance.keys())
                    rates = [rate * 100 for rate in time_performance.values()]
                    
                    fig = px.line(
                        x=times,
                        y=rates,
                        title="Response Rate by Send Time",
                        labels={'x': 'Time of Day', 'y': 'Response Rate (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Best performing templates and times
            st.markdown("#### 🏆 Best Performing Elements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Templates:**")
                best_templates = response_optimization.get('best_performing_templates', [])
                for template, rate in best_templates:
                    st.metric(template, f"{rate * 100:.1f}%")
            
            with col2:
                st.markdown("**Optimal Send Times:**")
                optimal_times = response_optimization.get('optimal_send_times', [])
                for time, rate in optimal_times:
                    st.metric(time, f"{rate * 100:.1f}%")
        
        # A/B Testing Interface
        st.markdown("#### 🔬 A/B Testing")
        
        with st.expander("Create A/B Test"):
            col1, col2 = st.columns(2)
            
            with col1:
                test_name = st.text_input("Test Name")
                test_duration = st.slider("Test Duration (days)", 7, 30, 14)
                sample_size = st.number_input("Sample Size per Variant", 100, 5000, 500)
            
            with col2:
                variant_a = st.text_area("Variant A Message", "Friendly reminder: Your payment is due. Thank you!")
                variant_b = st.text_area("Variant B Message", "Urgent: Payment required to avoid late fees. Reply for help.")
            
            if st.button("Start A/B Test"):
                st.success(f"A/B test '{test_name}' started with {sample_size} messages per variant")
    
    def render_compliance_monitoring(self, analysis):
        """Render compliance monitoring section"""
        st.markdown("---")
        st.subheader("⚖️ Compliance Monitoring")
        
        compliance_analysis = analysis.get('compliance_analysis', {})
        
        if compliance_analysis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                opt_out_count = compliance_analysis.get('opt_out_count', 0)
                st.metric("Opt-out Requests", opt_out_count)
            
            with col2:
                opt_out_rate = compliance_analysis.get('opt_out_rate', 0) * 100
                st.metric("Opt-out Rate", f"{opt_out_rate:.2f}%")
            
            with col3:
                compliance_rate = compliance_analysis.get('compliance_rate', 0) * 100
                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
            
            with col4:
                failure_rate = compliance_analysis.get('failure_rate', 0) * 100
                st.metric("Delivery Failure Rate", f"{failure_rate:.1f}%")
            
            # Compliance issues
            st.markdown("#### 📋 Compliance Issues")
            compliance_issues = compliance_analysis.get('compliance_issues', [])
            if compliance_issues:
                for issue in compliance_issues:
                    st.error(issue)
                
                st.warning("**Immediate Action Required:** Review and update non-compliant messages")
            else:
                st.success("✅ No compliance issues detected")
            
            # Opt-out trend
            st.markdown("#### 📉 Opt-out Trend")
            opt_out_trend = compliance_analysis.get('opt_out_trend', {})
            if opt_out_trend:
                months = list(opt_out_trend.keys())
                counts = list(opt_out_trend.values())
                
                fig = px.line(
                    x=months,
                    y=counts,
                    title="Monthly Opt-out Trend",
                    labels={'x': 'Month', 'y': 'Opt-out Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No compliance analysis data available")
    
    def render_cost_analysis(self, analysis):
        """Render cost analysis and ROI"""
        st.markdown("---")
        st.subheader("💰 Cost Analysis & ROI")
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        if cost_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Cost Metrics")
                
                cost_data = {
                    'Metric': [
                        'Total Messages',
                        'Total Cost',
                        'Cost per Message',
                        'Cost per Response',
                        'Cost per Conversion',
                        'Estimated Revenue',
                        'Overall ROI',
                        'Break-even Point'
                    ],
                    'Value': [
                        cost_analysis.get('total_messages', 0),
                        f"KES {cost_analysis.get('total_cost', 0):,.0f}",
                        f"KES {cost_analysis.get('cost_per_message', 0):,.2f}",
                        f"KES {cost_analysis.get('cost_per_response', 0):,.2f}",
                        f"KES {cost_analysis.get('cost_per_conversion', 0):,.2f}",
                        f"KES {cost_analysis.get('estimated_revenue', 0):,.0f}",
                        f"{cost_analysis.get('total_roi', 0) * 100:.1f}%",
                        f"{cost_analysis.get('break_even_point', 0):.0f} messages"
                    ]
                }
                
                cost_df = pd.DataFrame(cost_data)
                st.dataframe(cost_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Cost Trend")
                
                cost_trend = cost_analysis.get('cost_trend', {})
                if cost_trend:
                    months = list(cost_trend.keys())
                    costs = list(cost_trend.values())
                    
                    fig = px.line(
                        x=months,
                        y=costs,
                        title="Monthly SMS Cost Trend",
                        labels={'x': 'Month', 'y': 'Cost (KES)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Cost optimization recommendations
            st.markdown("#### 🎯 Cost Optimization")
            recommendations = analysis.get('recommendations', [])
            cost_recs = [rec for rec in recommendations if rec['type'] == 'Cost Optimization']
            
            if cost_recs:
                for rec in cost_recs:
                    with st.expander(f"💸 {rec['description']} (Priority: {rec['priority']})"):
                        st.write(f"**Recommended Action:** {rec['action']}")
                        st.write(f"**Expected Impact:** {rec['expected_impact']}")
            else:
                st.success("✅ Cost efficiency is optimal")
        else:
            st.info("No cost analysis data available")
    
    def run(self):
        """Run the collections SMS automation page"""
        st.title("📱 Collections SMS Automation & Funnel Management")
        
        st.markdown("""
        Intelligent SMS automation platform for collections management, featuring multi-stage conversation funnels, 
        A/B testing, compliance monitoring, and ROI optimization to maximize recovery rates while minimizing costs 
        through data-driven message personalization and send time optimization.
        """)
        
        try:
            self.render_sms_dashboard()
        except Exception as e:
            st.error(f"Error running SMS automation page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = CollectionsSMSAutomationPage()
    page.run()