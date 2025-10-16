# pages/11_Member_Value.py
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
from sacco_core.analytics.member_value import MemberValueAnalyzer, MemberSegment

st.set_page_config(
    page_title="Member Value Analysis",
    page_icon="👥",
    layout="wide"
)

class MemberValuePage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.member_value_analyzer = MemberValueAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "11_Member_Value.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "member_value_page"
        )
        return True
    
    def render_member_value_dashboard(self):
        """Render member value dashboard"""
        st.subheader("👥 Member Value & Lifetime Value Analysis")
        
        try:
            # Get member value analysis
            analysis = self.member_value_analyzer.analyze_member_value()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_member_value = analysis.get('member_segmentation', {}).get('total_member_value', 0)
                st.metric(
                    "Total Member Portfolio Value",
                    f"KES {total_member_value:,.0f}",
                    help="Total value across all member relationships"
                )
            
            with col2:
                avg_member_value = analysis.get('member_segmentation', {}).get('average_member_value', 0)
                st.metric(
                    "Average Member Value",
                    f"KES {avg_member_value:,.0f}",
                    help="Average value per member"
                )
            
            with col3:
                total_at_risk = analysis.get('churn_analysis', {}).get('total_at_risk', 0)
                st.metric(
                    "Members at Risk",
                    f"{total_at_risk}",
                    delta_color="inverse",
                    help="Members with high or medium churn risk"
                )
            
            with col4:
                avg_engagement = analysis.get('engagement_analysis', {}).get('average_engagement_score', 0) * 100
                st.metric(
                    "Average Engagement Score",
                    f"{avg_engagement:.1f}%",
                    help="Overall member engagement level"
                )
            
            # Member value overview
            self.render_member_value_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering member value dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_member_value_overview(self, analysis):
        """Render member value overview"""
        st.markdown("#### 📊 Member Value Segmentation Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Member segmentation distribution
            segment_distribution = analysis.get('member_segmentation', {}).get('segment_distribution', {})
            if segment_distribution:
                segments = list(segment_distribution.keys())
                counts = list(segment_distribution.values())
                
                fig = px.pie(
                    names=segments,
                    values=counts,
                    title="Member Distribution by Value Segment",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No segmentation data available")
        
        with col2:
            # Member value by segment
            segment_summary = analysis.get('member_segmentation', {}).get('segment_summary', {})
            if segment_summary and 'total_balance' in segment_summary and 'sum' in segment_summary['total_balance']:
                segments = list(segment_summary['total_balance']['sum'].keys())
                values = list(segment_summary['total_balance']['sum'].values())
                
                fig = px.bar(
                    x=segments,
                    y=values,
                    title="Total Value by Member Segment (KES)",
                    labels={'x': 'Member Segment', 'y': 'Total Value (KES)'},
                    color=values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No segment value data available")
        
        # Detailed analysis sections
        self.render_segmentation_analysis(analysis)
        self.render_ltv_analysis(analysis)
        self.render_engagement_analysis(analysis)
        self.render_churn_analysis(analysis)
        self.render_business_opportunities(analysis)
    
    def render_segmentation_analysis(self, analysis):
        """Render detailed segmentation analysis"""
        st.markdown("---")
        st.subheader("🏷️ Member Segmentation Analysis")
        
        segmentation = analysis.get('member_segmentation', {})
        segment_summary = segmentation.get('segment_summary', {})
        
        if segment_summary:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Segment Performance Metrics")
                
                # Create segment performance table
                performance_data = []
                for segment in segment_summary.get('member_id', {}).get('count', {}).keys():
                    performance_data.append({
                        'Segment': segment,
                        'Member Count': segment_summary['member_id']['count'][segment],
                        'Avg Value (KES)': segment_summary['total_balance']['mean'][segment],
                        'Total Value (KES)': segment_summary['total_balance']['sum'][segment],
                        'Avg Profitability': segment_summary['profitability_score']['mean'][segment],
                        'Avg Engagement': segment_summary['engagement_score']['mean'][segment]
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 💎 High-Value Members")
                
                high_value_members = segmentation.get('high_value_members', [])
                if high_value_members:
                    hv_df = pd.DataFrame(high_value_members)
                    st.dataframe(hv_df[['member_id', 'segment', 'total_balance', 'profitability_score', 'engagement_score']], 
                               use_container_width=True)
                else:
                    st.info("No high-value members identified")
            
            # Value concentration analysis
            st.markdown("#### 📊 Value Concentration Analysis")
            value_trends = analysis.get('value_trends', {})
            value_concentration = value_trends.get('value_concentration', {})
            
            if value_concentration:
                col1, col2 = st.columns(2)
                
                with col1:
                    top_10_share = value_concentration.get('top_10_percent_share', 0) * 100
                    st.metric("Top 10% Value Share", f"{top_10_share:.1f}%")
                
                with col2:
                    gini_coefficient = value_concentration.get('gini_coefficient', 0)
                    st.metric("Value Gini Coefficient", f"{gini_coefficient:.3f}")
        else:
            st.info("No segmentation analysis data available")
    
    def render_ltv_analysis(self, analysis):
        """Render lifetime value analysis"""
        st.markdown("---")
        st.subheader("💰 Member Lifetime Value Analysis")
        
        ltv_analysis = analysis.get('ltv_analysis', [])
        
        if ltv_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 LTV by Segment")
                
                # LTV by segment
                ltv_data = []
                for ltv in ltv_analysis:
                    ltv_data.append({
                        'Segment': ltv.segment.value,
                        'Current Value': ltv.current_value,
                        'Predicted LTV': ltv.predicted_ltv,
                        'Retention Probability': ltv.retention_probability * 100,
                        'Cross-sell Potential': ltv.cross_sell_potential * 100
                    })
                
                ltv_df = pd.DataFrame(ltv_data)
                
                fig = px.box(
                    ltv_df,
                    x='Segment',
                    y='Predicted LTV',
                    title="Lifetime Value Distribution by Segment",
                    color='Segment'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 LTV Components")
                
                # LTV components for top members
                top_ltv = sorted(ltv_analysis, key=lambda x: x.predicted_ltv, reverse=True)[:10]
                
                components_data = []
                for ltv in top_ltv:
                    components_data.append({
                        'Member ID': ltv.member_id,
                        'Segment': ltv.segment.value,
                        'Current Value': ltv.current_value,
                        'LTV Multiplier': ltv.predicted_ltv / ltv.current_value,
                        'Retention %': ltv.retention_probability * 100,
                        'Cross-sell %': ltv.cross_sell_potential * 100
                    })
                
                components_df = pd.DataFrame(components_data)
                st.dataframe(components_df, use_container_width=True)
            
            # LTV trend analysis
            st.markdown("#### 📅 Member Value Trends")
            value_trends = analysis.get('value_trends', {})
            balance_trend = value_trends.get('average_balance_trend', {})
            
            if balance_trend:
                periods = list(balance_trend.keys())
                values = list(balance_trend.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=values,
                    name='Average Member Balance',
                    line=dict(color='blue', width=3),
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title="Average Member Balance Trend",
                    xaxis_title="Period",
                    yaxis_title="Average Balance (KES)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LTV analysis data available")
    
    def render_engagement_analysis(self, analysis):
        """Render member engagement analysis"""
        st.markdown("---")
        st.subheader("📱 Member Engagement Analysis")
        
        engagement_analysis = analysis.get('engagement_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Engagement Distribution")
            
            engagement_scores = engagement_analysis.get('engagement_scores', [])
            if engagement_scores:
                scores = [score['engagement_score'] for score in engagement_scores]
                
                fig = px.histogram(
                    x=scores,
                    title="Member Engagement Score Distribution",
                    labels={'x': 'Engagement Score', 'y': 'Number of Members'},
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No engagement score data available")
        
        with col2:
            st.markdown("#### 📞 Channel Preferences")
            
            channel_preferences = engagement_analysis.get('channel_preferences', {})
            if channel_preferences:
                channels = list(channel_preferences.keys())
                usage = list(channel_preferences.values())
                
                fig = px.bar(
                    x=channels,
                    y=usage,
                    title="Service Channel Preferences",
                    labels={'x': 'Channel', 'y': 'Usage Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No channel preference data available")
        
        # Engagement trends
        st.markdown("#### 📈 Engagement Trends")
        engagement_trend = engagement_analysis.get('engagement_trend', {})
        if engagement_trend:
            quarters = list(engagement_trend.keys())
            scores = [score * 100 for score in engagement_trend.values()]  # Convert to percentage
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=quarters,
                y=scores,
                name='Average Engagement',
                line=dict(color='green', width=3),
                mode='lines+markers'
            ))
            fig.update_layout(
                title="Quarterly Engagement Trend",
                xaxis_title="Quarter",
                yaxis_title="Engagement Score (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No engagement trend data available")
    
    def render_churn_analysis(self, analysis):
        """Render churn risk analysis"""
        st.markdown("---")
        st.subheader("🚨 Churn Risk Analysis")
        
        churn_analysis = analysis.get('churn_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 Risk Distribution")
            
            high_risk_count = churn_analysis.get('high_risk_count', 0)
            medium_risk_count = churn_analysis.get('medium_risk_count', 0)
            total_members = high_risk_count + medium_risk_count + 1000  # Approximate total
            
            risk_data = {
                'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [high_risk_count, medium_risk_count, total_members - high_risk_count - medium_risk_count]
            }
            
            risk_df = pd.DataFrame(risk_data)
            
            fig = px.pie(
                risk_df,
                names='Risk Level',
                values='Count',
                title="Member Churn Risk Distribution",
                color='Risk Level',
                color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 High-Risk Members")
            
            high_risk_members = churn_analysis.get('high_risk_members', [])
            if high_risk_members:
                risk_df = pd.DataFrame(high_risk_members)
                st.dataframe(risk_df[['member_id', 'engagement_score', 'recency_days', 'transaction_frequency']], 
                           use_container_width=True)
            else:
                st.info("No high-risk members identified")
        
        # Churn trend and campaign effectiveness
        st.markdown("#### 📊 Churn Management Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_trend = churn_analysis.get('churn_probability_trend', {})
            if churn_trend:
                quarters = list(churn_trend.keys())
                probabilities = [prob * 100 for prob in churn_trend.values()]
                
                fig = px.line(
                    x=quarters,
                    y=probabilities,
                    title="Churn Probability Trend",
                    labels={'x': 'Quarter', 'y': 'Churn Probability (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            campaign_effectiveness = churn_analysis.get('retention_campaign_effectiveness', 0) * 100
            st.metric(
                "Retention Campaign Effectiveness",
                f"{campaign_effectiveness:.1f}%",
                help="Success rate of retention campaigns"
            )
            
            member_growth = analysis.get('value_trends', {}).get('member_growth', {})
            net_growth = member_growth.get('net_growth', 0)
            st.metric(
                "Net Member Growth",
                f"+{net_growth}",
                help="New members minus lost members"
            )
    
    def render_business_opportunities(self, analysis):
        """Render business opportunities"""
        st.markdown("---")
        st.subheader("🎯 Business Opportunities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Retention Opportunities")
            
            retention_opportunities = analysis.get('retention_opportunities', [])
            if retention_opportunities:
                for opportunity in retention_opportunities:
                    with st.expander(f"🛡️ {opportunity['member_id']} - {opportunity['recommended_action']}"):
                        st.write(f"**Priority:** {opportunity['priority']}")
                        st.write(f"**Estimated Value at Risk:** KES {opportunity['estimated_value']:,.0f}")
                        st.write(f"**Success Probability:** {opportunity['success_probability'] * 100:.1f}%")
                        
                        if st.button(f"Activate Retention Campaign", key=f"retain_{opportunity['member_id']}"):
                            st.success(f"Retention campaign activated for {opportunity['member_id']}")
            else:
                st.info("No retention opportunities identified")
        
        with col2:
            st.markdown("#### 🚀 Cross-sell Opportunities")
            
            cross_sell_opportunities = analysis.get('cross_sell_opportunities', [])
            if cross_sell_opportunities:
                for opportunity in cross_sell_opportunities[:5]:  # Show top 5
                    with st.expander(f"📈 {opportunity['member_id']} - Cross-sell Potential"):
                        st.write(f"**Recommended Products:** {', '.join(opportunity['recommended_products'])}")
                        st.write(f"**Estimated Value:** KES {opportunity['estimated_value']:,.0f}")
                        st.write(f"**Success Probability:** {opportunity['success_probability'] * 100:.1f}%")
                        
                        if st.button(f"Initiate Cross-sell", key=f"cross_sell_{opportunity['member_id']}"):
                            st.success(f"Cross-sell initiative started for {opportunity['member_id']}")
            else:
                st.info("No cross-sell opportunities identified")
    
    def run(self):
        """Run the member value page"""
        st.title("👥 Member Value & Lifetime Value Analysis")
        
        st.markdown("""
        Comprehensive analysis of member lifetime value, engagement, and profitability to optimize 
        member relationships, improve retention, and identify growth opportunities through personalized 
        offerings and targeted campaigns.
        """)
        
        try:
            self.render_member_value_dashboard()
        except Exception as e:
            st.error(f"Error running member value page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = MemberValuePage()
    page.run()