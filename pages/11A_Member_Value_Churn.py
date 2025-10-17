# pages/11A_Member_Value_Churn.py
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
from sacco_core.analytics.churn_analysis import ChurnAnalyzer, ChurnRiskLevel, InterventionType
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Churn Prediction & Retention",
    page_icon="🚨",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class ChurnPredictionPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.churn_analyzer = ChurnAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "11A_Member_Value_Churn.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "churn_prediction_page"
        )
        return True
    
    def render_churn_dashboard(self):
        """Render churn prediction dashboard"""
        st.subheader("🚨 Advanced Churn Prediction & Retention Analytics")
        
        try:
            # Get churn analysis
            analysis = self.churn_analyzer.analyze_churn_risk()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_distribution = analysis.get('risk_distribution', {})
                critical_risk = risk_distribution.get(ChurnRiskLevel.CRITICAL.value, 0)
                st.metric(
                    "Critical Risk Members",
                    f"{critical_risk}",
                    delta_color="inverse",
                    help="Members with critical churn risk requiring immediate action"
                )
            
            with col2:
                high_risk = risk_distribution.get(ChurnRiskLevel.HIGH.value, 0)
                st.metric(
                    "High Risk Members",
                    f"{high_risk}",
                    delta_color="inverse",
                    help="Members with high churn risk"
                )
            
            with col3:
                retention_roi = analysis.get('retention_roi', {})
                overall_roi = retention_roi.get('overall_roi', 0) * 100
                st.metric(
                    "Estimated Retention ROI",
                    f"{overall_roi:.1f}%",
                    help="Return on investment for retention interventions"
                )
            
            with col4:
                model_performance = analysis.get('model_performance', {})
                accuracy = model_performance.get('accuracy', 0) * 100
                st.metric(
                    "Model Accuracy",
                    f"{accuracy:.1f}%",
                    help="Churn prediction model accuracy"
                )
            
            # Churn overview
            self.render_churn_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering churn dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_churn_overview(self, analysis):
        """Render churn prediction overview"""
        st.markdown("#### 📊 Churn Risk Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            risk_distribution = analysis.get('risk_distribution', {})
            if risk_distribution:
                risks = list(risk_distribution.keys())
                counts = list(risk_distribution.values())
                
                # Color mapping for risk levels
                colors = {
                    ChurnRiskLevel.CRITICAL.value: 'red',
                    ChurnRiskLevel.HIGH.value: 'orange',
                    ChurnRiskLevel.MEDIUM.value: 'yellow',
                    ChurnRiskLevel.LOW.value: 'lightgreen',
                    ChurnRiskLevel.MINIMAL.value: 'green'
                }
                
                fig = px.bar(
                    x=risks,
                    y=counts,
                    title="Churn Risk Distribution",
                    labels={'x': 'Risk Level', 'y': 'Number of Members'},
                    color=risks,
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk distribution data available")
        
        with col2:
            # Churn drivers
            churn_drivers = analysis.get('churn_drivers', {})
            if churn_drivers:
                drivers = list(churn_drivers.keys())
                percentages = [p * 100 for p in churn_drivers.values()]
                
                fig = px.pie(
                    names=drivers,
                    values=percentages,
                    title="Primary Churn Drivers",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No churn driver data available")
        
        # Detailed analysis sections
        self.render_risk_analysis(analysis)
        self.render_intervention_strategies(analysis)
        self.render_campaign_performance(analysis)
        self.render_early_warnings(analysis)
        self.render_model_insights(analysis)
    
    def render_risk_analysis(self, analysis):
        """Render detailed risk analysis"""
        st.markdown("---")
        st.subheader("🎯 High-Risk Member Analysis")
        
        churn_predictions = analysis.get('churn_predictions', [])
        
        if churn_predictions:
            # Filter high and critical risk members
            high_risk_predictions = [
                p for p in churn_predictions 
                if p.risk_level in [ChurnRiskLevel.CRITICAL, ChurnRiskLevel.HIGH]
            ]
            
            if high_risk_predictions:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Risk Probability Distribution")
                    
                    probabilities = [p.churn_probability * 100 for p in high_risk_predictions]
                    
                    fig = px.histogram(
                        x=probabilities,
                        title="Churn Probability Distribution - High Risk Members",
                        labels={'x': 'Churn Probability (%)', 'y': 'Number of Members'},
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### ⏰ Predicted Churn Timeline")
                    
                    # Group by predicted churn timeframe
                    now = datetime.now()
                    timeframes = {
                        'Within 30 days': 0,
                        '30-60 days': 0,
                        '60-90 days': 0,
                        'Beyond 90 days': 0
                    }
                    
                    for prediction in high_risk_predictions:
                        days_to_churn = (prediction.predicted_churn_date - now).days
                        if days_to_churn <= 30:
                            timeframes['Within 30 days'] += 1
                        elif days_to_churn <= 60:
                            timeframes['30-60 days'] += 1
                        elif days_to_churn <= 90:
                            timeframes['60-90 days'] += 1
                        else:
                            timeframes['Beyond 90 days'] += 1
                    
                    fig = px.bar(
                        x=list(timeframes.keys()),
                        y=list(timeframes.values()),
                        title="Predicted Churn Timeline",
                        labels={'x': 'Timeframe', 'y': 'Number of Members'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # High-risk members table
                st.markdown("#### 📋 High-Risk Member Details")
                
                risk_data = []
                for prediction in high_risk_predictions[:20]:  # Show top 20
                    risk_data.append({
                        'Member ID': prediction.member_id,
                        'Churn Probability': f"{prediction.churn_probability * 100:.1f}%",
                        'Risk Level': prediction.risk_level.value,
                        'Predicted Churn': prediction.predicted_churn_date.strftime('%Y-%m-%d'),
                        'Key Drivers': ', '.join(prediction.key_drivers),
                        'Confidence': f"{prediction.confidence_score * 100:.1f}%"
                    })
                
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.success("✅ No high-risk members identified")
        else:
            st.info("No churn prediction data available")
    
    def render_intervention_strategies(self, analysis):
        """Render intervention strategies"""
        st.markdown("---")
        st.subheader("🛡️ Retention Intervention Strategies")
        
        interventions = analysis.get('intervention_strategies', [])
        
        if interventions:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💡 Intervention Types")
                
                # Intervention type distribution
                intervention_types = {}
                for intervention in interventions:
                    type_name = intervention.intervention_type.value
                    intervention_types[type_name] = intervention_types.get(type_name, 0) + 1
                
                fig = px.pie(
                    names=list(intervention_types.keys()),
                    values=list(intervention_types.values()),
                    title="Recommended Intervention Types",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Expected ROI by Intervention")
                
                # ROI analysis
                roi_data = []
                for intervention in interventions:
                    roi_data.append({
                        'Intervention Type': intervention.intervention_type.value,
                        'Expected ROI': intervention.estimated_roi * 100,
                        'Success Rate': intervention.expected_success_rate * 100
                    })
                
                roi_df = pd.DataFrame(roi_data)
                
                if not roi_df.empty:
                    fig = px.scatter(
                        roi_df,
                        x='Success Rate',
                        y='Expected ROI',
                        color='Intervention Type',
                        size='Success Rate',
                        title="Intervention Effectiveness: Success Rate vs ROI",
                        labels={'Success Rate': 'Expected Success Rate (%)', 'Expected ROI': 'Expected ROI (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Intervention details with action buttons
            st.markdown("#### 🎯 Recommended Interventions")
            
            for intervention in interventions[:10]:  # Show top 10
                with st.expander(f"🎯 {intervention.member_id} - {intervention.intervention_type.value} (Priority: {intervention.priority})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Success", f"{intervention.expected_success_rate * 100:.1f}%")
                    
                    with col2:
                        st.metric("Estimated ROI", f"{intervention.estimated_roi * 100:.1f}%")
                    
                    with col3:
                        st.metric("Optimal Timing", intervention.optimal_timing.strftime('%Y-%m-%d'))
                    
                    st.write(f"**Recommended Action:** {intervention.recommended_message}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"✅ Approve Intervention", key=f"approve_{intervention.member_id}"):
                            st.success(f"Intervention approved for {intervention.member_id}")
                    
                    with col2:
                        if st.button(f"📞 Assign to Agent", key=f"assign_{intervention.member_id}"):
                            st.info(f"Intervention assigned to retention agent for {intervention.member_id}")
                    
                    with col3:
                        if st.button(f"⏰ Schedule Follow-up", key=f"schedule_{intervention.member_id}"):
                            st.info(f"Follow-up scheduled for {intervention.member_id}")
        else:
            st.info("No intervention strategies recommended")
    
    def render_campaign_performance(self, analysis):
        """Render campaign performance analysis"""
        st.markdown("---")
        st.subheader("📈 Retention Campaign Performance")
        
        campaign_analysis = analysis.get('campaign_analysis', {})
        active_campaigns = campaign_analysis.get('active_campaigns', [])
        historical_performance = campaign_analysis.get('historical_performance', {})
        
        if active_campaigns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Active Campaigns")
                
                for campaign in active_campaigns:
                    with st.expander(f"📊 {campaign['name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Success Rate", f"{campaign['success_rate'] * 100:.1f}%")
                        
                        with col2:
                            st.metric("ROI", f"{campaign['roi']:.1f}x")
                        
                        with col3:
                            st.metric("Value Saved", f"KES {campaign['estimated_value_saved']:,.0f}")
                        
                        st.progress(campaign['success_rate'])
            
            with col2:
                st.markdown("#### 📅 Historical Performance")
                
                if historical_performance:
                    quarters = list(historical_performance.keys())
                    success_rates = [data['success_rate'] * 100 for data in historical_performance.values()]
                    roi_values = [data['roi'] for data in historical_performance.values()]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=quarters,
                        y=success_rates,
                        name='Success Rate',
                        line=dict(color='blue', width=3),
                        yaxis='y1'
                    ))
                    fig.add_trace(go.Scatter(
                        x=quarters,
                        y=roi_values,
                        name='ROI',
                        line=dict(color='green', width=3),
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        title="Historical Campaign Performance",
                        xaxis_title="Quarter",
                        yaxis=dict(title='Success Rate (%)', side='left'),
                        yaxis2=dict(title='ROI', side='right', overlaying='y')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Best performing interventions
        best_interventions = campaign_analysis.get('best_performing_interventions', [])
        if best_interventions:
            st.markdown("#### 🏆 Best Performing Interventions")
            
            for intervention in best_interventions:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Type", intervention['type'])
                with col2:
                    st.metric("Success Rate", f"{intervention['success_rate'] * 100:.1f}%")
                with col3:
                    st.metric("Cost per Member", f"KES {intervention['cost_per_member']:,.0f}")
    
    def render_early_warnings(self, analysis):
        """Render early warning indicators"""
        st.markdown("---")
        st.subheader("⚠️ Early Warning Indicators")
        
        early_warnings = analysis.get('early_warning_indicators', [])
        
        if early_warnings:
            for warning in early_warnings:
                severity_color = {
                    'High': 'red',
                    'Medium': 'orange', 
                    'Low': 'yellow'
                }.get(warning['severity'], 'gray')
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{warning['indicator']}**")
                        st.write(warning['description'])
                    
                    with col2:
                        st.metric("Affected Members", warning['affected_members'])
                    
                    with col3:
                        st.metric("Severity", warning['severity'])
                    
                    st.markdown("---")
        else:
            st.success("✅ No early warnings detected")
    
    def render_model_insights(self, analysis):
        """Render model insights and performance"""
        st.markdown("---")
        st.subheader("🤖 Model Insights & Performance")
        
        model_performance = analysis.get('model_performance', {})
        
        if model_performance:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Model Metrics")
                
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                    'Value': [
                        f"{model_performance.get('accuracy', 0) * 100:.1f}%",
                        f"{model_performance.get('precision', 0) * 100:.1f}%", 
                        f"{model_performance.get('recall', 0) * 100:.1f}%",
                        f"{model_performance.get('f1_score', 0) * 100:.1f}%",
                        f"{model_performance.get('auc_roc', 0) * 100:.1f}%"
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 Feature Importance")
                
                feature_importance = model_performance.get('feature_importance', {})
                if feature_importance:
                    features = list(feature_importance.keys())
                    importance = [imp * 100 for imp in feature_importance.values()]
                    
                    fig = px.bar(
                        x=importance,
                        y=features,
                        orientation='h',
                        title="Feature Importance in Churn Prediction",
                        labels={'x': 'Importance (%)', 'y': 'Features'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model information
            st.markdown("#### ℹ️ Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_performance.get('model_type', 'Unknown'))
            
            with col2:
                st.metric("Training Date", model_performance.get('training_date', 'Unknown'))
            
            with col3:
                st.metric("Data Freshness", model_performance.get('data_freshness', 'Unknown'))
        else:
            st.info("No model performance data available")
    
    def run(self):
        """Run the churn prediction page"""
        st.title("🚨 Advanced Churn Prediction & Retention Analytics")
        
        st.markdown("""
        Machine learning-powered churn prediction, early warning detection, and personalized retention 
        strategies to proactively protect member relationships and maximize lifetime value through 
        data-driven intervention optimization.
        """)
        
        try:
            self.render_churn_dashboard()
        except Exception as e:
            st.error(f"Error running churn prediction page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = ChurnPredictionPage()
    page.run()