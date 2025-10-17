# pages/10_Cybersecurity_BCP.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sacco_core.sidebar import render_sidebar

# Import with fallbacks for core modules
try:
    from sacco_core.audit import audit_log
except ImportError:
    # Fallback audit log function
    def audit_log(action: str, description: str, payload: Optional[Dict] = None):
        # Silent fail for audit logging
        print(f"AUDIT: {action} - {description}")

try:
    from sacco_core.rbac import check_permission, RBAC_ROLES
except ImportError:
    # Fallback RBAC functions
    def check_permission(page_module: str) -> bool:
        # Allow all access if RBAC fails
        return True
    
    RBAC_ROLES = {
        "admin": "Administrator",
        "risk_manager": "Risk Manager", 
        "credit_officer": "Credit Officer",
        "operations": "Operations Staff",
        "finance": "Finance Team",
        "compliance": "Compliance Officer",
        "member_services": "Member Services",
        "auditor": "Auditor"
    }

try:
    from sacco_core.analytics.cybersecurity import (
        CybersecurityAnalyzer, RiskLevel, ThreatCategory, 
        BCPStatus, SecurityIncident, BusinessImpact, SecurityControl
    )
except ImportError:
    # Create fallback classes if import fails
    class RiskLevel:
        CRITICAL = "Critical"
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"
        MINIMAL = "Minimal"
    
    class ThreatCategory:
        MALWARE = "Malware"
        PHISHING = "Phishing"
        DDoS = "DDoS"
        DATA_BREACH = "Data Breach"
        INSIDER_THREAT = "Insider Threat"
        SYSTEM_FAILURE = "System Failure"
        COMPLIANCE_VIOLATION = "Compliance Violation"
    
    class BCPStatus:
        FULLY_OPERATIONAL = "Fully Operational"
        MINIMAL_IMPACT = "Minimal Impact"
        MODERATE_IMPACT = "Moderate Impact"
        SEVERE_IMPACT = "Severe Impact"
        CRITICAL_FAILURE = "Critical Failure"
    
    class CybersecurityAnalyzer:
        def analyze_cybersecurity_risk(self):
            return self._get_fallback_analysis()
        
        def _get_fallback_analysis(self):
            return {
                'risk_assessment': {
                    'overall_risk_score': 0,
                    'risk_level': RiskLevel.MINIMAL,
                    'vulnerability_risk_score': 0,
                    'incident_risk_score': 0,
                    'critical_vulnerabilities': 0,
                    'unpatched_vulnerabilities': 0,
                    'trend_comparison': {}
                },
                'bcp_analysis': {
                    'bcp_status': BCPStatus.CRITICAL_FAILURE,
                    'readiness_score': 0,
                    'business_impact_analysis': [],
                    'recovery_capabilities': {},
                    'last_recovery_test': 'Unknown',
                    'recovery_test_success': False
                },
                'controls_assessment': {
                    'security_controls': [],
                    'overall_effectiveness': 0,
                    'fully_implemented': 0,
                    'partially_implemented': 0,
                    'not_implemented': 0
                },
                'compliance_analysis': {
                    'compliance_score': 0,
                    'failed_login_rate': 0,
                    'compliance_violations': ['Data unavailable'],
                    'gdpr_compliance': False,
                    'data_protection_compliance': False,
                    'access_control_compliance': False,
                    'audit_trail_compliance': False
                },
                'threat_analysis': {},
                'security_recommendations': [],
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# Page configuration
st.set_page_config(
    page_title="Cybersecurity & Business Continuity",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

def main():
    """Cybersecurity and Business Continuity Dashboard"""
    
    # RBAC check - fix the page module name
    if not check_permission("10_Cybersecurity_BCP.py"):
        st.error("You do not have permission to access this page.")
        return
    
    # Page header
    st.title("🔒 Cybersecurity & Business Continuity")
    st.markdown("---")
    
    # Initialize analyzer
    try:
        analyzer = CybersecurityAnalyzer()
        
        # Perform analysis
        with st.spinner("Analyzing cybersecurity risks and business continuity..."):
            analysis = analyzer.analyze_cybersecurity_risk()
    except Exception as e:
        st.error(f"Error initializing cybersecurity analyzer: {e}")
        # Use fallback analysis
        analyzer = CybersecurityAnalyzer()
        analysis = analyzer._get_fallback_analysis()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_score = analysis['risk_assessment']['overall_risk_score']
        risk_level = analysis['risk_assessment']['risk_level']
        
        # Color coding for risk level
        risk_color = {
            "Critical": "#FF0000",
            "High": "#FF6B00", 
            "Medium": "#FFA500",
            "Low": "#00FF00",
            "Minimal": "#008000"
        }.get(str(risk_level), "#666666")
        
        st.metric(
            label="Overall Cyber Risk Score",
            value=f"{risk_score:.1f}",
            delta=f"{risk_level} Risk",
            delta_color="inverse"
        )
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "green"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        bcp_score = analysis['bcp_analysis']['readiness_score']
        bcp_status = analysis['bcp_analysis']['bcp_status']
        
        st.metric(
            label="BCP Readiness Score", 
            value=f"{bcp_score:.1f}%",
            delta=f"{bcp_status}",
            delta_color="normal" if bcp_score >= 75 else "off"
        )
        
        # BCP readiness gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bcp_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BCP Readiness"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 75], 'color': "orange"},
                    {'range': [75, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        controls_score = analysis['controls_assessment']['overall_effectiveness']
        st.metric(
            label="Security Controls Effectiveness",
            value=f"{controls_score:.1f}%",
            delta="Optimal" if controls_score >= 85 else "Needs Review"
        )
        
        # Controls effectiveness pie chart
        controls_data = analysis['controls_assessment']
        labels = ['Fully Implemented', 'Partially Implemented', 'Not Implemented']
        values = [
            controls_data['fully_implemented'],
            controls_data['partially_implemented'], 
            controls_data['not_implemented']
        ]
        
        fig = px.pie(
            values=values, 
            names=labels,
            title="Security Controls Implementation",
            color_discrete_sequence=['green', 'orange', 'red']
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        compliance_score = analysis['compliance_analysis']['compliance_score']
        violations = len(analysis['compliance_analysis']['compliance_violations'])
        
        st.metric(
            label="Compliance Score",
            value=f"{compliance_score:.1f}",
            delta=f"{violations} Violations" if violations > 0 else "Compliant"
        )
        
        # Compliance status
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = compliance_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': 80},
            title = {'text': "Compliance Status"}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Threat Analysis", 
        "📊 Risk Assessment", 
        "🛡️ Business Continuity",
        "⚙️ Security Controls",
        "📋 Recommendations"
    ])
    
    with tab1:
        display_threat_analysis(analysis['threat_analysis'])
    
    with tab2:
        display_risk_assessment(analysis['risk_assessment'])
    
    with tab3:
        display_business_continuity(analysis['bcp_analysis'])
    
    with tab4:
        display_security_controls(analysis['controls_assessment'])
    
    with tab5:
        display_recommendations(analysis['security_recommendations'])
    
    # Audit logging
    try:
        audit_log(
            "cybersecurity_bcp_view",
            f"Viewed cybersecurity and BCP dashboard - Risk: {risk_level}, BCP: {bcp_status}",
            {"risk_score": risk_score, "bcp_score": bcp_score}
        )
    except Exception:
        pass  # Silent fail for audit logging

def display_threat_analysis(threat_analysis: Dict[str, Any]):
    """Display threat analysis section"""
    
    st.header("🔍 Threat Landscape Analysis")
    
    if not threat_analysis:
        st.warning("Threat analysis data unavailable")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Threat distribution chart
        threat_dist = threat_analysis.get('threat_distribution', {})
        if threat_dist:
            fig = px.bar(
                x=list(threat_dist.keys()),
                y=list(threat_dist.values()),
                title="Threat Category Distribution",
                labels={'x': 'Threat Category', 'y': 'Number of Incidents'},
                color=list(threat_dist.values()),
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_dist = threat_analysis.get('severity_distribution', {})
        if severity_dist:
            fig = px.pie(
                values=list(severity_dist.values()),
                names=list(severity_dist.keys()),
                title="Incident Severity Distribution",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Response time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Incidents",
            threat_analysis.get('total_incidents', 0)
        )
    
    with col2:
        st.metric(
            "Resolved Incidents", 
            threat_analysis.get('resolved_incidents', 0)
        )
    
    with col3:
        avg_response = threat_analysis.get('average_response_time_hours', 0)
        st.metric(
            "Avg Response Time",
            f"{avg_response:.1f} hours"
        )
    
    # Emerging threats
    st.subheader("🌍 Emerging Threats & Threat Actors")
    
    emerging_threats = threat_analysis.get('emerging_threats', [])
    threat_actors = threat_analysis.get('threat_actors', [])
    
    if emerging_threats:
        threats_df = pd.DataFrame(emerging_threats)
        st.dataframe(
            threats_df,
            use_container_width=True,
            column_config={
                "threat": "Threat",
                "severity": st.column_config.TextColumn("Severity"),
                "sector": "Target Sector"
            }
        )
    
    if threat_actors:
        actors_df = pd.DataFrame(threat_actors)
        st.dataframe(
            actors_df,
            use_container_width=True,
            column_config={
                "group": "Threat Group",
                "targets": "Primary Targets", 
                "tactics": "Common Tactics"
            }
        )

def display_risk_assessment(risk_assessment: Dict[str, Any]):
    """Display cybersecurity risk assessment"""
    
    st.header("📊 Cybersecurity Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk breakdown
        risk_components = {
            'Vulnerability Risk': risk_assessment.get('vulnerability_risk_score', 0),
            'Incident Risk': risk_assessment.get('incident_risk_score', 0)
        }
        
        fig = px.bar(
            x=list(risk_components.keys()),
            y=list(risk_components.values()),
            title="Risk Component Breakdown",
            labels={'x': 'Risk Component', 'y': 'Risk Score'},
            color=list(risk_components.values()),
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk trend
        trend_data = risk_assessment.get('trend_comparison', {})
        if trend_data:
            fig = px.line(
                x=list(trend_data.keys()),
                y=list(trend_data.values()),
                title="Risk Trend Over Time",
                labels={'x': 'Time Period', 'y': 'Risk Score'},
                markers=True
            )
            fig.update_traces(line=dict(color='red', width=3))
            st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerability metrics
    st.subheader("🛡️ Vulnerability Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Critical Vulnerabilities",
            risk_assessment.get('critical_vulnerabilities', 0)
        )
    
    with col2:
        st.metric(
            "Unpatched Vulnerabilities",
            risk_assessment.get('unpatched_vulnerabilities', 0)
        )
    
    with col3:
        vuln_score = risk_assessment.get('vulnerability_risk_score', 0)
        st.metric(
            "Vulnerability Risk Score",
            f"{vuln_score:.1f}"
        )
    
    # Risk mitigation actions
    st.subheader("🚨 Immediate Risk Mitigation Actions")
    
    risk_level = risk_assessment.get('risk_level', 'Minimal')
    
    if risk_level == "Critical":
        st.error("""
        **CRITICAL RISK LEVEL - IMMEDIATE ACTION REQUIRED:**
        - Isolate affected systems
        - Activate incident response team
        - Notify executive leadership
        - Implement emergency patches
        """)
    elif risk_level == "High":
        st.warning("""
        **HIGH RISK LEVEL - PRIORITY ACTIONS:**
        - Schedule immediate patching
        - Enhance monitoring
        - Review access controls
        - Update incident response plan
        """)
    elif risk_level == "Medium":
        st.info("""
        **MEDIUM RISK LEVEL - PLANNED ACTIONS:**
        - Schedule patching within 30 days
        - Conduct security review
        - Update security controls
        - Staff training
        """)
    else:
        st.success("""
        **LOW/MINIMAL RISK - MAINTENANCE ACTIONS:**
        - Continue regular monitoring
        - Maintain security controls
        - Regular staff training
        - Periodic security assessments
        """)

def display_business_continuity(bcp_analysis: Dict[str, Any]):
    """Display business continuity analysis"""
    
    st.header("🛡️ Business Continuity Planning")
    
    # BCP status overview
    bcp_status = bcp_analysis.get('bcp_status', 'Unknown')
    readiness_score = bcp_analysis.get('readiness_score', 0)
    
    status_colors = {
        "Fully Operational": "green",
        "Minimal Impact": "lightgreen", 
        "Moderate Impact": "orange",
        "Severe Impact": "red",
        "Critical Failure": "darkred"
    }
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {status_colors.get(str(bcp_status), 'gray')}; color: white;">
        <h3 style="margin: 0;">BCP Status: {bcp_status}</h3>
        <p style="margin: 0;">Readiness Score: {readiness_score:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Business Impact Analysis
    st.subheader("📈 Business Impact Analysis (BIA)")
    
    business_impacts = bcp_analysis.get('business_impact_analysis', [])
    
    if business_impacts:
        bia_data = []
        for impact in business_impacts:
            bia_data.append({
                'Business Process': getattr(impact, 'business_process', 'Unknown'),
                'RTO (hours)': getattr(impact, 'recovery_time_objective', 0),
                'RPO (hours)': getattr(impact, 'recovery_point_objective', 0),
                'Max Downtime (hours)': getattr(impact, 'maximum_tolerable_downtime', 0),
                'Financial Impact/hr': f"${getattr(impact, 'financial_impact_per_hour', 0):,.0f}",
                'Criticality': getattr(impact, 'criticality', 'Unknown')
            })
        
        bia_df = pd.DataFrame(bia_data)
        st.dataframe(bia_df, use_container_width=True)
    else:
        st.info("No business impact analysis data available")
    
    # Recovery capabilities
    st.subheader("🔄 Recovery Capabilities")
    
    recovery_caps = bcp_analysis.get('recovery_capabilities', {})
    
    if recovery_caps:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "RTO (Recovery Time Objective)",
                f"{recovery_caps.get('recovery_time_objective', 'N/A')} hours"
            )
        
        with col2:
            st.metric(
                "RPO (Recovery Point Objective)", 
                f"{recovery_caps.get('recovery_point_objective', 'N/A')} hours"
            )
        
        with col3:
            backup_success = recovery_caps.get('backup_success_rate', 0) * 100
            st.metric(
                "Backup Success Rate",
                f"{backup_success:.1f}%"
            )
        
        with col4:
            last_test = recovery_caps.get('last_recovery_test', 'Unknown')
            if isinstance(last_test, datetime):
                last_test = last_test.strftime('%Y-%m-%d')
            st.metric("Last Recovery Test", str(last_test))
    
    # Recovery testing status
    st.subheader("🧪 Recovery Testing")
    
    test_success = bcp_analysis.get('recovery_test_success', False)
    last_test = bcp_analysis.get('last_recovery_test', 'Unknown')
    
    if test_success:
        st.success("✅ Last recovery test was successful")
    else:
        st.error("❌ Last recovery test failed or not conducted")
    
    # Recovery capability visualization
    recovery_data = bcp_analysis.get('recovery_capabilities', {})
    if recovery_data:
        capabilities = {
            'Backup Frequency': recovery_data.get('backup_frequency_hours', 0),
            'Backup Success Rate': recovery_data.get('backup_success_rate', 0) * 100,
            'Encryption': 100 if recovery_data.get('backup_encryption', False) else 0,
            'Offsite Backup': 100 if recovery_data.get('offsite_backup', False) else 0
        }
        
        fig = px.bar(
            x=list(capabilities.keys()),
            y=list(capabilities.values()),
            title="Recovery Capability Scores",
            labels={'x': 'Capability', 'y': 'Score'},
            color=list(capabilities.values()),
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_security_controls(controls_assessment: Dict[str, Any]):
    """Display security controls assessment"""
    
    st.header("⚙️ Security Controls Assessment")
    
    # Controls effectiveness overview
    effectiveness = controls_assessment.get('overall_effectiveness', 0)
    
    st.metric(
        "Overall Controls Effectiveness",
        f"{effectiveness:.1f}%",
        delta="Optimal" if effectiveness >= 85 else "Needs Improvement",
        delta_color="normal" if effectiveness >= 85 else "off"
    )
    
    # Controls implementation status
    controls_data = controls_assessment.get('security_controls', [])
    
    if controls_data:
        # Create controls dataframe
        controls_list = []
        for control in controls_data:
            controls_list.append({
                'Control ID': getattr(control, 'control_id', 'Unknown'),
                'Control Name': getattr(control, 'control_name', 'Unknown'),
                'Category': getattr(control, 'category', 'Unknown'),
                'Status': getattr(control, 'implementation_status', 'Unknown'),
                'Effectiveness': f"{getattr(control, 'effectiveness_score', 0) * 100:.1f}%",
                'Last Test': getattr(control, 'last_test_date', datetime.now()).strftime('%Y-%m-%d'),
                'Next Test': getattr(control, 'next_test_date', datetime.now()).strftime('%Y-%m-%d')
            })
        
        controls_df = pd.DataFrame(controls_list)
        
        # Display with conditional formatting
        st.dataframe(
            controls_df,
            use_container_width=True,
            column_config={
                "Effectiveness": st.column_config.ProgressColumn(
                    "Effectiveness",
                    help="Control effectiveness score",
                    format="%f",
                    min_value=0,
                    max_value=100,
                )
            }
        )
    else:
        st.info("No security controls data available")
    
    # Controls by category
    if controls_data:
        categories = {}
        for control in controls_data:
            category = getattr(control, 'category', 'Unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(getattr(control, 'effectiveness_score', 0))
        
        # Calculate average effectiveness by category
        category_effectiveness = {
            cat: np.mean(scores) * 100 for cat, scores in categories.items()
        }
        
        fig = px.bar(
            x=list(category_effectiveness.keys()),
            y=list(category_effectiveness.values()),
            title="Controls Effectiveness by Category",
            labels={'x': 'Control Category', 'y': 'Average Effectiveness (%)'},
            color=list(category_effectiveness.values()),
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Implementation status summary
    st.subheader("📋 Implementation Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Fully Implemented",
            controls_assessment.get('fully_implemented', 0)
        )
    
    with col2:
        st.metric(
            "Partially Implemented",
            controls_assessment.get('partially_implemented', 0)
        )
    
    with col3:
        st.metric(
            "Not Implemented", 
            controls_assessment.get('not_implemented', 0)
        )

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display security recommendations"""
    
    st.header("📋 Security Improvement Recommendations")
    
    if not recommendations:
        st.info("No specific recommendations at this time. Current security posture appears adequate.")
        return
    
    # Priority-based recommendations
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    sorted_recommendations = sorted(
        recommendations, 
        key=lambda x: priority_order.get(x.get('priority', 'Low'), 3)
    )
    
    for i, rec in enumerate(sorted_recommendations, 1):
        priority = rec.get('priority', 'Medium')
        category = rec.get('category', 'General')
        recommendation = rec.get('recommendation', '')
        effort = rec.get('estimated_effort', 'Unknown')
        impact = rec.get('impact', 'Medium')
        
        # Color coding based on priority
        priority_colors = {
            'Critical': 'red',
            'High': 'orange', 
            'Medium': 'yellow',
            'Low': 'green'
        }
        
        with st.container():
            st.markdown(f"""
            <div style="padding: 15px; border-left: 5px solid {priority_colors.get(priority, 'gray')}; 
                        background-color: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                <h4 style="margin: 0;">{i}. {recommendation}</h4>
                <p style="margin: 5px 0;">
                    <strong>Priority:</strong> <span style="color: {priority_colors.get(priority, 'black')};">{priority}</span> | 
                    <strong>Category:</strong> {category} | 
                    <strong>Effort:</strong> {effort} | 
                    <strong>Impact:</strong> {impact}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Action plan summary
    st.subheader("🎯 Recommended Action Plan")
    
    critical_actions = [r for r in sorted_recommendations if r.get('priority') in ['Critical', 'High']]
    medium_actions = [r for r in sorted_recommendations if r.get('priority') == 'Medium']
    low_actions = [r for r in sorted_recommendations if r.get('priority') == 'Low']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Critical/High Priority Actions", len(critical_actions))
        if critical_actions:
            for action in critical_actions:
                st.write(f"• {action.get('recommendation')}")
    
    with col2:
        st.metric("Medium Priority Actions", len(medium_actions))
        if medium_actions:
            for action in medium_actions:
                st.write(f"• {action.get('recommendation')}")
    
    with col3:
        st.metric("Low Priority Actions", len(low_actions))
        if low_actions:
            for action in low_actions:
                st.write(f"• {action.get('recommendation')}")
    
    # Export recommendations
    st.subheader("📤 Export Action Plan")
    
    if st.button("Generate Security Improvement Report"):
        report_data = {
            "generated_date": datetime.now().isoformat(),
            "recommendations": sorted_recommendations,
            "summary": {
                "total_recommendations": len(recommendations),
                "critical_high_priority": len(critical_actions),
                "medium_priority": len(medium_actions),
                "low_priority": len(low_actions)
            }
        }
        
        # Create downloadable JSON
        st.download_button(
            label="Download Recommendations Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"security_recommendations_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()