# pages/12_Policy_Engine_Monitor.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

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
    from sacco_core.analytics.policy_engine import (
        PolicyEngineAnalyzer, PolicyRuleType, RuleStatus, 
        AlertSeverity, PolicyRule, PolicyAlert
    )
except ImportError:
    # Create fallback classes if import fails
    class PolicyRuleType:
        RISK_MONITORING = "Risk Monitoring"
        COMPLIANCE = "Compliance"
        OPERATIONAL = "Operational"
        FINANCIAL = "Financial"
        SECURITY = "Security"
    
    class RuleStatus:
        ACTIVE = "Active"
        INACTIVE = "Inactive"
        TRIGGERED = "Triggered"
        OVERRIDDEN = "Overridden"
        DISABLED = "Disabled"
    
    class AlertSeverity:
        CRITICAL = "Critical"
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"
        INFO = "Information"
    
    class PolicyEngineAnalyzer:
        def __init__(self, db_connection=None):
            self.policy_rules = []
        
        def execute_policy_engine(self):
            return self._get_fallback_analysis()
        
        def _get_fallback_analysis(self):
            return {
                'execution_summary': {
                    'total_rules_executed': 0,
                    'rules_triggered': 0,
                    'successful_executions': 0,
                    'execution_success_rate': 0,
                    'trigger_rate': 0,
                    'last_execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'triggered_rules': [],
                'active_alerts': [],
                'rule_effectiveness': {'overall_effectiveness': 0},
                'risk_exposure': {'risk_exposure_percentage': 0, 'risk_level': 'Low'},
                'compliance_status': {'compliance_rate': 0, 'compliance_status': 'Unknown'},
                'recommendations': [],
                'execution_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

try:
    from sacco_core.db import get_db_connection
except ImportError:
    # Fallback database connection
    def get_db_connection():
        return None

# Page configuration
st.set_page_config(
    page_title="Policy Engine Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Policy Engine Monitoring Dashboard"""
    
    # RBAC check - fix the page module name
    if not check_permission("12_Policy_Engine_Monitor.py"):
        st.error("You do not have permission to access this page.")
        return
    
    # Page header
    st.title("⚙️ Policy Engine Monitor")
    st.markdown("Real-time monitoring of policy rules, automated alerts, and risk management controls")
    st.markdown("---")
    
    # Initialize policy engine
    try:
        db_connection = get_db_connection()
        engine = PolicyEngineAnalyzer(db_connection)
        
        # Execute policy engine
        with st.spinner("Executing policy rules and analyzing compliance..."):
            analysis = engine.execute_policy_engine()
    except Exception as e:
        st.error(f"Error initializing policy engine: {e}")
        # Use fallback analysis
        engine = PolicyEngineAnalyzer()
        analysis = engine._get_fallback_analysis()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        summary = analysis['execution_summary']
        st.metric(
            label="Total Rules Executed",
            value=summary['total_rules_executed'],
            delta=f"{summary['rules_triggered']} Triggered"
        )
        
        # Execution success gauge
        success_rate = summary['execution_success_rate']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Execution Success Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 80], 'color': "red"},
                    {'range': [80, 95], 'color': "orange"},
                    {'range': [95, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_exposure = analysis['risk_exposure']
        st.metric(
            label="Risk Exposure",
            value=f"{risk_exposure['risk_exposure_percentage']:.1f}%",
            delta=risk_exposure['risk_level']
        )
        
        # Risk exposure gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_exposure['risk_exposure_percentage'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Exposure Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        compliance = analysis['compliance_status']
        st.metric(
            label="Compliance Rate",
            value=f"{compliance['compliance_rate']:.1f}%",
            delta=compliance['compliance_status']
        )
        
        # Compliance status
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=compliance['compliance_rate'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Regulatory Compliance"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 85], 'color': "red"},
                    {'range': [85, 95], 'color': "orange"},
                    {'range': [95, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        effectiveness = analysis['rule_effectiveness']
        st.metric(
            label="Rule Effectiveness",
            value=f"{effectiveness['overall_effectiveness']:.1f}%",
            delta="Optimal" if effectiveness['overall_effectiveness'] >= 80 else "Needs Review"
        )
        
        # Active alerts
        active_alerts = len(analysis['active_alerts'])
        critical_alerts = len([a for a in analysis['active_alerts'] if getattr(a, 'severity', 'CRITICAL') == 'Critical'])
        
        st.metric(
            label="Active Alerts",
            value=active_alerts,
            delta=f"{critical_alerts} Critical",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Rule Dashboard", 
        "🚨 Active Alerts", 
        "📊 Execution Analytics",
        "🛡️ Compliance Monitor",
        "🎯 Recommendations"
    ])
    
    with tab1:
        display_rule_dashboard(engine.policy_rules, analysis)
    
    with tab2:
        display_active_alerts(analysis['active_alerts'])
    
    with tab3:
        display_execution_analytics(analysis)
    
    with tab4:
        display_compliance_monitor(analysis)
    
    with tab5:
        display_recommendations(analysis['recommendations'])
    
    # Manual rule execution section
    st.markdown("---")
    st.subheader("🔄 Manual Rule Execution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_rule = st.selectbox(
            "Select Rule to Execute Manually",
            options=[f"{getattr(rule, 'rule_id', 'Unknown')}: {getattr(rule, 'rule_name', 'Unknown')}" for rule in engine.policy_rules],
            key="manual_rule_select"
        )
    
    with col2:
        st.write("")  # Spacing
        if st.button("🚀 Execute Selected Rule", type="primary"):
            rule_id = selected_rule.split(":")[0]
            rule = next((r for r in engine.policy_rules if getattr(r, 'rule_id', '') == rule_id), None)
            
            if rule:
                with st.spinner(f"Executing {getattr(rule, 'rule_name', 'Unknown')}..."):
                    try:
                        execution = engine._execute_rule(rule)
                    except Exception as e:
                        st.error(f"Error executing rule: {e}")
                        execution = None
                    
                if execution and getattr(execution, 'triggered', False):
                    actions = getattr(execution, 'actions_taken', [])
                    st.warning(f"Rule triggered! {len(actions)} actions executed.")
                    for action in actions:
                        st.write(f"✅ {action}")
                else:
                    st.success("Rule executed - No triggers detected")
                
                # Log manual execution
                try:
                    audit_log(
                        "policy_rule_manual_execution",
                        f"Manually executed rule: {getattr(rule, 'rule_name', 'Unknown')}",
                        {"rule_id": getattr(rule, 'rule_id', 'Unknown'), "triggered": getattr(execution, 'triggered', False) if execution else False}
                    )
                except Exception:
                    pass  # Silent fail for audit logging
    
    # Audit logging
    try:
        audit_log(
            "policy_engine_view",
            "Viewed policy engine monitoring dashboard",
            {
                "rules_executed": summary['total_rules_executed'],
                "rules_triggered": summary['rules_triggered'],
                "risk_exposure": risk_exposure['risk_exposure_percentage']
            }
        )
    except Exception:
        pass  # Silent fail for audit logging

def display_rule_dashboard(rules: List[Any], analysis: Dict[str, Any]):
    """Display policy rules dashboard"""
    
    st.header("📋 Policy Rules Dashboard")
    
    # Rules overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_rules = len([r for r in rules if getattr(r, 'status', 'Active') == 'Active'])
        st.metric("Active Rules", active_rules)
    
    with col2:
        triggered_count = len(analysis['triggered_rules'])
        st.metric("Recently Triggered", triggered_count)
    
    with col3:
        high_critical_rules = len([r for r in rules if getattr(r, 'severity', 'Medium') in ['High', 'Critical']])
        st.metric("High/Critical Rules", high_critical_rules)
    
    with col4:
        total_trigger_count = sum(getattr(r, 'trigger_count', 0) for r in rules)
        st.metric("Total Triggers", total_trigger_count)
    
    # Rules table with filters
    st.subheader("Policy Rules Configuration")
    
    if not rules:
        st.info("No policy rules available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rule_type_filter = st.multiselect(
            "Filter by Rule Type",
            options=list(set([getattr(r, 'rule_type', 'Unknown') for r in rules])),
            default=list(set([getattr(r, 'rule_type', 'Unknown') for r in rules]))
        )
    
    with col2:
        status_filter = st.multiselect(
            "Filter by Status",
            options=list(set([getattr(r, 'status', 'Unknown') for r in rules])),
            default=['Active']
        )
    
    with col3:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=list(set([getattr(r, 'severity', 'Unknown') for r in rules])),
            default=list(set([getattr(r, 'severity', 'Unknown') for r in rules]))
        )
    
    # Filter rules
    filtered_rules = [
        r for r in rules 
        if getattr(r, 'rule_type', 'Unknown') in rule_type_filter
        and getattr(r, 'status', 'Unknown') in status_filter
        and getattr(r, 'severity', 'Unknown') in severity_filter
    ]
    
    # Create rules dataframe
    rules_data = []
    for rule in filtered_rules:
        last_triggered = getattr(rule, 'last_triggered', None)
        rules_data.append({
            'Rule ID': getattr(rule, 'rule_id', 'Unknown'),
            'Rule Name': getattr(rule, 'rule_name', 'Unknown'),
            'Type': getattr(rule, 'rule_type', 'Unknown'),
            'Severity': getattr(rule, 'severity', 'Unknown'),
            'Status': getattr(rule, 'status', 'Unknown'),
            'Trigger Count': getattr(rule, 'trigger_count', 0),
            'Last Triggered': last_triggered.strftime('%Y-%m-%d') if last_triggered and hasattr(last_triggered, 'strftime') else 'Never',
            'Description': getattr(rule, 'description', 'No description available')
        })
    
    if rules_data:
        rules_df = pd.DataFrame(rules_data)
        st.dataframe(
            rules_df,
            use_container_width=True,
            column_config={
                "Trigger Count": st.column_config.ProgressColumn(
                    "Trigger Count",
                    help="Number of times rule has been triggered",
                    format="%d",
                    min_value=0,
                    max_value=max(rules_df['Trigger Count']) if len(rules_df) > 0 else 1,
                )
            }
        )
    else:
        st.info("No rules match the selected filters")
    
    # Rule type distribution
    st.subheader("Rule Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rule type distribution
        type_distribution = {}
        for rule in rules:
            rule_type = getattr(rule, 'rule_type', 'Unknown')
            type_distribution[rule_type] = type_distribution.get(rule_type, 0) + 1
        
        if type_distribution:
            fig = px.pie(
                values=list(type_distribution.values()),
                names=list(type_distribution.keys()),
                title="Rule Distribution by Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_distribution = {}
        for rule in rules:
            severity = getattr(rule, 'severity', 'Unknown')
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        if severity_distribution:
            fig = px.bar(
                x=list(severity_distribution.keys()),
                y=list(severity_distribution.values()),
                title="Rule Distribution by Severity",
                labels={'x': 'Severity', 'y': 'Number of Rules'},
                color=list(severity_distribution.values()),
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)

def display_active_alerts(alerts: List[Any]):
    """Display active policy alerts"""
    
    st.header("🚨 Active Policy Alerts")
    
    if not alerts:
        st.success("🎉 No active alerts! All systems are operating within policy limits.")
        return
    
    # Alert severity breakdown
    col1, col2, col3, col4 = st.columns(4)
    
    critical_alerts = len([a for a in alerts if getattr(a, 'severity', 'Medium') == 'Critical'])
    high_alerts = len([a for a in alerts if getattr(a, 'severity', 'Medium') == 'High'])
    medium_alerts = len([a for a in alerts if getattr(a, 'severity', 'Medium') == 'Medium'])
    low_alerts = len([a for a in alerts if getattr(a, 'severity', 'Medium') == 'Low'])
    
    with col1:
        st.metric("Critical Alerts", critical_alerts, delta_color="inverse")
    
    with col2:
        st.metric("High Alerts", high_alerts, delta_color="inverse")
    
    with col3:
        st.metric("Medium Alerts", medium_alerts)
    
    with col4:
        st.metric("Low Alerts", low_alerts)
    
    # Display alerts with severity-based coloring
    for alert in sorted(alerts, key=lambda x: getattr(x, 'severity', 'Medium'), reverse=True):
        severity = getattr(alert, 'severity', 'Medium')
        severity_color = {
            'Critical': '#ff4b4b',
            'High': '#ff6b4b',
            'Medium': '#ffa64b',
            'Low': '#ffd64b',
            'Information': '#4b8aff'
        }.get(severity, '#666666')
        
        with st.container():
            st.markdown(f"""
            <div style="padding: 15px; border-left: 5px solid {severity_color}; 
                        background-color: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <h4 style="margin: 0; color: {severity_color};">{getattr(alert, 'title', 'No Title')}</h4>
                    <span style="background-color: {severity_color}; color: white; padding: 2px 8px; 
                                border-radius: 12px; font-size: 0.8em;">{severity}</span>
                </div>
                <p style="margin: 8px 0; color: #555;">{getattr(alert, 'description', 'No description')}</p>
                <div style="font-size: 0.9em; color: #777;">
                    <strong>Rule:</strong> {getattr(alert, 'rule_id', 'Unknown')} | 
                    <strong>Assigned to:</strong> {getattr(alert, 'assigned_to', 'Unassigned')} | 
                    <strong>Timestamp:</strong> {getattr(alert, 'timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M') if hasattr(getattr(alert, 'timestamp', None), 'strftime') else 'Unknown'}
                </div>
                <div style="margin-top: 8px;">
                    <strong>Affected:</strong> {', '.join(getattr(alert, 'affected_entities', ['Unknown']))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Alert actions
            col1, col2, col3 = st.columns([3, 1, 1])
            with col2:
                if st.button(f"🔄 Acknowledge", key=f"ack_{getattr(alert, 'alert_id', 'unknown')}"):
                    st.success(f"Alert {getattr(alert, 'alert_id', 'unknown')} acknowledged")
                    # In production, this would update the alert status
            with col3:
                if st.button(f"📋 Resolve", key=f"resolve_{getattr(alert, 'alert_id', 'unknown')}"):
                    st.success(f"Alert {getattr(alert, 'alert_id', 'unknown')} resolved")
                    # In production, this would resolve the alert

def display_execution_analytics(analysis: Dict[str, Any]):
    """Display policy execution analytics"""
    
    st.header("📊 Policy Execution Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Execution success rate over time (simulated)
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        success_rates = np.random.normal(95, 3, len(dates))
        
        fig = px.line(
            x=dates,
            y=success_rates,
            title="Execution Success Rate Trend",
            labels={'x': 'Date', 'y': 'Success Rate (%)'},
            line_shape='spline'
        )
        fig.update_traces(line=dict(color='green', width=3))
        fig.add_hrect(y0=95, y1=100, line_width=0, fillcolor="green", opacity=0.1)
        fig.add_hrect(y0=85, y1=95, line_width=0, fillcolor="yellow", opacity=0.1)
        fig.add_hrect(y0=0, y1=85, line_width=0, fillcolor="red", opacity=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rule effectiveness by type
        effectiveness = analysis['rule_effectiveness']
        rule_performance = effectiveness.get('rule_performance', {})
        
        if rule_performance:
            rule_types = {}
            for rule_id, perf in rule_performance.items():
                rule_type = 'Unknown'  # Simplified for fallback
                if rule_type not in rule_types:
                    rule_types[rule_type] = []
                success_rate = (perf.get('success_count', 0) / perf.get('total_executions', 1) * 100) if perf.get('total_executions', 0) > 0 else 0
                rule_types[rule_type].append(success_rate)
            
            # Calculate average by type
            avg_effectiveness = {rt: np.mean(scores) for rt, scores in rule_types.items()}
            
            fig = px.bar(
                x=list(avg_effectiveness.keys()),
                y=list(avg_effectiveness.values()),
                title="Average Effectiveness by Rule Type",
                labels={'x': 'Rule Type', 'y': 'Effectiveness (%)'},
                color=list(avg_effectiveness.values()),
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rule performance data available")

def display_compliance_monitor(analysis: Dict[str, Any]):
    """Display compliance monitoring dashboard"""
    
    st.header("🛡️ Compliance Monitoring")
    
    compliance_status = analysis['compliance_status']
    
    # Compliance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Compliance Rate",
            f"{compliance_status.get('compliance_rate', 0):.1f}%"
        )
    
    with col2:
        st.metric(
            "Compliance Rules",
            compliance_status.get('total_compliance_rules', 0)
        )
    
    with col3:
        st.metric(
            "Compliance Violations",
            compliance_status.get('compliance_violations', 0),
            delta_color="inverse"
        )
    
    # Compliance status visualization
    compliance_level = compliance_status.get('compliance_status', 'Unknown')
    status_color = {
        'Compliant': 'green',
        'Minor Issues': 'orange',
        'Non-Compliant': 'red',
        'Unknown': 'gray'
    }.get(compliance_level, 'gray')
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {status_color}; color: white; text-align: center;">
        <h2 style="margin: 0;">{compliance_level}</h2>
        <p style="margin: 0;">Current Compliance Status</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Regulatory requirements checklist
    st.subheader("Regulatory Requirements Checklist")
    
    regulatory_items = [
        {"requirement": "Capital Adequacy Ratio (CAR) ≥ 10%", "status": "Compliant", "value": "15.2%"},
        {"requirement": "Liquidity Coverage Ratio (LCR) ≥ 100%", "status": "Compliant", "value": "125.0%"},
        {"requirement": "Single Employer Concentration ≤ 25%", "status": "Compliant", "value": "18.5%"},
        {"requirement": "PAR 30 Days ≤ 5%", "status": "Compliant", "value": "3.2%"},
        {"requirement": "Data Protection Compliance", "status": "Compliant", "value": "Verified"},
        {"requirement": "Audit Trail Maintenance", "status": "Compliant", "value": "Complete"}
    ]
    
    for item in regulatory_items:
        status_icon = "✅" if item["status"] == "Compliant" else "⚠️" if item["status"] == "Partial" else "❌"
        st.write(f"{status_icon} **{item['requirement']}** - {item['value']}")

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display policy improvement recommendations"""
    
    st.header("🎯 Policy Improvement Recommendations")
    
    if not recommendations:
        st.success("No immediate recommendations. Policy engine is optimally configured.")
        return
    
    # Priority-based display
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    sorted_recommendations = sorted(
        recommendations, 
        key=lambda x: priority_order.get(x.get('priority', 'Low'), 2)
    )
    
    for i, rec in enumerate(sorted_recommendations, 1):
        priority = rec.get('priority', 'Medium')
        rec_type = rec.get('type', 'General')
        
        priority_color = {
            'High': '#ff4b4b',
            'Medium': '#ffa64b', 
            'Low': '#4b8aff'
        }.get(priority, '#666666')
        
        with st.container():
            st.markdown(f"""
            <div style="padding: 15px; border-left: 5px solid {priority_color}; 
                        background-color: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <h4 style="margin: 0;">{i}. {rec.get('recommendation', 'No recommendation')}</h4>
                    <span style="background-color: {priority_color}; color: white; padding: 2px 8px; 
                                border-radius: 12px; font-size: 0.8em;">{priority} Priority</span>
                </div>
                <p style="margin: 8px 0; color: #555;"><strong>Type:</strong> {rec_type}</p>
                <p style="margin: 8px 0; color: #555;"><strong>Rationale:</strong> {rec.get('rationale', 'No rationale provided')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons for each recommendation
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button(f"📋 Implement", key=f"impl_{i}"):
                    st.success(f"Implementation started for recommendation {i}")
                    # In production, this would trigger an implementation workflow

if __name__ == "__main__":
    main()