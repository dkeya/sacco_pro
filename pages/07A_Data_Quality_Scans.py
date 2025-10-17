# pages/07A_Data_Quality_Scans.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sacco_core.config import ConfigManager
from sacco_core.rbac import RBACManager
from sacco_core.audit import AuditLogger
from sacco_core.analytics.dq_scans import DataQualityScanner
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Data Quality Scans",
    page_icon="🔍",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class DataQualityScansPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.dq_scanner = DataQualityScanner()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "07A_Data_Quality_Scans.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "data_quality_scans_page"
        )
        return True
    
    def render_scan_dashboard(self):
        """Render data quality scanning dashboard"""
        st.subheader("🔍 Automated Data Quality Scans")
        
        # Get scan statistics from scanner
        scan_stats = self.dq_scanner.get_scan_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Scans",
                f"{scan_stats.get('active_scans', 0)}",
                delta=f"+{scan_stats.get('new_scans_week', 0)} this week",
                help="Number of active data quality monitoring scans"
            )
        
        with col2:
            st.metric(
                "Issues Detected", 
                f"{scan_stats.get('total_issues', 0)}",
                delta=f"{scan_stats.get('issues_change', 0)} from last scan",
                delta_color="inverse" if scan_stats.get('issues_change', 0) > 0 else "normal",
                help="Total data quality issues detected"
            )
        
        with col3:
            st.metric(
                "Scan Coverage",
                f"{scan_stats.get('coverage', 0)}%",
                delta=f"+{scan_stats.get('coverage_change', 0)}%",
                help="Percentage of data assets covered by scans"
            )
        
        with col4:
            st.metric(
                "Avg. Scan Time",
                f"{scan_stats.get('avg_scan_time', 0):.1f}s",
                delta=f"{scan_stats.get('time_change', 0):.1f}s",
                help="Average time to complete a data quality scan"
            )
        
        # Recent scan activity
        self.render_recent_scans()
    
    def render_recent_scans(self):
        """Render recent scan activity"""
        st.markdown("#### 📊 Recent Scan Activity")
        
        # Get scan trends from scanner
        scan_trends = self.dq_scanner.get_scan_trends()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scan success rate over time
            fig = px.line(
                scan_trends,
                x='date',
                y='success_rate',
                title="Scan Success Rate Trend",
                markers=True
            )
            fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Issues by severity over time
            fig = px.area(
                scan_trends,
                x='date',
                y=['critical_issues', 'high_issues', 'medium_issues'],
                title="Data Issues Trend by Severity",
                labels={'value': 'Issue Count', 'variable': 'Severity'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick scan actions
        self.render_quick_scan_actions()
    
    def render_quick_scan_actions(self):
        """Render quick scan action buttons"""
        st.markdown("#### 🚀 Quick Scan Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Run Full Scan", use_container_width=True):
                scan_result = self.dq_scanner.run_comprehensive_scan()
                st.session_state.last_scan_result = scan_result
                st.success("Full data quality scan completed successfully!")
        
        with col2:
            if st.button("📊 Profile Data", use_container_width=True):
                profile_result = self.dq_scanner.run_data_profiling_scan()
                st.session_state.profile_result = profile_result
                st.success("Data profiling completed!")
        
        with col3:
            if st.button("🔍 Validate Rules", use_container_width=True):
                validation_result = self.dq_scanner.run_validation_scan()
                st.session_state.validation_result = validation_result
                st.success("Validation rules executed successfully!")
        
        with col4:
            if st.button("🧹 Clean Data", use_container_width=True):
                cleaning_result = self.dq_scanner.run_data_cleaning_scan()
                st.session_state.cleaning_result = cleaning_result
                st.success("Data cleaning completed!")
    
    def render_scan_configuration(self):
        """Render scan configuration interface"""
        st.markdown("---")
        st.subheader("⚙️ Scan Configuration")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📅 Scheduled Scans", "🎯 Custom Rules", "🔔 Alert Settings", "📋 Scan Templates"
        ])
        
        with tab1:
            self.render_scheduled_scans()
        
        with tab2:
            self.render_custom_rules()
        
        with tab3:
            self.render_alert_settings()
        
        with tab4:
            self.render_scan_templates()
    
    def render_scheduled_scans(self):
        """Render scheduled scans configuration"""
        st.markdown("#### 📅 Scheduled Data Quality Scans")
        
        # Get scheduled scans from scanner
        scheduled_scans = self.dq_scanner.get_scheduled_scans()
        
        for scan in scheduled_scans:
            status_color = "🟢" if scan['status'] == 'Active' else "🔴"
            
            with st.expander(f"{status_color} {scan['name']} - {scan['frequency']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Schedule**: {scan['time']}")
                    st.write(f"**Tables**: {scan['tables']}")
                
                with col2:
                    st.write(f"**Last Run**: {scan['last_run']}")
                    st.write(f"**Next Run**: {scan['next_run']}")
                
                with col3:
                    if st.button("🔄 Run Now", key=f"run_{scan['name']}"):
                        result = self.dq_scanner.run_scheduled_scan(scan['name'])
                        if result['success']:
                            st.success(f"Scan '{scan['name']}' completed!")
                        else:
                            st.error(f"Scan failed: {result['error']}")
                    
                    if scan['status'] == 'Active':
                        if st.button("⏸️ Pause", key=f"pause_{scan['name']}"):
                            if self.dq_scanner.pause_scan(scan['name']):
                                st.success(f"Scan '{scan['name']}' paused")
                            else:
                                st.error("Failed to pause scan")
                    else:
                        if st.button("▶️ Activate", key=f"activate_{scan['name']}"):
                            if self.dq_scanner.activate_scan(scan['name']):
                                st.success(f"Scan '{scan['name']}' activated")
                            else:
                                st.error("Failed to activate scan")
        
        # Add new scheduled scan
        st.markdown("#### ➕ Add New Scheduled Scan")
        
        with st.form("new_scheduled_scan"):
            col1, col2 = st.columns(2)
            
            with col1:
                scan_name = st.text_input("Scan Name")
                frequency = st.selectbox(
                    "Frequency",
                    ["Hourly", "Daily", "Weekly", "Monthly", "Custom"]
                )
            
            with col2:
                tables = st.multiselect(
                    "Tables to Scan",
                    ["members", "loans", "employers", "deposits", "transactions", "All Tables"],
                    default=["members", "loans"]
                )
                enabled = st.checkbox("Enable immediately", value=True)
            
            if st.form_submit_button("💾 Save Scheduled Scan"):
                scan_config = {
                    'name': scan_name,
                    'frequency': frequency,
                    'tables': tables,
                    'enabled': enabled
                }
                if self.dq_scanner.save_scheduled_scan(scan_config):
                    st.success(f"Scheduled scan '{scan_name}' saved successfully!")
                else:
                    st.error("Failed to save scheduled scan")
    
    def render_custom_rules(self):
        """Render custom data quality rules configuration"""
        st.markdown("#### 🎯 Custom Data Quality Rules")
        
        # Get custom rules from scanner
        custom_rules = self.dq_scanner.get_custom_rules()
        
        for rule in custom_rules:
            status_color = "🟢" if rule['status'] == 'Active' else "🔴"
            
            with st.expander(f"{status_color} {rule['name']} - {rule['table']} - Severity: {rule['severity']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description**: {rule['description']}")
                    st.write(f"**Condition**: `{rule['condition']}`")
                
                with col2:
                    st.write(f"**Status**: {rule['status']}")
                    st.write(f"**Rule ID**: {rule['id']}")
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        if st.button("Test Rule", key=f"test_{rule['id']}"):
                            result = self.dq_scanner.test_custom_rule(rule['id'])
                            if result['success']:
                                st.success(f"Rule test passed: {result['message']}")
                            else:
                                st.error(f"Rule test failed: {result['message']}")
                    
                    with col2b:
                        if rule['status'] == 'Active':
                            if st.button("Deactivate", key=f"deactivate_{rule['id']}"):
                                if self.dq_scanner.deactivate_rule(rule['id']):
                                    st.success("Rule deactivated")
                                else:
                                    st.error("Failed to deactivate rule")
                        else:
                            if st.button("Activate", key=f"activate_{rule['id']}"):
                                if self.dq_scanner.activate_rule(rule['id']):
                                    st.success("Rule activated")
                                else:
                                    st.error("Failed to activate rule")
        
        # Add new custom rule
        st.markdown("#### ➕ Create New Custom Rule")
        
        with st.form("new_custom_rule"):
            col1, col2 = st.columns(2)
            
            with col1:
                rule_name = st.text_input("Rule Name")
                description = st.text_area("Description")
                table = st.selectbox(
                    "Table",
                    ["members", "loans", "employers", "deposits", "transactions"]
                )
            
            with col2:
                condition = st.text_area("SQL Condition", placeholder="column_name operator value")
                severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
                active = st.checkbox("Activate immediately", value=True)
            
            if st.form_submit_button("💾 Save Custom Rule"):
                rule_config = {
                    'name': rule_name,
                    'description': description,
                    'table': table,
                    'condition': condition,
                    'severity': severity,
                    'active': active
                }
                result = self.dq_scanner.save_custom_rule(rule_config)
                if result['success']:
                    st.success(f"Custom rule '{rule_name}' saved successfully!")
                else:
                    st.error(f"Failed to save rule: {result['error']}")
    
    def render_alert_settings(self):
        """Render alert configuration settings"""
        st.markdown("#### 🔔 Data Quality Alert Settings")
        
        # Get current alert settings
        alert_settings = self.dq_scanner.get_alert_settings()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📧 Email Alerts")
            
            email_settings = alert_settings.get('email', {})
            
            email_enabled = st.checkbox("Enable Email Alerts", value=email_settings.get('enabled', True))
            if email_enabled:
                critical_emails = st.checkbox("Critical Issues", value=email_settings.get('critical_issues', True))
                high_emails = st.checkbox("High Issues", value=email_settings.get('high_issues', True))
                medium_emails = st.checkbox("Medium Issues", value=email_settings.get('medium_issues', False))
                daily_summary = st.checkbox("Daily Summary", value=email_settings.get('daily_summary', True))
                
                recipients = st.text_input(
                    "Recipients (comma-separated)",
                    value=", ".join(email_settings.get('recipients', []))
                )
        
        with col2:
            st.markdown("##### 📱 Notification Channels")
            
            # SMS alerts
            sms_settings = alert_settings.get('sms', {})
            sms_enabled = st.checkbox("SMS Alerts", value=sms_settings.get('enabled', False))
            if sms_enabled:
                sms_recipients = st.text_input(
                    "SMS Numbers (comma-separated)",
                    value=", ".join(sms_settings.get('recipients', []))
                )
                sms_critical_only = st.checkbox("Critical issues only", 
                                              value=sms_settings.get('critical_only', True))
            
            # Alert thresholds
            st.markdown("##### 📊 Alert Thresholds")
            thresholds = alert_settings.get('thresholds', {})
            critical_threshold = st.slider("Critical Issues Threshold", 1, 10, 
                                         value=thresholds.get('critical', 1))
            high_threshold = st.slider("High Issues Threshold", 1, 20, 
                                     value=thresholds.get('high', 5))
        
        if st.button("💾 Save Alert Settings"):
            new_settings = {
                'email': {
                    'enabled': email_enabled,
                    'critical_issues': critical_emails,
                    'high_issues': high_emails,
                    'medium_issues': medium_emails,
                    'daily_summary': daily_summary,
                    'recipients': [r.strip() for r in recipients.split(',')] if recipients else []
                },
                'sms': {
                    'enabled': sms_enabled,
                    'recipients': [r.strip() for r in sms_recipients.split(',')] if sms_enabled and sms_recipients else [],
                    'critical_only': sms_critical_only if sms_enabled else True
                },
                'thresholds': {
                    'critical': critical_threshold,
                    'high': high_threshold
                }
            }
            
            if self.dq_scanner.save_alert_settings(new_settings):
                st.success("Alert settings saved successfully!")
            else:
                st.error("Failed to save alert settings")
    
    def render_scan_templates(self):
        """Render scan templates configuration"""
        st.markdown("#### 📋 Data Quality Scan Templates")
        
        templates = self.dq_scanner.get_scan_templates()
        
        cols = st.columns(2)
        for i, template in enumerate(templates):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{template['name']}**")
                    st.write(template['description'])
                    st.write(f"**Checks**: {', '.join(template['checks'])}")
                    st.write(f"**Tables**: {template['tables']}")
                    st.write(f"**Duration**: {template['duration']}")
                    
                    if st.button("🚀 Use Template", key=f"use_{template['name']}"):
                        result = self.dq_scanner.apply_scan_template(template['name'])
                        if result['success']:
                            st.success(f"Template '{template['name']}' applied successfully!")
                        else:
                            st.error(f"Failed to apply template: {result['error']}")
    
    def render_scan_results(self):
        """Render detailed scan results and history"""
        st.markdown("---")
        st.subheader("📈 Scan Results & History")
        
        # Get scan history
        scan_history = self.dq_scanner.get_scan_history()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.selectbox(
                "Date Range",
                ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"]
            )
        
        with col2:
            scan_type = st.multiselect(
                "Scan Type",
                ["Full Scan", "Quick Scan", "Custom Scan", "Scheduled Scan"],
                default=["Full Scan", "Scheduled Scan"]
            )
        
        with col3:
            status_filter = st.multiselect(
                "Status",
                ["Completed", "Failed", "Running", "Scheduled"],
                default=["Completed"]
            )
        
        # Filter scan history
        filtered_history = scan_history[
            (scan_history['scan_type'].isin(scan_type)) &
            (scan_history['status'].isin(status_filter))
        ]
        
        # Scan results table
        st.dataframe(filtered_history, use_container_width=True)
        
        # Detailed scan analysis
        if not filtered_history.empty:
            selected_scan = st.selectbox(
                "Select Scan for Detailed Analysis",
                filtered_history['scan_id'].tolist()
            )
            
            if selected_scan:
                self.render_scan_details(selected_scan)
    
    def render_scan_details(self, scan_id):
        """Render detailed analysis of a specific scan"""
        st.markdown(f"#### 🔍 Detailed Analysis: {scan_id}")
        
        # Get scan details from scanner
        scan_details = self.dq_scanner.get_scan_details(scan_id)
        
        if not scan_details:
            st.error("Scan details not found")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary", "🚨 Issues", "✅ Validations", "📋 Recommendations"
        ])
        
        with tab1:
            self.render_scan_summary(scan_details)
        
        with tab2:
            self.render_scan_issues(scan_details)
        
        with tab3:
            self.render_validation_results(scan_details)
        
        with tab4:
            self.render_scan_recommendations(scan_details)
    
    def render_scan_summary(self, scan_details):
        """Render scan summary"""
        summary = scan_details.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{summary.get('score', 0)}%")
        
        with col2:
            st.metric("Issues Found", summary.get('issues_found', 0))
        
        with col3:
            st.metric("Tables Scanned", summary.get('tables_scanned', 0))
        
        with col4:
            st.metric("Scan Duration", f"{summary.get('duration', 0)}s")
        
        # Quality scores by table
        st.markdown("##### 📊 Quality Scores by Table")
        table_scores = scan_details.get('table_scores', [])
        
        if table_scores:
            scores_df = pd.DataFrame(table_scores)
            fig = px.bar(
                scores_df,
                x='table',
                y='score',
                title="Data Quality Scores by Table",
                color='score',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_scan_issues(self, scan_details):
        """Render issues found in scan"""
        issues = scan_details.get('issues', [])
        
        if issues:
            issues_df = pd.DataFrame(issues)
            
            # Issue severity distribution
            col1, col2 = st.columns(2)
            
            with col1:
                severity_counts = issues_df['severity'].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Issue Distribution by Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                table_counts = issues_df['table'].value_counts().head(10)
                fig = px.bar(
                    x=table_counts.values,
                    y=table_counts.index,
                    orientation='h',
                    title="Top 10 Tables with Issues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed issues table
            st.markdown("##### 📋 Detailed Issues")
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("🎉 No issues found in this scan!")
    
    def render_validation_results(self, scan_details):
        """Render validation rule results"""
        validations = scan_details.get('validations', [])
        
        if validations:
            validation_df = pd.DataFrame(validations)
            
            # Validation status summary
            status_counts = validation_df['status'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Passing", status_counts.get('Pass', 0))
            with col2:
                st.metric("Failing", status_counts.get('Fail', 0))
            with col3:
                st.metric("Total Rules", len(validations))
            
            # Validation results table
            st.dataframe(validation_df, use_container_width=True)
        else:
            st.info("No validation rules executed in this scan.")
    
    def render_scan_recommendations(self, scan_details):
        """Render scan recommendations"""
        recommendations = scan_details.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. **{rec['type']}**: {rec['description']}")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"   **Impact**: {rec['impact']}")
                with col2:
                    if st.button("Implement", key=f"impl_{i}"):
                        result = self.dq_scanner.implement_recommendation(rec)
                        if result['success']:
                            st.success(f"Implemented: {rec['description']}")
                        else:
                            st.error(f"Failed to implement: {result['error']}")
        else:
            st.success("🎉 No recommendations - data quality is excellent!")
    
    def run(self):
        """Run the data quality scans page"""
        st.title("🔍 Data Quality Scans")
        
        st.markdown("""
        Automated data quality scanning, monitoring, and alerting system.
        Configure scheduled scans, custom rules, and receive proactive alerts when data quality issues are detected.
        """)
        
        self.render_scan_dashboard()
        self.render_scan_configuration()
        self.render_scan_results()

if __name__ == "__main__":
    page = DataQualityScansPage()
    page.run()