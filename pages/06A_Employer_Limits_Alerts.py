# pages/06A_Employer_Limits_Alerts.py
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
from sacco_core.analytics.concentration import ConcentrationAnalyzer

st.set_page_config(
    page_title="Employer Limits & Alerts",
    page_icon="🚨",
    layout="wide"
)

class EmployerLimitsAlertsPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.concentration_analyzer = ConcentrationAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "06A_Employer_Limits_Alerts.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "employer_limits_alerts_page"
        )
        return True
    
    def render_limits_dashboard(self):
        """Render employer limits dashboard"""
        st.subheader("📊 Employer Limits Dashboard")
        
        # Get current employer analysis
        analysis = self.concentration_analyzer.analyze_employer_concentration()
        employer_exposures = analysis.get('employer_exposures', [])
        
        # Calculate key metrics
        total_breaches = len([e for e in employer_exposures if e.get('exposure_share', 0) > self.config.limits.single_employer_share_max])
        warning_employers = len([e for e in employer_exposures if 0.8 * self.config.limits.single_employer_share_max < e.get('exposure_share', 0) <= self.config.limits.single_employer_share_max])
        total_exposure_at_risk = sum([e.get('outstanding_amount', 0) for e in employer_exposures if e.get('exposure_share', 0) > self.config.limits.single_employer_share_max])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Limit Breaches",
                f"{total_breaches}",
                delta_color="inverse" if total_breaches > 0 else "normal",
                help="Employers exceeding single employer limit"
            )
        
        with col2:
            st.metric(
                "Warning Employers", 
                f"{warning_employers}",
                help="Employers approaching limit (80-100%)"
            )
        
        with col3:
            st.metric(
                "Exposure at Risk",
                f"KES {total_exposure_at_risk:,.0f}",
                delta_color="inverse" if total_exposure_at_risk > 0 else "normal",
                help="Total exposure in breach of limits"
            )
        
        with col4:
            compliance_rate = ((len(employer_exposures) - total_breaches) / len(employer_exposures)) * 100 if employer_exposures else 100
            st.metric(
                "Compliance Rate",
                f"{compliance_rate:.1f}%",
                help="Percentage of employers within limits"
            )
        
        # Real-time alerts
        self.render_real_time_alerts(employer_exposures)
    
    def render_real_time_alerts(self, employer_exposures):
        """Render real-time limit breach alerts"""
        st.markdown("#### 🚨 Real-Time Limit Breach Alerts")
        
        # Identify breaches and warnings
        breaches = []
        warnings = []
        
        for employer in employer_exposures:
            exposure_share = employer.get('exposure_share', 0)
            if exposure_share > self.config.limits.single_employer_share_max:
                breaches.append(employer)
            elif exposure_share > 0.8 * self.config.limits.single_employer_share_max:
                warnings.append(employer)
        
        # Display breaches
        if breaches:
            st.error("### 🔴 ACTIVE BREACHES")
            for breach in breaches:
                with st.expander(f"🚨 {breach['employer_name']} - {breach['exposure_share']*100:.1f}% (Limit: {self.config.limits.single_employer_share_max*100:.1f}%)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Exposure",
                            f"KES {breach.get('outstanding_amount', 0):,.0f}",
                            help="Total outstanding exposure"
                        )
                    
                    with col2:
                        st.metric(
                            "Excess Over Limit",
                            f"KES {(breach.get('outstanding_amount', 0) * (breach['exposure_share'] - self.config.limits.single_employer_share_max)):,.0f}",
                            delta_color="inverse",
                            help="Amount exceeding regulatory limit"
                        )
                    
                    with col3:
                        st.metric(
                            "Breach Duration",
                            "7 days",
                            help="How long this breach has been active"
                        )
                    
                    # Action buttons
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        if st.button("📧 Notify Risk Team", key=f"notify_{breach['employer_name']}"):
                            self._send_notification(breach, "Risk Team")
                            st.success("Risk team notified!")
                    
                    with col5:
                        if st.button("📋 Create Action Plan", key=f"plan_{breach['employer_name']}"):
                            self._create_action_plan(breach)
                            st.success("Action plan created!")
                    
                    with col6:
                        if st.button("⚡ Request Waiver", key=f"waiver_{breach['employer_name']}"):
                            self._request_waiver(breach)
                            st.success("Waiver request submitted!")
        
        # Display warnings
        if warnings:
            st.warning("### 🟡 APPROACHING LIMITS")
            for warning in warnings:
                with st.expander(f"⚠️ {warning['employer_name']} - {warning['exposure_share']*100:.1f}% (Limit: {self.config.limits.single_employer_share_max*100:.1f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Current Exposure",
                            f"KES {warning.get('outstanding_amount', 0):,.0f}",
                            help="Total outstanding exposure"
                        )
                    
                    with col2:
                        buffer = self.config.limits.single_employer_share_max - warning['exposure_share']
                        st.metric(
                            "Buffer Remaining",
                            f"{buffer*100:.2f}%",
                            help="Remaining capacity before breach"
                        )
                    
                    # Preventive actions
                    if st.button("🛡️ Initiate Preventive Measures", key=f"prevent_{warning['employer_name']}"):
                        self._initiate_preventive_measures(warning)
                        st.success("Preventive measures initiated!")
        
        if not breaches and not warnings:
            st.success("### ✅ ALL EMPLOYERS WITHIN LIMITS")
            st.info("No active limit breaches or warnings detected.")
    
    def render_employer_limits_monitor(self):
        """Render employer limits monitoring table"""
        st.markdown("---")
        st.subheader("📋 Employer Limits Monitoring")
        
        # Get employer data
        analysis = self.concentration_analyzer.analyze_employer_concentration()
        employer_exposures = analysis.get('employer_exposures', [])
        
        if employer_exposures:
            # Create monitoring dataframe
            monitor_data = []
            for employer in employer_exposures:
                exposure_share = employer.get('exposure_share', 0)
                limit_utilization = (exposure_share / self.config.limits.single_employer_share_max) * 100
                
                # Determine status
                if exposure_share > self.config.limits.single_employer_share_max:
                    status = "BREACH"
                    alert_level = "High"
                elif exposure_share > 0.8 * self.config.limits.single_employer_share_max:
                    status = "WARNING"
                    alert_level = "Medium"
                else:
                    status = "WITHIN LIMIT"
                    alert_level = "Low"
                
                monitor_data.append({
                    'Employer': employer['employer_name'],
                    'Exposure (KES)': employer.get('outstanding_amount', 0),
                    'Portfolio Share': f"{exposure_share*100:.2f}%",
                    'Limit Utilization': f"{limit_utilization:.1f}%",
                    'Status': status,
                    'Alert Level': alert_level,
                    'Member Count': employer.get('loan_id', 0),
                    'Average DPD': f"{employer.get('average_dpd', 0):.1f}",
                    'Last Review': (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d')
                })
            
            monitor_df = pd.DataFrame(monitor_data)
            
            # Add filtering options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=['WITHIN LIMIT', 'WARNING', 'BREACH'],
                    default=['BREACH', 'WARNING']
                )
            
            with col2:
                alert_filter = st.multiselect(
                    "Filter by Alert Level",
                    options=['Low', 'Medium', 'High'],
                    default=['High', 'Medium']
                )
            
            with col3:
                search_term = st.text_input("Search Employer")
            
            # Apply filters
            filtered_df = monitor_df[
                (monitor_df['Status'].isin(status_filter)) &
                (monitor_df['Alert Level'].isin(alert_filter))
            ]
            
            if search_term:
                filtered_df = filtered_df[filtered_df['Employer'].str.contains(search_term, case=False, na=False)]
            
            # Color coding
            def color_status(status):
                if status == 'BREACH':
                    return 'background-color: #FFB6C1; color: black;'
                elif status == 'WARNING':
                    return 'background-color: #FFE4B5; color: black;'
                else:
                    return 'background-color: #90EE90; color: black;'
            
            def color_alert_level(level):
                if level == 'High':
                    return 'color: red; font-weight: bold;'
                elif level == 'Medium':
                    return 'color: orange; font-weight: bold;'
                else:
                    return 'color: green; font-weight: bold;'
            
            styled_df = filtered_df.style.map(
                color_status, subset=['Status']
            ).map(
                color_alert_level, subset=['Alert Level']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Generate Limits Report"):
                    report = self._generate_limits_report(monitor_df)
                    st.success("Employer limits report generated!")
            
            with col2:
                csv_data = monitor_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Monitoring Data",
                    data=csv_data,
                    file_name=f"employer_limits_monitoring_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    def render_limit_configuration(self):
        """Render employer limit configuration interface"""
        st.markdown("---")
        st.subheader("⚙️ Limit Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📏 Regulatory Limits")
            
            # Display current regulatory limits
            st.info(f"**Current Single Employer Limit**: {self.config.limits.single_employer_share_max * 100:.1f}%")
            st.info(f"**Regulatory Source**: SASRA Prudential Guidelines")
            st.info(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}")
            
            # Internal limit controls
            st.markdown("#### 🎯 Internal Control Limits")
            
            warning_threshold = st.slider(
                "Warning Threshold (% of Limit)",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
                help="Percentage of limit at which warnings are triggered"
            )
            
            review_threshold = st.slider(
                "Management Review Threshold",
                min_value=60,
                max_value=90,
                value=70,
                step=5,
                help="Percentage of limit requiring management review"
            )
        
        with col2:
            st.markdown("#### 🏢 Employer-Specific Limits")
            
            # Get employer list for specific limits
            analysis = self.concentration_analyzer.analyze_employer_concentration()
            employer_exposures = analysis.get('employer_exposures', [])
            
            if employer_exposures:
                selected_employer = st.selectbox(
                    "Select Employer for Custom Limit",
                    options=[e['employer_name'] for e in employer_exposures],
                    help="Set custom limit for specific employer"
                )
                
                # Find selected employer data
                selected_data = next((e for e in employer_exposures if e['employer_name'] == selected_employer), None)
                
                if selected_data:
                    current_share = selected_data.get('exposure_share', 0) * 100
                    current_limit = self.config.limits.single_employer_share_max * 100
                    
                    st.metric("Current Exposure", f"{current_share:.1f}%")
                    st.metric("Standard Limit", f"{current_limit:.1f}%")
                    
                    custom_limit = st.slider(
                        "Custom Limit for Selected Employer",
                        min_value=5.0,
                        max_value=float(current_limit),
                        value=float(min(current_limit, current_share + 5.0)),
                        step=0.5,
                        help="Set custom limit for this employer"
                    )
                    
                    if st.button("💾 Save Custom Limit"):
                        self._save_custom_limit(selected_employer, custom_limit / 100)
                        st.success(f"Custom limit of {custom_limit:.1f}% saved for {selected_employer}")
            
            # Bulk limit management
            st.markdown("#### 📦 Bulk Limit Management")
            
            if st.button("🔄 Apply Standard Limits to All"):
                self._apply_standard_limits()
                st.success("Standard limits applied to all employers")
            
            if st.button("🧹 Clear All Custom Limits"):
                self._clear_custom_limits()
                st.success("All custom limits cleared")
    
    def render_alert_management(self):
        """Render alert management and notification system"""
        st.markdown("---")
        st.subheader("🔔 Alert Management System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📧 Notification Configuration")
            
            # Alert recipients
            primary_recipients = st.multiselect(
                "Primary Alert Recipients",
                options=['risk@example.com', 'credit@example.com', 'ceo@example.com', 'board@example.com'],
                default=['risk@example.com', 'credit@example.com'],
                help="Who receives immediate breach alerts"
            )
            
            # Alert frequency - FIXED: removed 'default' parameter
            alert_frequency_options = ['Immediate', 'Hourly', 'Daily', 'Weekly']
            alert_frequency = st.selectbox(
                "Alert Frequency",
                options=alert_frequency_options,
                index=0,  # Use index instead of default
                help="How often to send alert summaries"
            )
            
            # Escalation rules
            st.markdown("#### ⚡ Escalation Rules")
            
            escalation_days = st.slider(
                "Escalation After (Days)",
                min_value=1,
                max_value=14,
                value=7,
                help="Days before escalating unresolved breaches"
            )
            
            escalation_recipients = st.multiselect(
                "Escalation Recipients",
                options=['ceo@example.com', 'board@example.com', 'compliance@example.com'],
                default=['ceo@example.com'],
                help="Who receives escalation alerts"
            )
        
        with col2:
            st.markdown("#### 📱 Notification Channels")
            
            # Channel configuration
            email_enabled = st.checkbox("📧 Email Notifications", value=True)
            sms_enabled = st.checkbox("💬 SMS Alerts", value=True)
            push_enabled = st.checkbox("🔔 Push Notifications", value=False)
            dashboard_enabled = st.checkbox("📊 Dashboard Alerts", value=True)
            
            # Alert templates
            st.markdown("#### 📝 Alert Templates")
            
            template_options = ['Breach Alert', 'Warning Alert', 'Escalation Alert', 'Resolution Alert']
            template_type = st.selectbox(
                "Select Template Type",
                options=template_options,
                index=0,  # Use index instead of default
                help="Configure alert message templates"
            )
            
            # Template preview
            if template_type == 'Breach Alert':
                template = f"""
                🚨 EMPLOYER LIMIT BREACH ALERT
                
                Employer: {{employer_name}}
                Current Exposure: {{exposure_share:.1f}}%
                Regulatory Limit: {self.config.limits.single_employer_share_max*100:.1f}%
                Breach Amount: KES {{breach_amount:,.0f}}
                
                Immediate action required!
                """
            elif template_type == 'Warning Alert':
                template = """
                ⚠️ EMPLOYER LIMIT WARNING
                
                Employer: {employer_name}
                Current Exposure: {exposure_share:.1f}%
                Limit: {limit:.1f}%
                Buffer Remaining: {buffer:.1f}%
                
                Preventive measures recommended.
                """
            elif template_type == 'Escalation Alert':
                template = """
                ⚡ ESCALATION ALERT - UNRESOLVED BREACH
                
                Employer: {employer_name}
                Breach Duration: {breach_days} days
                Current Exposure: {exposure_share:.1f}%
                Required Action: {required_action}
                
                Management attention required!
                """
            else:  # Resolution Alert
                template = """
                ✅ BREACH RESOLUTION CONFIRMATION
                
                Employer: {employer_name}
                Previous Exposure: {previous_exposure:.1f}%
                Current Exposure: {current_exposure:.1f}%
                Resolution Date: {resolution_date}
                
                Breach successfully resolved.
                """
            
            st.text_area("Template Preview", template, height=150)
            
            if st.button("💾 Save Template"):
                self._save_alert_template(template_type, template)
                st.success("Alert template saved!")
    
    def render_breach_analytics(self):
        """Render breach analytics and trends"""
        st.markdown("---")
        st.subheader("📈 Breach Analytics & Trends")
        
        # Generate sample breach history
        breach_history = self._generate_breach_history()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Breach frequency chart
            breach_freq = breach_history['monthly_breaches']
            fig = px.bar(
                breach_freq,
                x='Month',
                y='Breach_Count',
                title="Monthly Limit Breaches",
                color='Breach_Count',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Breach duration analysis
            duration_data = breach_history['breach_duration']
            fig = px.box(
                duration_data,
                y='Duration_Days',
                title="Breach Duration Distribution",
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top breaching employers
        st.markdown("#### 🏆 Top Breaching Employers (Last 6 Months)")
        
        top_breachers = breach_history['top_breachers']
        if not top_breachers.empty:
            fig = px.bar(
                top_breachers,
                x='Employer',
                y='Breach_Count',
                title="Frequent Limit Breachers",
                color='Total_Breach_Days',
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Resolution time analysis
        st.markdown("#### ⏱️ Breach Resolution Performance")
        
        resolution_data = breach_history['resolution_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_resolution = resolution_data.get('average_resolution_days', 0)
            st.metric(
                "Average Resolution Time",
                f"{avg_resolution:.1f} days",
                help="Average time to resolve breaches"
            )
        
        with col2:
            sla_compliance = resolution_data.get('sla_compliance_rate', 0) * 100
            st.metric(
                "SLA Compliance Rate",
                f"{sla_compliance:.1f}%",
                help="Percentage of breaches resolved within SLA"
            )
        
        with col3:
            recurring_rate = resolution_data.get('recurring_breach_rate', 0) * 100
            st.metric(
                "Recurring Breach Rate",
                f"{recurring_rate:.1f}%",
                delta_color="inverse",
                help="Percentage of employers with repeated breaches"
            )
    
    def render_action_plans(self):
        """Render breach action plans and tracking"""
        st.markdown("---")
        st.subheader("📋 Breach Action Plans")
        
        # Sample action plans
        action_plans = self._get_sample_action_plans()
        
        for plan in action_plans:
            with st.expander(f"📝 {plan['employer']} - {plan['status']}", expanded=plan['status'] == 'Active'):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Exposure", f"{plan['current_exposure']:.1f}%")
                    st.metric("Target Exposure", f"{plan['target_exposure']:.1f}%")
                
                with col2:
                    st.metric("Days Since Breach", plan['days_since_breach'])
                    st.metric("SLA Deadline", plan['sla_deadline'])
                
                with col3:
                    progress = (plan['current_exposure'] - plan['target_exposure']) / (plan['initial_exposure'] - plan['target_exposure']) * 100
                    st.metric("Resolution Progress", f"{max(0, progress):.1f}%")
                
                # Action items
                st.markdown("#### ✅ Action Items")
                for i, action in enumerate(plan['action_items']):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{'✅' if action['completed'] else '⏳'} {action['description']}")
                    with col2:
                        st.write(f"Due: {action['due_date']}")
                    with col3:
                        if not action['completed']:
                            if st.button("Mark Complete", key=f"complete_{plan['employer']}_{i}"):
                                action['completed'] = True
                                st.rerun()
                
                # Resolution actions
                if plan['status'] == 'Active':
                    st.markdown("#### 🎯 Resolution Actions")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🚀 Accelerate Resolution", key=f"accelerate_{plan['employer']}"):
                            self._accelerate_resolution(plan)
                            st.success("Resolution acceleration initiated!")
                    
                    with col2:
                        if st.button("📋 Request Extension", key=f"extension_{plan['employer']}"):
                            self._request_extension(plan)
                            st.success("Extension request submitted!")
    
    def _send_notification(self, employer_data, recipient):
        """Send notification for limit breach"""
        # In production, this would integrate with email/SMS systems
        message = f"Limit breach alert for {employer_data['employer_name']}"
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "send_notification",
            "employer_limit",
            employer_data['employer_name'],
            {'recipient': recipient, 'message': message}
        )
    
    def _create_action_plan(self, employer_data):
        """Create action plan for limit breach"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "create_action_plan",
            "employer_limit",
            employer_data['employer_name'],
            {'exposure_share': employer_data['exposure_share']}
        )
    
    def _request_waiver(self, employer_data):
        """Request regulatory waiver for limit breach"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "request_waiver",
            "employer_limit",
            employer_data['employer_name'],
            {'exposure_share': employer_data['exposure_share']}
        )
    
    def _initiate_preventive_measures(self, employer_data):
        """Initiate preventive measures for approaching limits"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "initiate_preventive_measures",
            "employer_limit",
            employer_data['employer_name'],
            {'exposure_share': employer_data['exposure_share']}
        )
    
    def _save_custom_limit(self, employer_name, custom_limit):
        """Save custom limit for employer"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "save_custom_limit",
            "employer_limit",
            employer_name,
            {'custom_limit': custom_limit}
        )
    
    def _apply_standard_limits(self):
        """Apply standard limits to all employers"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "apply_standard_limits",
            "employer_limit",
            None,
            {}
        )
    
    def _clear_custom_limits(self):
        """Clear all custom limits"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "clear_custom_limits",
            "employer_limit",
            None,
            {}
        )
    
    def _generate_limits_report(self, monitor_df):
        """Generate employer limits report"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "generate_limits_report",
            "employer_limit",
            None,
            {'employer_count': len(monitor_df)}
        )
        return {"status": "success", "report_id": f"EMP_LIMIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    
    def _save_alert_template(self, template_type, template):
        """Save alert template"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "save_alert_template",
            "employer_limit",
            template_type,
            {'template_length': len(template)}
        )
    
    def _generate_breach_history(self):
        """Generate sample breach history data with consistent array lengths"""
        try:
            # Generate consistent date range (7 months)
            months = pd.date_range(start='2023-08-01', end='2024-02-01', freq='M')
        
            # Ensure all arrays have exactly 7 elements
            monthly_breaches = pd.DataFrame({
                'Month': [m.strftime('%b %Y') for m in months],
                'Breach_Count': [3, 2, 4, 1, 2, 3, 2]
            })
        
            # Breach duration data (can have different length since it's a different dataset)
            breach_duration = pd.DataFrame({
                'Duration_Days': [7, 14, 3, 21, 5, 10, 8, 15, 6, 12]
            })
        
            # Top breachers data (5 employers)
            top_breachers = pd.DataFrame({
                'Employer': ['Employer_A', 'Employer_B', 'Employer_C', 'Employer_D', 'Employer_E'],
                'Breach_Count': [5, 4, 3, 2, 2],
                'Total_Breach_Days': [45, 32, 28, 14, 12]
            })
        
            resolution_metrics = {
                'average_resolution_days': 8.5,
                'sla_compliance_rate': 0.75,
                'recurring_breach_rate': 0.30
            }
        
            return {
                'monthly_breaches': monthly_breaches,
                'breach_duration': breach_duration,
                'top_breachers': top_breachers,
                'resolution_metrics': resolution_metrics
            }
        
        except Exception as e:
            # Fallback with guaranteed consistent data
            st.error(f"Error generating breach history: {str(e)}")
        
            months_list = ['Aug 2023', 'Sep 2023', 'Oct 2023', 'Nov 2023', 'Dec 2023', 'Jan 2024', 'Feb 2024']
        
            return {
                'monthly_breaches': pd.DataFrame({
                    'Month': months_list,
                    'Breach_Count': [3, 2, 4, 1, 2, 3, 2]
                }),
                'breach_duration': pd.DataFrame({
                    'Duration_Days': [7, 14, 3, 21, 5, 10, 8, 15, 6, 12]
                }),
                'top_breachers': pd.DataFrame({
                    'Employer': ['Employer_A', 'Employer_B', 'Employer_C', 'Employer_D', 'Employer_E'],
                    'Breach_Count': [5, 4, 3, 2, 2],
                    'Total_Breach_Days': [45, 32, 28, 14, 12]
                }),
                'resolution_metrics': {
                    'average_resolution_days': 8.5,
                    'sla_compliance_rate': 0.75,
                    'recurring_breach_rate': 0.30
                }
            }
    
    def _get_sample_action_plans(self):
        """Get sample action plans"""
        return [
            {
                'employer': 'Employer_A',
                'status': 'Active',
                'current_exposure': 26.5,
                'target_exposure': 24.0,
                'initial_exposure': 28.2,
                'days_since_breach': 7,
                'sla_deadline': '2024-03-15',
                'action_items': [
                    {'description': 'Suspend new loan disbursements', 'due_date': '2024-02-20', 'completed': True},
                    {'description': 'Develop exposure reduction plan', 'due_date': '2024-02-25', 'completed': True},
                    {'description': 'Implement accelerated collections', 'due_date': '2024-03-10', 'completed': False},
                    {'description': 'Review and approve waiver request', 'due_date': '2024-03-15', 'completed': False}
                ]
            },
            {
                'employer': 'Employer_B',
                'status': 'Completed',
                'current_exposure': 23.8,
                'target_exposure': 24.0,
                'initial_exposure': 26.1,
                'days_since_breach': 0,
                'sla_deadline': '2024-02-10',
                'action_items': [
                    {'description': 'Temporary lending suspension', 'due_date': '2024-01-20', 'completed': True},
                    {'description': 'Member education on limits', 'due_date': '2024-01-25', 'completed': True},
                    {'description': 'Portfolio rebalancing', 'due_date': '2024-02-05', 'completed': True}
                ]
            }
        ]
    
    def _accelerate_resolution(self, action_plan):
        """Accelerate breach resolution"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "accelerate_resolution",
            "employer_limit",
            action_plan['employer'],
            {}
        )
    
    def _request_extension(self, action_plan):
        """Request SLA extension"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "request_extension",
            "employer_limit",
            action_plan['employer'],
            {}
        )
    
    def run(self):
        """Run the employer limits and alerts page"""
        st.title("🚨 Employer Limits & Alerts")
        
        st.markdown("""
        Real-time monitoring of employer exposure limits, automated alerting for breaches, 
        and comprehensive management of regulatory compliance for single employer concentration limits.
        """)
        
        self.render_limits_dashboard()
        self.render_employer_limits_monitor()
        self.render_limit_configuration()
        self.render_alert_management()
        self.render_breach_analytics()
        self.render_action_plans()

if __name__ == "__main__":
    page = EmployerLimitsAlertsPage()
    page.run()