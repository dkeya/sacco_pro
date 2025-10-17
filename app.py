# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import json
import hashlib
import time
from pathlib import Path
import sqlite3
import duckdb
import os
from typing import Dict, List, Optional, Any
import tempfile

# Import core modules
from sacco_core.config import ConfigManager
from sacco_core.rbac import RBACManager
from sacco_core.audit import AuditLogger
from sacco_core.db import DatabaseManager

# Page configuration - HIDE THE SIDEBAR NAVIGATION
st.set_page_config(
    page_title="SACCO Pro Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS to hide Streamlit's default page navigation
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Hide Streamlit's default page navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide sidebar on login page */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Optional: Additional customizations */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Remove padding from main container */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Custom header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            color: white;
        }
        
        /* Show sidebar only when authenticated */
        .sidebar-visible [data-testid="stSidebar"] {
            display: block;
        }
        
        /* Collapsible sidebar styling */
        .sidebar-category {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #e9ecef;
        }
        
        /* Indented page buttons */
        .indented-button {
            margin-left: 1rem;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class SaccoApp:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.audit_logger = AuditLogger()
        self.db_manager = DatabaseManager()
        self.rbac_manager = RBACManager()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.config = None
            st.session_state.current_page = "dashboard"
        
        # Initialize sidebar collapsed state
        if 'sidebar_collapsed' not in st.session_state:
            st.session_state.sidebar_collapsed = {
                "📊 Analytics": True,
                "🎯 Risk Management": True,
                "⏱️ Operations": True,
                "🔒 Compliance & Security": True
            }
        
        # Load configuration
        self.load_configuration()
    
    def load_configuration(self):
        """Load application configuration"""
        try:
            st.session_state.config = self.config_manager.load_settings()
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.session_state.config = self.config_manager.get_default_config()
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with username and password"""
        # Simplified authentication for demo - in production, use proper hashing and database
        valid_users = {
            'admin': {'password': 'admin123', 'role': 'Admin', 'name': 'Administrator'},
            'risk': {'password': 'risk123', 'role': 'Risk', 'name': 'Risk Analyst'},
            'finance': {'password': 'finance123', 'role': 'Finance', 'name': 'Finance Manager'},
            'collections': {'password': 'collections123', 'role': 'Collections', 'name': 'Collections Officer'},
            'auditor': {'password': 'auditor123', 'role': 'Auditor', 'name': 'Internal Auditor'},
            'board': {'password': 'board123', 'role': 'Board', 'name': 'Board Member'}
        }
        
        if username in valid_users and password == valid_users[username]['password']:
            st.session_state.authenticated = True
            st.session_state.user = valid_users[username]['name']
            st.session_state.role = valid_users[username]['role']
            st.session_state.username = username
            self.audit_logger.log_login(username, valid_users[username]['role'])
            return True
        return False
    
    def render_login(self):
        """Render login page with hidden sidebar"""
        # Apply CSS to hide sidebar specifically for login page
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none !important;
            }
            .main .block-container {
                max-width: 100% !important;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>🏦 SACCO Pro Management System</h1>
            <p>Comprehensive SACCO Risk Management & Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.container():
                st.subheader("🔐 System Login")
                
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input(
                        "Username", 
                        placeholder="Enter your username",
                        help="Enter your system username"
                    )
                    password = st.text_input(
                        "Password", 
                        type="password", 
                        placeholder="Enter your password",
                        help="Enter your system password"
                    )
                    submit = st.form_submit_button(
                        "🚀 Login to System",
                        use_container_width=True
                    )
                    
                    if submit:
                        if username and password:
                            if self.authenticate_user(username, password):
                                st.success("✅ Login successful! Redirecting...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid username or password")
                        else:
                            st.warning("⚠️ Please enter both username and password")
                
                # Security notice
                st.markdown("---")
                st.info("""
                **Security Notice**: 
                - This system contains confidential financial information
                - Unauthorized access is prohibited
                - All activities are logged and monitored
                """)
    
    def get_page_categories(self):
        """Define the page categories and their modules"""
        return {
            "📊 Analytics": [
                {"name": "📊 Overview Dashboard", "module": "01_Overview.py"},
                {"name": "📈 Liquidity ALM", "module": "03_Liquidity_ALM.py"},
                {"name": "💰 Pricing Economics", "module": "04_Pricing_Economics.py"},
                {"name": "💧 Dividend Capacity", "module": "03A_Dividend_Capacity.py"}
            ],
            "🎯 Risk Management": [
                {"name": "🎯 Credit Risk PAR", "module": "02_Credit_Risk_PAR.py"},
                {"name": "📉 Vintages & Roll Rates", "module": "02A_Vintages_RollRates.py"},
                {"name": "⚖️ Concentration Risk", "module": "06_Concentration_Risk.py"},
                {"name": "🚨 Employer Limits & Alerts", "module": "06A_Employer_Limits_Alerts.py"},
                {"name": "📊 Provisioning & Writeoff", "module": "09_Provisioning_Writeoff.py"},
                {"name": "🌀 ALM Stress Tests", "module": "03B_ALM_Stress_Tests.py"}
            ],
            "⏱️ Operations": [
                {"name": "⏱️ Operations TAT", "module": "08_Operations_TAT.py"},
                {"name": "📞 Collections Recovery", "module": "13_Collections_Recovery.py"},
                {"name": "📱 Collections Funnel SMS", "module": "13A_Collections_Funnel_SMS.py"},
                {"name": "👥 Member Value", "module": "11_Member_Value.py"},
                {"name": "📈 Member Value & Churn", "module": "11A_Member_Value_Churn.py"}
            ],
            "🔒 Compliance & Security": [
                {"name": "🔒 Cybersecurity & BCP", "module": "10_Cybersecurity_BCP.py"},
                {"name": "⚖️ Governance Compliance", "module": "05_Governance_Compliance.py"},
                {"name": "📋 SASRA Returns", "module": "05A_SASRA_Returns.py"},
                {"name": "📊 Data Quality MIS", "module": "07_Data_Quality_MIS.py"},
                {"name": "🔍 Data Quality Scans", "module": "07A_Data_Quality_Scans.py"},
                {"name": "⚙️ Policy Engine Monitor", "module": "12_Policy_Engine_Monitor.py"},
                {"name": "📄 AGM Dividend Paper", "module": "14_AGM_Dividend_Paper.py"}
            ]
        }
    
    def render_sidebar(self):
        """Render sidebar with collapsible navigation based on user role"""
        # Show sidebar only when authenticated
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: block !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            # User info section
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 1rem;">
                <h3>🏦 SACCO Pro</h3>
                <p><strong>{st.session_state.user}</strong></p>
                <p><em>{st.session_state.role} Role</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📊 Navigation")
            
            # Get page categories
            page_categories = self.get_page_categories()
            
            # Display Home/Dashboard button
            if st.button("🏠 Dashboard Home", use_container_width=True, key="nav_home"):
                st.session_state.current_page = "dashboard"
                st.rerun()
            
            st.markdown("---")
            
            # Display collapsible categories
            for category_name, pages in page_categories.items():
                # Create a unique key for each category toggle
                toggle_key = f"toggle_{category_name.replace(' ', '_').replace('&', 'and')}"
                
                # Toggle button with arrow indicator
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"{category_name}", 
                        key=toggle_key,
                        use_container_width=True,
                        help=f"Click to {'expand' if st.session_state.sidebar_collapsed[category_name] else 'collapse'} {category_name}"
                    ):
                        # Toggle the collapsed state
                        st.session_state.sidebar_collapsed[category_name] = not st.session_state.sidebar_collapsed[category_name]
                        st.rerun()
                
                with col2:
                    # Show arrow indicator
                    arrow = "⬇️" if not st.session_state.sidebar_collapsed[category_name] else "➡️"
                    st.write(arrow)
                
                # Show pages if category is expanded
                if not st.session_state.sidebar_collapsed[category_name]:
                    for page in pages:
                        if st.button(
                            f"  {page['name']}",  # Indent pages for hierarchy
                            key=f"nav_{page['module']}", 
                            use_container_width=True,
                            help=f"Go to {page['name']}"
                        ):
                            st.switch_page(f"pages/{page['module']}")
            
            st.markdown("---")
            
            # Expand/Collapse All buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Expand All", use_container_width=True, key="expand_all"):
                    for category in st.session_state.sidebar_collapsed:
                        st.session_state.sidebar_collapsed[category] = False
                    st.rerun()
            
            with col2:
                if st.button("📁 Collapse All", use_container_width=True, key="collapse_all"):
                    for category in st.session_state.sidebar_collapsed:
                        st.session_state.sidebar_collapsed[category] = True
                    st.rerun()
            
            st.markdown("---")
            
            # System info and logout
            st.caption(f"Logged in as: {st.session_state.username}")
            st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
                self.audit_logger.log_logout(st.session_state.username, st.session_state.role)
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.role = None
                st.session_state.username = None
                st.session_state.current_page = "dashboard"
                st.rerun()
    
    def render_main(self):
        """Render main dashboard"""
        # Custom header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h1>Welcome back, {st.session_state.user}! 👋</h1>
            <p>SACCO Pro Management System - Real-time Analytics & Risk Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats in cards
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Members", 
                value="15,247",
                delta="+234",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Total Assets", 
                value="KES 2.45B",
                delta="+KES 120M",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="PAR 30 Days", 
                value="3.2%",
                delta="-0.4%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="Liquidity Ratio", 
                value="125%",
                delta="+5%",
                delta_color="normal"
            )
        
        # Recent activity and alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔔 Recent Alerts")
            alerts_data = {
                "Time": ["2 hours ago", "4 hours ago", "1 day ago", "2 days ago"],
                "Alert": [
                    "PAR 30 approaching threshold (4.8%)",
                    "High employer concentration detected",
                    "Cybersecurity scan completed",
                    "Monthly compliance reports generated"
                ],
                "Severity": ["Medium", "High", "Low", "Info"]
            }
            alerts_df = pd.DataFrame(alerts_data)
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚡ Quick Actions")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("📊 View Dashboard", use_container_width=True, key="quick_dash"):
                    st.switch_page("pages/01_Overview.py")
                
                if st.button("💧 Liquidity Analysis", use_container_width=True, key="quick_liquidity"):
                    st.switch_page("pages/03_Liquidity_ALM.py")
                
                if st.button("🎯 Risk Monitoring", use_container_width=True, key="quick_risk"):
                    st.switch_page("pages/06_Concentration_Risk.py")
            
            with action_col2:
                if st.button("📋 Generate Reports", use_container_width=True, key="quick_reports"):
                    st.switch_page("pages/05A_SASRA_Returns.py")
                
                if st.button("🔒 Security Status", use_container_width=True, key="quick_security"):
                    st.switch_page("pages/10_Cybersecurity_BCP.py")
                
                if st.button("⚙️ Policy Engine", use_container_width=True, key="quick_policy"):
                    st.switch_page("pages/12_Policy_Engine_Monitor.py")
            
            # System status
            st.markdown("---")
            st.subheader("🟢 System Status")
            status_data = {
                "Component": ["Database", "Analytics Engine", "Security", "Backup"],
                "Status": ["Operational", "Operational", "Active", "Completed"],
                "Last Check": ["5 min ago", "2 min ago", "1 min ago", "2 hours ago"]
            }
            status_df = pd.DataFrame(status_data)
            st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    def run(self):
        """Main application runner"""
        if not st.session_state.authenticated:
            self.render_login()
        else:
            self.render_sidebar()
            self.render_main()

# Initialize and run the app
if __name__ == "__main__":
    app = SaccoApp()
    app.run()