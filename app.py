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

# Page configuration
st.set_page_config(
    page_title="SACCO Pro Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced version with more customization
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
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
        
        /* Style sidebar better */
        .css-1d391kg {
            padding-top: 2rem;
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
            st.session_state.config = None
        
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
        """Render login page"""
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
    
    def render_sidebar(self):
        """Render sidebar with navigation based on user role"""
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
            
            # Get accessible pages for user role
            accessible_pages = self.rbac_manager.get_accessible_pages(
                st.session_state.role, 
                st.session_state.config
            )
            
            # Group pages by category for better organization
            analytics_pages = [p for p in accessible_pages if any(x in p['name'] for x in ['📊', '📈', '💧', '💰'])]
            risk_pages = [p for p in accessible_pages if any(x in p['name'] for x in ['🎯', '🚨', '📉'])]
            operations_pages = [p for p in accessible_pages if any(x in p['name'] for x in ['⏱️', '📞', '📱'])]
            compliance_pages = [p for p in accessible_pages if any(x in p['name'] for x in ['⚖️', '📋', '🔒', '⚙️'])]
            other_pages = [p for p in accessible_pages if p not in analytics_pages + risk_pages + operations_pages + compliance_pages]
            
            # Display categorized pages
            if analytics_pages:
                st.markdown("**Analytics**")
                for page_info in analytics_pages:
                    if st.button(page_info['name'], key=f"nav_{page_info['module']}", use_container_width=True):
                        st.switch_page(f"pages/{page_info['module']}")
            
            if risk_pages:
                st.markdown("**Risk Management**")
                for page_info in risk_pages:
                    if st.button(page_info['name'], key=f"nav_{page_info['module']}", use_container_width=True):
                        st.switch_page(f"pages/{page_info['module']}")
            
            if operations_pages:
                st.markdown("**Operations**")
                for page_info in operations_pages:
                    if st.button(page_info['name'], key=f"nav_{page_info['module']}", use_container_width=True):
                        st.switch_page(f"pages/{page_info['module']}")
            
            if compliance_pages:
                st.markdown("**Compliance & Security**")
                for page_info in compliance_pages:
                    if st.button(page_info['name'], key=f"nav_{page_info['module']}", use_container_width=True):
                        st.switch_page(f"pages/{page_info['module']}")
            
            if other_pages:
                for page_info in other_pages:
                    if st.button(page_info['name'], key=f"nav_{page_info['module']}", use_container_width=True):
                        st.switch_page(f"pages/{page_info['module']}")
            
            st.markdown("---")
            
            # System info and logout
            st.caption(f"Logged in as: {st.session_state.username}")
            st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            if st.button("🚪 Logout", use_container_width=True):
                self.audit_logger.log_logout(st.session_state.username, st.session_state.role)
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.role = None
                st.session_state.username = None
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
                if st.button("📊 View Dashboard", use_container_width=True):
                    st.switch_page("pages/01_Overview.py")
                
                if st.button("💧 Liquidity Analysis", use_container_width=True):
                    st.switch_page("pages/03_Liquidity_ALM.py")
                
                if st.button("🎯 Risk Monitoring", use_container_width=True):
                    st.switch_page("pages/06_Concentration_Risk.py")
            
            with action_col2:
                if st.button("📋 Generate Reports", use_container_width=True):
                    st.switch_page("pages/05A_SASRA_Returns.py")
                
                if st.button("🔒 Security Status", use_container_width=True):
                    st.switch_page("pages/10_Cybersecurity_BCP.py")
                
                if st.button("⚙️ Policy Engine", use_container_width=True):
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