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
    
    def authenticate_user(self, email: str, password: str) -> bool:
        """Authenticate user with email and password"""
        # For demo purposes - in production, use proper hashing and database
        demo_users = {
            'risk@example.com': {'password': 'Risk@123', 'role': 'Risk'},
            'board@example.com': {'password': 'Board@123', 'role': 'Board'},
            'admin@example.com': {'password': 'Admin@123', 'role': 'Admin'},
            'collections@example.com': {'password': 'Collections@123', 'role': 'Collections'},
            'finance@example.com': {'password': 'Finance@123', 'role': 'Finance'},
            'auditor@example.com': {'password': 'Auditor@123', 'role': 'Auditor'}
        }
        
        if email in demo_users and password == demo_users[email]['password']:
            st.session_state.authenticated = True
            st.session_state.user = email
            st.session_state.role = demo_users[email]['role']
            self.audit_logger.log_login(email, demo_users[email]['role'])
            return True
        return False
    
    def render_login(self):
        """Render login page"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>🏦 SACCO Pro Management System</h1>
            <p>Comprehensive SACCO Risk Management & Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.subheader("Login")
                email = st.text_input("Email", placeholder="risk@example.com")
                password = st.text_input("Password", type="password", placeholder="Risk@123")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if self.authenticate_user(email, password):
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            
            st.markdown("---")
            st.subheader("Demo Credentials")
            demo_info = """
            **Risk Analyst**: risk@example.com / Risk@123  
            **Board Member**: board@example.com / Board@123  
            **Administrator**: admin@example.com / Admin@123  
            **Collections**: collections@example.com / Collections@123  
            **Finance**: finance@example.com / Finance@123  
            **Auditor**: auditor@example.com / Auditor@123  
            """
            st.info(demo_info)
    
    def render_sidebar(self):
        """Render sidebar with navigation based on user role"""
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.user}")
            st.caption(f"Role: {st.session_state.role}")
            
            st.markdown("---")
            st.subheader("Navigation")
            
            # Get accessible pages for user role
            accessible_pages = self.rbac_manager.get_accessible_pages(
                st.session_state.role, 
                st.session_state.config
            )
            
            for page_info in accessible_pages:
                if st.button(page_info['name'], key=page_info['module']):
                    st.switch_page(f"pages/{page_info['module']}")
            
            st.markdown("---")
            if st.button("Logout"):
                self.audit_logger.log_logout(st.session_state.user, st.session_state.role)
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.role = None
                st.rerun()
    
    def render_main(self):
        """Render main dashboard"""
        st.title("SACCO Management Dashboard")
        st.markdown("Welcome to the comprehensive SACCO management system.")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Members", "1,250", "5%")
        with col2:
            st.metric("Loan Portfolio", "KES 245M", "3.2%")
        with col3:
            st.metric("PAR > 30 Days", "4.2%", "-0.5%")
        with col4:
            st.metric("Liquidity Ratio", "18.5%", "1.2%")
        
        # Recent alerts
        st.subheader("Recent Alerts")
        alert_data = {
            "Date": ["2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12"],
            "Alert": ["PAR30 approaching threshold", "Employer concentration high", "Liquidity ratio improved", "Employer limit breach"],
            "Severity": ["Medium", "High", "Low", "Medium"]
        }
        st.dataframe(alert_data, use_container_width=True)
    
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