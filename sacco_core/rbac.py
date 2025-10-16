# sacco_core/rbac.py
import fnmatch
from typing import List, Dict, Any
import streamlit as st

# RBAC Role Definitions
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

class RBACManager:
    def __init__(self):
        pass
    
    def get_accessible_pages(self, role: str, config: Any) -> List[Dict[str, str]]:
        """Get list of pages accessible for a given role"""
        all_pages = [
            {"module": "01_Overview.py", "name": "📊 Overview"},
            {"module": "02_Credit_Risk_PAR.py", "name": "📈 Credit Risk & PAR"},
            {"module": "02A_Vintages_RollRates.py", "name": "📊 Vintages & Roll Rates"},
            {"module": "03_Liquidity_ALM.py", "name": "💧 Liquidity & ALM"},
            {"module": "03A_Dividend_Capacity.py", "name": "💰 Dividend Capacity"},
            {"module": "03B_ALM_Stress_Tests.py", "name": "🌊 ALM Stress Tests"},
            {"module": "04_Pricing_Economics.py", "name": "💳 Pricing Economics"},
            {"module": "05_Governance_Compliance.py", "name": "⚖️ Governance & Compliance"},
            {"module": "05A_SASRA_Returns.py", "name": "📋 SASRA Returns"},
            {"module": "06_Concentration_Risk.py", "name": "🎯 Concentration Risk"},
            {"module": "06A_Employer_Limits_Alerts.py", "name": "🚨 Employer Limits & Alerts"},
            {"module": "07_Data_Quality_MIS.py", "name": "📋 Data Quality MIS"},
            {"module": "07A_Data_Quality_Scans.py", "name": "🔍 Data Quality Scans"},
            {"module": "08_Operations_TAT.py", "name": "⏱️ Operations TAT"},
            {"module": "09_Provisioning_Writeoff.py", "name": "📉 Provisioning & Write-off"},
            {"module": "10_Cybersecurity_BCP.py", "name": "🔒 Cybersecurity & BCP"},
            {"module": "11_Member_Value.py", "name": "👥 Member Value"},
            {"module": "11A_Member_Value_Churn.py", "name": "📊 Member Value & Churn"},
            {"module": "12_Policy_Engine_Monitor.py", "name": "⚙️ Policy Engine Monitor"},
            {"module": "13_Collections_Recovery.py", "name": "📞 Collections & Recovery"},
            {"module": "13A_Collections_Funnel_SMS.py", "name": "📱 Collections Funnel & SMS"},
            {"module": "14_AGM_Dividend_Paper.py", "name": "📄 AGM Dividend Paper"}
        ]
        
        if role not in config.rbac.page_patterns:
            return []
        
        patterns = config.rbac.page_patterns[role]
        accessible_pages = []
        
        for page in all_pages:
            page_name = page["module"].replace('.py', '')
            for pattern in patterns:
                if fnmatch.fnmatch(page_name, pattern):
                    accessible_pages.append(page)
                    break
        
        return accessible_pages
    
    def check_page_access(self, page_module: str, role: str, config: Any) -> bool:
        """Check if role has access to specific page"""
        if role not in config.rbac.page_patterns:
            return False
        
        page_name = page_module.replace('.py', '')
        patterns = config.rbac.page_patterns[role]
        
        for pattern in patterns:
            if fnmatch.fnmatch(page_name, pattern):
                return True
        
        return False

# Create a global instance for easy access
_rbac_manager = RBACManager()

# Standalone function for easy importing
def check_permission(page_module: str) -> bool:
    """
    Check if current user has permission to access a page
    
    Args:
        page_module: The page module name (e.g., "01_Overview.py")
        
    Returns:
        bool: True if user has access, False otherwise
    """
    try:
        # Get current user role from session state
        role = st.session_state.get('role', 'unknown')
        
        # Admin has access to everything
        if role == 'admin':
            return True
            
        # Import config here to avoid circular imports
        try:
            from sacco_core.config import get_config
            config = get_config()
        except ImportError:
            # Fallback to default config if import fails
            from sacco_core.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_default_config()
        
        # Check page access using RBAC manager
        return _rbac_manager.check_page_access(page_module, role, config)
        
    except Exception as e:
        # If there's any error, default to no access for security
        st.error(f"Access check error: {e}")
        return False

def get_accessible_pages() -> List[Dict[str, str]]:
    """
    Get list of pages accessible for current user
    
    Returns:
        List of page dictionaries with module and name
    """
    try:
        role = st.session_state.get('role', 'unknown')
        
        # Import config here to avoid circular imports
        try:
            from sacco_core.config import get_config
            config = get_config()
        except ImportError:
            # Fallback to default config if import fails
            from sacco_core.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_default_config()
        
        return _rbac_manager.get_accessible_pages(role, config)
        
    except Exception as e:
        st.error(f"Error getting accessible pages: {e}")
        return []

def has_role(required_role: str) -> bool:
    """
    Check if current user has a specific role
    
    Args:
        required_role: The role to check for
        
    Returns:
        bool: True if user has the role, False otherwise
    """
    try:
        current_role = st.session_state.get('role', 'unknown')
        return current_role == required_role
    except Exception:
        return False

def get_current_role() -> str:
    """
    Get current user's role
    
    Returns:
        str: Current user role
    """
    return st.session_state.get('role', 'unknown')

def get_current_username() -> str:
    """
    Get current username
    
    Returns:
        str: Current username
    """
    return st.session_state.get('username', 'unknown')