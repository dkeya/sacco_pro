# sacco_core/sidebar.py
import streamlit as st
from datetime import datetime
from .audit import AuditLogger
from .ui import hide_default_streamlit_elements, apply_custom_styling

def render_sidebar():
    """Render consistent sidebar across all pages with collapsible categories"""
    
    # Apply consistent styling
    hide_default_streamlit_elements()
    apply_custom_styling()
    
    # Initialize session state for collapsed categories
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = {
            "📊 Analytics": True,
            "🎯 Risk Management": True,
            "⏱️ Operations": True,
            "🔒 Compliance & Security": True
        }
    
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
        
        # Display Home/Dashboard button
        if st.button("🏠 Dashboard Home", use_container_width=True, key="nav_home"):
            st.switch_page("app.py")
        
        st.markdown("---")
        
        # Get page categories
        page_categories = get_page_categories()
        
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
            audit_logger = AuditLogger()
            audit_logger.log_logout(st.session_state.username, st.session_state.role)
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()

def get_page_categories():
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