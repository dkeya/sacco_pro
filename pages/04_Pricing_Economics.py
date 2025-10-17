# pages/04_Pricing_Economics.py
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
from sacco_core.analytics.pricing import PricingOptimizer
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Pricing Economics",
    page_icon="💳",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class PricingEconomicsPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.pricing_optimizer = PricingOptimizer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "04_Pricing_Economics.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "pricing_economics_page"
        )
        return True
    
    def render_pricing_dashboard(self):
        """Render the main pricing dashboard"""
        st.subheader("📊 Current Pricing Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Loan Yield",
                "14.2%",
                "0.3%",
                help="Weighted average yield across all loan products"
            )
        
        with col2:
            st.metric(
                "Average Deposit Cost", 
                "5.8%",
                "-0.1%",
                help="Weighted average cost of deposits"
            )
        
        with col3:
            st.metric(
                "Net Interest Margin",
                "8.4%",
                "0.2%",
                help="Net interest income as percentage of earning assets"
            )
        
        with col4:
            st.metric(
                "Pricing Efficiency",
                "86%",
                "2%", 
                help="Ratio of actual vs optimal pricing"
            )
        
        # Current product pricing
        self.render_current_pricing()
    
    def render_current_pricing(self):
        """Render current product pricing analysis"""
        st.markdown("#### 📈 Current Product Pricing Analysis")
        
        # Sample product pricing data
        products_data = pd.DataFrame({
            'Product': ['Personal Loans', 'Business Loans', 'Asset Finance', 'Emergency Loans', 'School Fees'],
            'Current_Rate': [13.5, 15.2, 12.8, 18.5, 10.5],
            'Risk_Adjusted_Rate': [14.1, 16.8, 13.2, 20.2, 10.8],
            'Market_Average': [14.5, 16.0, 13.5, 19.0, 11.0],
            'Volume_KES_M': [85.2, 92.5, 45.3, 12.8, 9.9],
            'Profitability': [3.2, 4.1, 2.8, 5.2, 1.8]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pricing comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current Rate',
                x=products_data['Product'],
                y=products_data['Current_Rate'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Risk Adjusted',
                x=products_data['Product'], 
                y=products_data['Risk_Adjusted_Rate'],
                marker_color='orange'
            ))
            
            fig.add_trace(go.Bar(
                name='Market Average',
                x=products_data['Product'],
                y=products_data['Market_Average'],
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Loan Product Pricing Comparison",
                xaxis_title="Product",
                yaxis_title="Interest Rate (%)",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profitability vs Volume scatter
            fig = px.scatter(
                products_data,
                x='Volume_KES_M',
                y='Profitability',
                size='Volume_KES_M',
                color='Product',
                hover_name='Product',
                title="Product Profitability vs Volume",
                labels={'Volume_KES_M': 'Portfolio Volume (KES M)', 'Profitability': 'Profit Margin (%)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_pricing_optimizer(self):
        """Render the pricing optimization tool"""
        st.markdown("---")
        st.subheader("🎯 Loan Pricing Optimizer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Loan Parameters")
            
            product_type = st.selectbox(
                "Product Type",
                ["Personal Loan", "Business Loan", "Asset Finance", "Emergency Loan", "School Fees"],
                help="Select the loan product to price"
            )
            
            loan_amount = st.number_input(
                "Loan Amount (KES)",
                min_value=10000.0,
                value=500000.0,
                step=10000.0,
                help="Requested loan amount"
            )
            
            loan_term = st.slider(
                "Loan Term (Months)",
                min_value=1,
                max_value=84,
                value=24,
                help="Loan repayment period in months"
            )
            
            risk_category = st.select_slider(
                "Risk Category",
                options=["Low", "Medium", "High", "Very High"],
                value="Medium",
                help="Borrower risk assessment category"
            )
        
        with col2:
            st.markdown("#### ⚙️ Cost Parameters")
            
            cost_of_funds = st.slider(
                "Cost of Funds (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.8,
                step=0.1,
                help="Weighted average cost of funds"
            )
            
            operating_cost = st.slider(
                "Operating Cost (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.2,
                step=0.1,
                help="Operating cost as percentage of loan amount"
            )
            
            target_roa = st.slider(
                "Target Return on Assets (%)",
                min_value=0.5,
                max_value=5.0,
                value=1.8,
                step=0.1,
                help="Desired return on assets"
            )
            
            risk_premium = st.slider(
                "Risk Premium (%)",
                min_value=0.5,
                max_value=10.0,
                value=self._get_risk_premium(risk_category),
                step=0.1,
                help="Additional premium for risk category"
            )
        
        # Calculate optimal pricing
        if st.button("🚀 Calculate Optimal Pricing", type="primary"):
            pricing_result = self.pricing_optimizer.calculate_optimal_pricing(
                product_type=product_type,
                loan_amount=loan_amount,
                loan_term=loan_term,
                risk_category=risk_category,
                cost_of_funds=cost_of_funds / 100,
                operating_cost=operating_cost / 100,
                target_roa=target_roa / 100,
                risk_premium=risk_premium / 100
            )
            
            st.session_state.pricing_result = pricing_result
        
        # Display pricing results
        if 'pricing_result' in st.session_state:
            self.render_pricing_results(st.session_state.pricing_result)
    
    def _get_risk_premium(self, risk_category: str) -> float:
        """Get default risk premium based on category"""
        premiums = {
            "Low": 1.5,
            "Medium": 3.0,
            "High": 6.0,
            "Very High": 9.0
        }
        return premiums.get(risk_category, 3.0)
    
    def render_pricing_results(self, result):
        """Render pricing calculation results"""
        st.markdown("---")
        st.subheader("💰 Optimal Pricing Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Recommended Rate",
                f"{result['recommended_rate'] * 100:.2f}%",
                help="Optimal interest rate considering all factors"
            )
        
        with col2:
            st.metric(
                "Base Break-even Rate",
                f"{result['break_even_rate'] * 100:.2f}%",
                help="Minimum rate to cover costs"
            )
        
        with col3:
            st.metric(
                "Risk-Adjusted Return",
                f"{result['risk_adjusted_return'] * 100:.2f}%",
                help="Expected return after risk adjustments"
            )
        
        with col4:
            competitiveness = "Competitive" if result['market_competitive'] else "Review Needed"
            st.metric(
                "Market Position",
                competitiveness,
                help="Comparison with market rates"
            )
        
        # Pricing breakdown
        st.markdown("#### 📊 Pricing Component Breakdown")
        
        components_data = {
            'Component': ['Cost of Funds', 'Operating Cost', 'Risk Provision', 'Target Return', 'Risk Premium'],
            'Percentage': [
                result['cost_of_funds'] * 100,
                result['operating_cost'] * 100,
                result['risk_provision'] * 100,
                result['target_roa'] * 100,
                result['risk_premium'] * 100
            ]
        }
        
        fig = px.pie(
            components_data,
            values='Percentage',
            names='Component',
            title="Interest Rate Component Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        self.render_sensitivity_analysis(result)
    
    def render_sensitivity_analysis(self, base_result):
        """Render pricing sensitivity analysis"""
        st.markdown("#### 📈 Pricing Sensitivity Analysis")
        
        # Generate sensitivity scenarios
        scenarios = self.pricing_optimizer.sensitivity_analysis(base_result)
        
        fig = go.Figure()
        
        # Cost of funds sensitivity
        fig.add_trace(go.Scatter(
            x=scenarios['cost_of_funds']['values'] * 100,
            y=scenarios['cost_of_funds']['rates'] * 100,
            name='Cost of Funds',
            line=dict(color='blue', width=3)
        ))
        
        # Operating cost sensitivity
        fig.add_trace(go.Scatter(
            x=scenarios['operating_cost']['values'] * 100,
            y=scenarios['operating_cost']['rates'] * 100,
            name='Operating Cost',
            line=dict(color='red', width=3)
        ))
        
        # Risk premium sensitivity
        fig.add_trace(go.Scatter(
            x=scenarios['risk_premium']['values'] * 100,
            y=scenarios['risk_premium']['rates'] * 100,
            name='Risk Premium',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Pricing Sensitivity to Cost Components",
            xaxis_title="Cost Component Change (%)",
            yaxis_title="Recommended Interest Rate (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_deposit_pricing(self):
        """Render deposit pricing analysis"""
        st.markdown("---")
        st.subheader("🏦 Deposit Pricing Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Current Deposit Rates")
            
            deposit_products = pd.DataFrame({
                'Product': ['Savings Account', 'Fixed Deposit 30D', 'Fixed Deposit 90D', 'Fixed Deposit 180D', 'Fixed Deposit 1Y'],
                'Current_Rate': [3.5, 5.2, 6.8, 7.5, 8.2],
                'Market_Average': [3.8, 5.5, 7.0, 7.8, 8.5],
                'Balance_KES_M': [45.2, 28.5, 35.8, 42.1, 38.9],
                'Growth_Rate': [5.2, 3.8, 4.5, 6.2, 5.8]
            })
            
            fig = px.bar(
                deposit_products,
                x='Product',
                y=['Current_Rate', 'Market_Average'],
                title="Deposit Product Rates vs Market",
                barmode='group',
                labels={'value': 'Interest Rate (%)', 'variable': 'Rate Type'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Deposit Cost Analysis")
            
            # Deposit cost composition
            cost_components = pd.DataFrame({
                'Component': ['Interest Cost', 'Operating Cost', 'Transaction Cost', 'Reserve Cost'],
                'Percentage': [68, 15, 12, 5]
            })
            
            fig = px.pie(
                cost_components,
                values='Percentage',
                names='Component',
                title="Deposit Cost Composition",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Deposit pricing recommendations
            st.markdown("##### 💡 Pricing Recommendations")
            st.success("**Savings Accounts**: Consider increasing to 4.0% to match market")
            st.warning("**Fixed Deposits 90D**: Competitive, maintain current rates")
            st.error("**Fixed Deposits 1Y**: Below market, review for increase")
    
    def render_competitor_analysis(self):
        """Render competitor pricing analysis"""
        st.markdown("---")
        st.subheader("🏆 Competitor Pricing Analysis")
        
        # Sample competitor data
        competitors = ['SACCO A', 'SACCO B', 'Commercial Bank', 'Microfinance', 'Market Average']
        products = ['Personal Loan', 'Business Loan', 'Savings Account', 'Fixed Deposit']
        
        # Generate sample competitor rates
        competitor_data = []
        for competitor in competitors:
            for product in products:
                base_rate = {
                    'Personal Loan': 14.0,
                    'Business Loan': 16.0,
                    'Savings Account': 3.5,
                    'Fixed Deposit': 7.0
                }[product]
                
                # Add some variation
                variation = np.random.uniform(-2.0, 2.0)
                rate = max(base_rate + variation, 0.5)
                
                competitor_data.append({
                    'Competitor': competitor,
                    'Product': product,
                    'Rate': rate
                })
        
        competitor_df = pd.DataFrame(competitor_data)
        
        # Heatmap of competitor rates
        pivot_df = competitor_df.pivot(index='Competitor', columns='Product', values='Rate')
        
        fig = px.imshow(
            pivot_df,
            title="Competitor Pricing Heatmap",
            color_continuous_scale='RdYlGn_r',
            aspect='auto'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market positioning
        st.markdown("#### 🎯 Market Positioning Analysis")
        
        positioning_data = pd.DataFrame({
            'Product': products * 2,
            'Rate_Type': ['Our Rates'] * len(products) + ['Market Average'] * len(products),
            'Rate': [13.5, 15.2, 3.5, 7.2, 14.5, 16.0, 3.8, 7.0]
        })
        
        fig = px.bar(
            positioning_data,
            x='Product',
            y='Rate',
            color='Rate_Type',
            barmode='group',
            title="Our Rates vs Market Average",
            labels={'Rate': 'Interest Rate (%)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_pricing_strategy(self):
        """Render pricing strategy recommendations"""
        st.markdown("---")
        st.subheader("🎯 Strategic Pricing Recommendations")
        
        strategy_recommendations = """
        ### Recommended Pricing Actions
        
        **1. Loan Products:**
        - **Personal Loans**: Increase to 14.5% (currently 13.5%) to align with risk-adjusted returns
        - **Business Loans**: Maintain at 15.2% - competitive positioning with good margins
        - **Emergency Loans**: Consider risk-based pricing tiers (18-22% based on credit score)
        
        **2. Deposit Products:**
        - **Savings Accounts**: Increase to 4.0% to improve deposit growth
        - **Fixed Deposits**: Optimize term structure - increase 90D to 7.2%
        - **Introduce**: Premium savings account at 4.5% for balances >KES 100,000
        
        **3. Strategic Objectives:**
        - Achieve NIM of 8.5% (current 8.4%)
        - Improve deposit mix towards longer-term fixed deposits
        - Implement risk-based pricing for all loan products
        - Monitor competitor moves quarterly
        """
        
        st.markdown(strategy_recommendations)
        
        # Implementation timeline
        with st.expander("📅 Pricing Implementation Timeline"):
            timeline_data = {
                'Phase': ['Immediate (1-2 weeks)', 'Short-term (1 month)', 'Medium-term (3 months)', 'Long-term (6 months)'],
                'Actions': [
                    'Adjust savings account rates\nReview emergency loan pricing',
                    'Implement risk-based pricing framework\nLaunch premium savings product',
                    'Optimize fixed deposit term structure\nCompetitor analysis update',
                    'Advanced pricing analytics implementation\nDynamic pricing capabilities'
                ]
            }
            
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True)
    
    def run(self):
        """Run the pricing economics page"""
        st.title("💳 Pricing Economics & Optimization")
        
        st.markdown("""
        Comprehensive loan and deposit pricing analysis, optimization, and competitor benchmarking 
        to maximize SACCO profitability while maintaining market competitiveness.
        """)
        
        self.render_pricing_dashboard()
        self.render_pricing_optimizer()
        self.render_deposit_pricing()
        self.render_competitor_analysis()
        self.render_pricing_strategy()

if __name__ == "__main__":
    page = PricingEconomicsPage()
    page.run()