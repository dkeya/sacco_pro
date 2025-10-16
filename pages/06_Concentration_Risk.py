# pages/06_Concentration_Risk.py
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
    page_title="Concentration Risk",
    page_icon="🎯",
    layout="wide"
)

class ConcentrationRiskPage:
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
            "06_Concentration_Risk.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "concentration_risk_page"
        )
        return True
    
    def safe_get_config_limit(self, limit_name, default=0.25):
        """Safely get configuration limits with fallbacks"""
        try:
            if hasattr(self.config, 'limits') and hasattr(self.config.limits, limit_name):
                return getattr(self.config.limits, limit_name)
            return default
        except Exception:
            return default
    
    def render_concentration_dashboard(self):
        """Render concentration risk dashboard"""
        st.subheader("📊 Concentration Risk Dashboard")
        
        try:
            # Get concentration analysis
            analysis = self.concentration_analyzer.analyze_concentration_risk()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                single_employer_limit = self.safe_get_config_limit('single_employer_share_max', 0.25) * 100
                current_single = analysis.get('employer_concentration', {}).get('single_largest_share', 0) * 100
                delta_value = current_single - single_employer_limit
                st.metric(
                    "Single Employer Exposure",
                    f"{current_single:.1f}%",
                    f"{delta_value:+.1f}% vs limit",
                    delta_color="inverse" if current_single > single_employer_limit else "normal",
                    help=f"Maximum allowed: {single_employer_limit:.1f}%"
                )
            
            with col2:
                top_5_share = analysis.get('employer_concentration', {}).get('top_5_share', 0) * 100
                st.metric(
                    "Top 5 Employers Share", 
                    f"{top_5_share:.1f}%",
                    help="Percentage of portfolio from top 5 employers"
                )
            
            with col3:
                product_concentration = analysis.get('product_concentration', {}).get('concentration_risk', {}).get('herfindahl_index', 0)
                st.metric(
                    "Product Concentration",
                    f"{product_concentration:.3f}",
                    help="Herfindahl Index (0-1, higher = more concentrated)"
                )
            
            with col4:
                breach_count = analysis.get('regulatory_breaches', {}).get('total_breaches', 0)
                st.metric(
                    "Regulatory Breaches",
                    f"{breach_count}",
                    delta_color="inverse" if breach_count > 0 else "normal",
                    help="Number of concentration limit breaches"
                )
            
            # Concentration overview
            self.render_concentration_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering concentration dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_concentration_overview(self, analysis):
        """Render concentration risk overview"""
        st.markdown("#### 🎯 Concentration Risk Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Employer concentration chart
            employer_data = analysis.get('employer_concentration', {}).get('top_employers', [])
            if employer_data:
                try:
                    employer_df = pd.DataFrame(employer_data)
                    
                    fig = px.bar(
                        employer_df.head(10),
                        x='employer_name',
                        y='exposure_share',
                        title="Top 10 Employer Exposures (% of Portfolio)",
                        labels={'exposure_share': 'Portfolio Share (%)', 'employer_name': 'Employer'},
                        color='exposure_share',
                        color_continuous_scale='RdYlGn_r'
                    )
                    single_employer_limit = self.safe_get_config_limit('single_employer_share_max', 0.25)
                    fig.add_hline(
                        y=single_employer_limit,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Single Employer Limit"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering employer chart: {str(e)}")
            else:
                st.info("No employer concentration data available")
        
        with col2:
            # Product concentration
            product_data = analysis.get('product_concentration', {}).get('product_shares', {})
            if product_data:
                try:
                    product_df = pd.DataFrame({
                        'Product': list(product_data.keys()),
                        'Share': [share * 100 for share in product_data.values()]
                    })
                    
                    fig = px.pie(
                        product_df,
                        values='Share',
                        names='Product',
                        title="Loan Portfolio by Product Type",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering product chart: {str(e)}")
            else:
                st.info("No product concentration data available")
        
        # Geographic concentration
        self.render_geographic_concentration(analysis)
    
    def render_geographic_concentration(self, analysis):
        """Render geographic concentration analysis"""
        st.markdown("#### 🌍 Geographic Concentration")
        
        geographic_data = analysis.get('geographic_concentration', {})
        
        if geographic_data:
            try:
                regions_df = pd.DataFrame({
                    'Region': list(geographic_data.keys()),
                    'Exposure_Share': [data.get('exposure_share', 0) * 100 for data in geographic_data.values()],
                    'Member_Count': [data.get('member_count', 0) for data in geographic_data.values()],
                    'Average_Balance': [data.get('average_balance', 0) for data in geographic_data.values()]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Regional exposure chart
                    fig = px.bar(
                        regions_df,
                        x='Region',
                        y='Exposure_Share',
                        title="Portfolio Exposure by Region (%)",
                        color='Exposure_Share',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Regional member distribution
                    fig = px.scatter(
                        regions_df,
                        x='Member_Count',
                        y='Average_Balance',
                        size='Exposure_Share',
                        color='Region',
                        hover_name='Region',
                        title="Regional Analysis: Members vs Average Balance",
                        labels={'Member_Count': 'Number of Members', 'Average_Balance': 'Average Loan Balance (KES)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error rendering geographic charts: {str(e)}")
        else:
            st.info("No geographic concentration data available")
    
    def render_employer_analysis(self):
        """Render detailed employer concentration analysis"""
        st.markdown("---")
        st.subheader("🏢 Employer Concentration Analysis")
        
        try:
            # Get employer concentration data
            analysis = self.concentration_analyzer.analyze_employer_concentration()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Employer Exposure Trends")
                
                # Employer concentration over time
                trend_data = analysis.get('trend_analysis', {})
                if trend_data:
                    try:
                        periods = list(trend_data.keys())
                        top_5_trend = [data.get('top_5_share', 0) * 100 for data in trend_data.values()]
                        single_largest_trend = [data.get('single_largest_share', 0) * 100 for data in trend_data.values()]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=periods,
                            y=top_5_trend,
                            name='Top 5 Employers Share',
                            line=dict(color='blue', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=periods,
                            y=single_largest_trend,
                            name='Single Largest Employer',
                            line=dict(color='red', width=3)
                        ))
                        single_employer_limit = self.safe_get_config_limit('single_employer_share_max', 0.25) * 100
                        fig.add_hline(
                            y=single_employer_limit,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Single Employer Limit"
                        )
                        fig.update_layout(
                            title="Employer Concentration Trends",
                            xaxis_title="Period",
                            yaxis_title="Portfolio Share (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering trend chart: {str(e)}")
                else:
                    st.info("No trend data available")
            
            with col2:
                st.markdown("#### ⚠️ Concentration Risk Indicators")
                
                risk_indicators = analysis.get('risk_indicators', {})
                
                # Herfindahl Index
                hhi = risk_indicators.get('herfindahl_index', 0)
                st.metric(
                    "Herfindahl-Hirschman Index",
                    f"{hhi:.4f}",
                    help="HHI > 0.25 indicates high concentration"
                )
                
                # Gini Coefficient
                gini = risk_indicators.get('gini_coefficient', 0)
                st.metric(
                    "Gini Coefficient",
                    f"{gini:.3f}",
                    help="Measures inequality in exposure distribution"
                )
                
                # Concentration Ratio
                cr4 = risk_indicators.get('concentration_ratio_4', 0) * 100
                st.metric(
                    "CR4 (Top 4 Employers)",
                    f"{cr4:.1f}%",
                    help="Share of top 4 employers"
                )
                
                # Number of significant exposures
                significant_exposures = risk_indicators.get('significant_exposures_count', 0)
                st.metric(
                    "Significant Exposures",
                    f"{significant_exposures}",
                    help="Exposures > 5% of portfolio"
                )
            
            # Employer exposure details
            self.render_employer_exposure_details(analysis)
            
        except Exception as e:
            st.error(f"Error rendering employer analysis: {str(e)}")
    
    def render_employer_exposure_details(self, analysis):
        """Render detailed employer exposure table"""
        st.markdown("#### 📋 Detailed Employer Exposures")
        
        employer_data = analysis.get('employer_exposures', [])
        if employer_data:
            try:
                employer_df = pd.DataFrame(employer_data)
                
                # Calculate additional metrics
                if 'exposure_share' in employer_df.columns:
                    employer_df['exposure_share_pct'] = employer_df['exposure_share'] * 100
                    single_employer_limit = self.safe_get_config_limit('single_employer_share_max', 0.25)
                    employer_df['breach_status'] = employer_df['exposure_share'].apply(
                        lambda x: 'BREACH' if x > single_employer_limit else 'WITHIN LIMIT'
                    )
                    employer_df['risk_category'] = employer_df['exposure_share'].apply(
                        lambda x: 'High' if x > 0.15 else 'Medium' if x > 0.08 else 'Low'
                    )
                
                # Color coding for breach status
                def color_breach_status(status):
                    if status == 'BREACH':
                        return 'background-color: #FFB6C1'
                    else:
                        return 'background-color: #90EE90'
                
                if 'breach_status' in employer_df.columns:
                    styled_employers = employer_df.style.applymap(
                        color_breach_status, subset=['breach_status']
                    )
                    st.dataframe(styled_employers, use_container_width=True)
                else:
                    st.dataframe(employer_df, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Generate Concentration Report"):
                        try:
                            report = self.concentration_analyzer.generate_concentration_report(analysis)
                            st.success("Concentration risk report generated!")
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                
                with col2:
                    try:
                        csv_data = employer_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Employer Data",
                            data=csv_data,
                            file_name=f"employer_concentration_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error preparing download: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error processing employer data: {str(e)}")
        else:
            st.info("No detailed employer exposure data available")
    
    def render_product_concentration(self):
        """Render product concentration analysis"""
        st.markdown("---")
        st.subheader("💳 Product Concentration Analysis")
        
        try:
            analysis = self.concentration_analyzer.analyze_product_concentration()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Product Portfolio Mix")
                
                product_data = analysis.get('product_shares', {})
                product_quality = analysis.get('product_quality', {})
                
                if product_data and product_quality:
                    try:
                        product_df = pd.DataFrame({
                            'Product': list(product_data.keys()),
                            'Share': [share * 100 for share in product_data.values()],
                            'PAR_Ratio': [product_quality.get(product, {}).get('par_30', 0) * 100 
                                         for product in product_data.keys()]
                        })
                        
                        # Product share vs PAR scatter
                        fig = px.scatter(
                            product_df,
                            x='Share',
                            y='PAR_Ratio',
                            size='Share',
                            color='Product',
                            hover_name='Product',
                            title="Product Concentration vs Risk (PAR 30)",
                            labels={'Share': 'Portfolio Share (%)', 'PAR_Ratio': 'PAR 30 (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering product scatter chart: {str(e)}")
                else:
                    st.info("No product mix data available")
            
            with col2:
                st.markdown("#### 📈 Product Performance Metrics")
                
                product_quality = analysis.get('product_quality', {})
                if product_quality:
                    try:
                        metrics_data = []
                        for product, metrics in product_quality.items():
                            metrics_data.append({
                                'Product': product,
                                'PAR_30': metrics.get('par_30', 0) * 100,
                                'NPL_Ratio': metrics.get('npl_ratio', 0) * 100,
                                'Average_Balance': metrics.get('average_balance', 0),
                                'Growth_Rate': metrics.get('growth_rate', 0) * 100
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering product metrics: {str(e)}")
                else:
                    st.info("No product quality data available")
            
            # Product concentration risk
            self.render_product_risk_analysis(analysis)
            
        except Exception as e:
            st.error(f"Error rendering product concentration: {str(e)}")
    
    def render_product_risk_analysis(self, analysis):
        """Render product concentration risk analysis"""
        st.markdown("#### ⚠️ Product Concentration Risk")
        
        risk_analysis = analysis.get('concentration_risk', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hhi = risk_analysis.get('herfindahl_index', 0)
            st.metric(
                "Product HHI",
                f"{hhi:.4f}",
                help="Herfindahl Index for product concentration"
            )
        
        with col2:
            dominant_product_share = risk_analysis.get('dominant_product_share', 0) * 100
            st.metric(
                "Largest Product Share",
                f"{dominant_product_share:.1f}%",
                help="Share of largest product category"
            )
        
        with col3:
            risk_score = risk_analysis.get('overall_risk_score', 0)
            risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
            st.metric(
                "Product Concentration Risk",
                risk_level,
                help="Overall product concentration risk assessment"
            )
        
        # Product diversification strategy
        st.markdown("#### 🎯 Product Diversification Strategy")
        
        diversification_needs = analysis.get('diversification_recommendations', [])
        if diversification_needs:
            for recommendation in diversification_needs:
                priority = recommendation.get('priority', 'Medium')
                action = recommendation.get('action', 'No action specified')
                if priority == 'High':
                    st.error(f"🚨 {action}")
                elif priority == 'Medium':
                    st.warning(f"⚠️ {action}")
                else:
                    st.info(f"💡 {action}")
        else:
            st.info("No diversification recommendations at this time")
    
    def render_mitigation_strategies(self):
        """Render concentration risk mitigation strategies"""
        st.markdown("---")
        st.subheader("🛡️ Concentration Risk Mitigation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Mitigation Strategies")
            
            mitigation_strategies = """
            **1. Employer Concentration:**
            - Implement strict single employer limits
            - Develop employer diversification strategy
            - Enhance risk-based pricing for concentrated exposures
            - Regular monitoring of top employer exposures
            
            **2. Product Concentration:**
            - Develop new product offerings
            - Targeted marketing for under-represented segments
            - Product portfolio optimization
            - Regular product performance reviews
            
            **3. Geographic Concentration:**
            - Expand to new geographic markets
            - Digital channels for wider reach
            - Branch network optimization
            - Local market risk assessment
            
            **4. General Strategies:**
            - Portfolio stress testing
            - Concentration limits in credit policy
            - Regular board reporting
            - Early warning indicators
            """
            
            st.info(mitigation_strategies)
        
        with col2:
            st.markdown("#### 🎯 Limit Monitoring Framework")
            
            # Current limits and utilization
            limits_data = {
                'Limit Type': ['Single Employer', 'Top 5 Employers', 'Product HHI', 'Geographic HHI'],
                'Current Limit': ['25%', '60%', '0.25', '0.30'],
                'Current Utilization': ['22.8%', '52.4%', '0.18', '0.22'],
                'Status': ['Within Limit', 'Within Limit', 'Within Limit', 'Within Limit']
            }
            
            try:
                limits_df = pd.DataFrame(limits_data)
                
                # Color code status
                def color_limit_status(status):
                    if status == 'Within Limit':
                        return 'color: green; font-weight: bold'
                    else:
                        return 'color: red; font-weight: bold'
                
                styled_limits = limits_df.style.applymap(
                    color_limit_status, subset=['Status']
                )
                
                st.dataframe(styled_limits, use_container_width=True)
                
                # Limit utilization chart
                fig = px.bar(
                    limits_df,
                    x='Limit Type',
                    y='Current Utilization',
                    title="Limit Utilization Analysis",
                    color='Status',
                    color_discrete_map={'Within Limit': 'green', 'Breach': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering limit monitoring: {str(e)}")
    
    def render_regulatory_compliance(self):
        """Render regulatory compliance section"""
        st.markdown("---")
        st.subheader("⚖️ Regulatory Compliance")
        
        try:
            analysis = self.concentration_analyzer.analyze_regulatory_compliance()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Compliance Status")
                
                compliance_data = analysis.get('compliance_status', {})
                breaches = analysis.get('regulatory_breaches', [])
                
                # Overall compliance score
                compliance_score = compliance_data.get('overall_score', 0) * 100
                st.metric(
                    "Regulatory Compliance Score",
                    f"{compliance_score:.1f}%",
                    help="Overall compliance with concentration limits"
                )
                
                # Breach details
                if breaches:
                    st.error("**Regulatory Breaches Identified:**")
                    for breach in breaches:
                        description = breach.get('description', 'Unknown breach')
                        current_value = breach.get('current_value', 0) * 100
                        limit_value = breach.get('limit_value', 0) * 100
                        st.write(f"❌ {description}")
                        st.write(f"   Current: {current_value:.1f}% | Limit: {limit_value:.1f}%")
                else:
                    st.success("✅ No regulatory breaches identified")
            
            with col2:
                st.markdown("#### 📈 Compliance Trends")
                
                trend_data = analysis.get('compliance_trends', {})
                if trend_data:
                    try:
                        periods = list(trend_data.keys())
                        scores = [data.get('compliance_score', 0) * 100 for data in trend_data.values()]
                        breach_counts = [data.get('breach_count', 0) for data in trend_data.values()]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=periods,
                            y=scores,
                            name='Compliance Score',
                            line=dict(color='green', width=3),
                            yaxis='y1'
                        ))
                        fig.add_trace(go.Bar(
                            x=periods,
                            y=breach_counts,
                            name='Breach Count',
                            marker_color='red',
                            yaxis='y2'
                        ))
                        fig.update_layout(
                            title="Concentration Compliance Trends",
                            xaxis_title="Period",
                            yaxis=dict(title='Compliance Score (%)', side='left'),
                            yaxis2=dict(title='Breach Count', side='right', overlaying='y')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering compliance trends: {str(e)}")
                else:
                    st.info("No compliance trend data available")
                    
        except Exception as e:
            st.error(f"Error rendering regulatory compliance: {str(e)}")
    
    def render_early_warning_indicators(self):
        """Render early warning indicators"""
        st.markdown("---")
        st.subheader("🚨 Early Warning Indicators")
        
        try:
            warning_indicators = self.concentration_analyzer.calculate_early_warning_indicators()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                employer_hhi_trend = warning_indicators.get('employer_hhi_trend', 0)
                trend_status = "Increasing" if employer_hhi_trend > 0.05 else "Stable" if employer_hhi_trend > -0.05 else "Decreasing"
                trend_color = "inverse" if employer_hhi_trend > 0.05 else "normal"
                st.metric(
                    "Employer HHI Trend",
                    trend_status,
                    delta=f"{employer_hhi_trend:.3f}",
                    delta_color=trend_color,
                    help="Quarterly change in employer concentration"
                )
            
            with col2:
                new_exposure_growth = warning_indicators.get('new_exposure_growth', 0) * 100
                growth_status = "High" if new_exposure_growth > 20 else "Moderate" if new_exposure_growth > 10 else "Low"
                st.metric(
                    "New Exposure Growth",
                    f"{new_exposure_growth:.1f}%",
                    help="Growth in new large exposures"
                )
            
            with col3:
                limit_utilization = warning_indicators.get('limit_utilization', 0) * 100
                utilization_status = "High" if limit_utilization > 80 else "Moderate" if limit_utilization > 60 else "Low"
                st.metric(
                    "Limit Utilization",
                    f"{limit_utilization:.1f}%",
                    utilization_status,
                    delta_color="inverse" if limit_utilization > 80 else "normal",
                    help="Average utilization of concentration limits"
                )
            
            # Warning alerts
            alerts = warning_indicators.get('alerts', [])
            if alerts:
                st.markdown("#### ⚠️ Active Alerts")
                for alert in alerts:
                    severity = alert.get('severity', 'Medium')
                    message = alert.get('message', 'No message')
                    if severity == 'High':
                        st.error(f"🔴 {message}")
                    elif severity == 'Medium':
                        st.warning(f"🟡 {message}")
                    else:
                        st.info(f"🔵 {message}")
            else:
                st.success("✅ No active alerts")
                
        except Exception as e:
            st.error(f"Error rendering early warning indicators: {str(e)}")
    
    def run(self):
        """Run the concentration risk page"""
        st.title("🎯 Concentration Risk Management")
        
        st.markdown("""
        Comprehensive monitoring and analysis of portfolio concentrations across employers, products, 
        and geographic regions to ensure compliance with regulatory limits and maintain a well-diversified portfolio.
        """)
        
        try:
            self.render_concentration_dashboard()
            self.render_employer_analysis()
            self.render_product_concentration()
            self.render_mitigation_strategies()
            self.render_regulatory_compliance()
            self.render_early_warning_indicators()
        except Exception as e:
            st.error(f"Error running concentration risk page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = ConcentrationRiskPage()
    page.run()