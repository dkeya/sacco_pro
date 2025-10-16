# pages/13_Collections_Recovery.py
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
from sacco_core.analytics.collections import CollectionsAnalyzer, DelinquencyBucket, CollectionStrategy, RecoveryProbability

st.set_page_config(
    page_title="Collections & Recovery",
    page_icon="📞",
    layout="wide"
)

class CollectionsRecoveryPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.collections_analyzer = CollectionsAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "13_Collections_Recovery.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "collections_recovery_page"
        )
        return True
    
    def render_collections_dashboard(self):
        """Render collections and recovery dashboard"""
        st.subheader("📞 Collections & Recovery Management")
        
        try:
            # Get collections analysis
            analysis = self.collections_analyzer.analyze_collections_portfolio()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_stats = analysis.get('portfolio_segmentation', {}).get('portfolio_statistics', {})
                total_delinquent = portfolio_stats.get('total_delinquent_loans', 0)
                st.metric(
                    "Delinquent Accounts",
                    f"{total_delinquent}",
                    help="Total number of delinquent loans"
                )
            
            with col2:
                total_outstanding = portfolio_stats.get('total_outstanding_amount', 0)
                st.metric(
                    "Outstanding Amount",
                    f"KES {total_outstanding:,.0f}",
                    help="Total outstanding amount in delinquent portfolio"
                )
            
            with col3:
                performance_metrics = analysis.get('performance_metrics', {})
                recovery_rate = performance_metrics.get('recovery_rate', 0) * 100
                st.metric(
                    "Recovery Rate",
                    f"{recovery_rate:.1f}%",
                    help="Percentage of successful recovery actions"
                )
            
            with col4:
                efficiency_score = performance_metrics.get('collection_efficiency_score', 0) * 100
                st.metric(
                    "Efficiency Score",
                    f"{efficiency_score:.1f}%",
                    help="Overall collections efficiency score"
                )
            
            # Collections overview
            self.render_collections_overview(analysis)
            
        except Exception as e:
            st.error(f"Error rendering collections dashboard: {str(e)}")
            st.info("Please check the data connection and try again.")
    
    def render_collections_overview(self, analysis):
        """Render collections overview"""
        st.markdown("#### 📊 Collections Portfolio Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delinquency bucket distribution
            bucket_segmentation = analysis.get('portfolio_segmentation', {}).get('bucket_segmentation', {})
            if bucket_segmentation:
                buckets = list(bucket_segmentation.keys())
                counts = list(bucket_segmentation.values())
                
                colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
                
                fig = px.bar(
                    x=buckets,
                    y=counts,
                    title="Delinquency Bucket Distribution",
                    labels={'x': 'Delinquency Bucket', 'y': 'Number of Loans'},
                    color=buckets,
                    color_discrete_sequence=colors
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delinquency bucket data available")
        
        with col2:
            # Recovery probability distribution
            strategy_loans = analysis.get('strategy_recommendations', [])
            if strategy_loans:
                recovery_levels = {}
                for loan in strategy_loans:
                    level = loan.recovery_probability.value
                    recovery_levels[level] = recovery_levels.get(level, 0) + 1
                
                if recovery_levels:
                    levels = list(recovery_levels.keys())
                    counts = list(recovery_levels.values())
                    
                    fig = px.pie(
                        names=levels,
                        values=counts,
                        title="Recovery Probability Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recovery probability data available")
        
        # Detailed analysis sections
        self.render_portfolio_segmentation(analysis)
        self.render_strategy_recommendations(analysis)
        self.render_agent_performance(analysis)
        self.render_workflow_optimization(analysis)
        self.render_performance_analytics(analysis)
    
    def render_portfolio_segmentation(self, analysis):
        """Render detailed portfolio segmentation"""
        st.markdown("---")
        st.subheader("🏷️ Portfolio Segmentation Analysis")
        
        portfolio_segmentation = analysis.get('portfolio_segmentation', {})
        
        if portfolio_segmentation:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Amount Segmentation")
                
                amount_segmentation = portfolio_segmentation.get('amount_segmentation', {})
                if amount_segmentation:
                    segments = list(amount_segmentation.keys())
                    counts = list(amount_segmentation.values())
                    
                    fig = px.bar(
                        x=segments,
                        y=counts,
                        title="Portfolio by Outstanding Amount",
                        labels={'x': 'Amount Segment', 'y': 'Number of Loans'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Portfolio Statistics")
                
                portfolio_stats = portfolio_segmentation.get('portfolio_statistics', {})
                stats_data = {
                    'Metric': [
                        'Total Delinquent Loans',
                        'Total Outstanding',
                        'Average Delinquency Days',
                        'Oldest Delinquency'
                    ],
                    'Value': [
                        portfolio_stats.get('total_delinquent_loans', 0),
                        f"KES {portfolio_stats.get('total_outstanding_amount', 0):,.0f}",
                        f"{portfolio_stats.get('average_delinquency_days', 0):.0f} days",
                        f"{portfolio_stats.get('oldest_delinquency', 0):.0f} days"
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # Top delinquent accounts
            st.markdown("#### ⚠️ Top Delinquent Accounts")
            top_accounts = portfolio_segmentation.get('top_delinquent_accounts', [])
            if top_accounts:
                accounts_data = []
                for account in top_accounts[:10]:  # Show top 10
                    accounts_data.append({
                        'Loan ID': account.loan_id,
                        'Member ID': account.member_id,
                        'Outstanding': f"KES {account.outstanding_amount:,.0f}",
                        'Days Delinquent': account.days_delinquent,
                        'Bucket': account.delinquency_bucket.value,
                        'Contact': account.contact_number
                    })
                
                accounts_df = pd.DataFrame(accounts_data)
                st.dataframe(accounts_df, use_container_width=True)
            else:
                st.info("No top delinquent accounts data available")
        else:
            st.info("No portfolio segmentation data available")
    
    def render_strategy_recommendations(self, analysis):
        """Render collection strategy recommendations"""
        st.markdown("---")
        st.subheader("🎯 Collection Strategy Recommendations")
        
        strategy_loans = analysis.get('strategy_recommendations', [])
        
        if strategy_loans:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Strategy Distribution")
                
                strategy_distribution = {}
                for loan in strategy_loans:
                    strategy = loan.recommended_strategy.value
                    strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
                
                if strategy_distribution:
                    strategies = list(strategy_distribution.keys())
                    counts = list(strategy_distribution.values())
                    
                    fig = px.pie(
                        names=strategies,
                        values=counts,
                        title="Recommended Strategy Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 High-Priority Actions")
                
                # Filter high-priority strategies
                high_priority_strategies = [CollectionStrategy.NEGOTIATION, CollectionStrategy.RESTRUCTURING, CollectionStrategy.LEGAL_ACTION]
                high_priority_loans = [loan for loan in strategy_loans if loan.recommended_strategy in high_priority_strategies]
                
                if high_priority_loans:
                    priority_data = []
                    for loan in high_priority_loans[:15]:  # Show top 15
                        priority_data.append({
                            'Loan ID': loan.loan_id,
                            'Strategy': loan.recommended_strategy.value,
                            'Recovery Probability': loan.recovery_probability.value,
                            'Amount': f"KES {loan.outstanding_amount:,.0f}",
                            'Days Delinquent': loan.days_delinquent
                        })
                    
                    priority_df = pd.DataFrame(priority_data)
                    st.dataframe(priority_df, use_container_width=True)
                else:
                    st.info("No high-priority actions identified")
            
            # Strategy implementation interface
            st.markdown("#### 🚀 Strategy Implementation")
            
            selected_loan = st.selectbox(
                "Select Loan for Action",
                [f"{loan.loan_id} - {loan.member_name} (KES {loan.outstanding_amount:,.0f})" 
                 for loan in strategy_loans[:50]]
            )
            
            if selected_loan:
                loan_id = selected_loan.split(' - ')[0]
                selected_loan_data = next((loan for loan in strategy_loans if loan.loan_id == loan_id), None)
                
                if selected_loan_data:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Recommended Strategy", selected_loan_data.recommended_strategy.value)
                    
                    with col2:
                        st.metric("Recovery Probability", selected_loan_data.recovery_probability.value)
                    
                    with col3:
                        st.metric("Days Delinquent", selected_loan_data.days_delinquent)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Assign to Agent", key=f"assign_{loan_id}"):
                            st.success(f"Loan {loan_id} assigned to collection agent")
                    
                    with col2:
                        if st.button("📞 Initiate Contact", key=f"contact_{loan_id}"):
                            st.info(f"Contact initiated for {selected_loan_data.member_name}")
                    
                    with col3:
                        if st.button("🔄 Update Strategy", key=f"update_{loan_id}"):
                            st.warning(f"Strategy update initiated for {loan_id}")
        else:
            st.info("No strategy recommendations available")
    
    def render_agent_performance(self, analysis):
        """Render agent performance analysis"""
        st.markdown("---")
        st.subheader("👥 Collection Agent Performance")
        
        agent_analysis = analysis.get('agent_analysis', {})
        agent_performance = agent_analysis.get('agent_performance', {})
        
        if agent_performance:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Performance Metrics")
                
                # Create performance table
                performance_data = []
                for agent_id, data in agent_performance.items():
                    performance_data.append({
                        'Agent ID': agent_id,
                        'Total Actions': data['total_actions'],
                        'Success Rate': f"{data['success_rate'] * 100:.1f}%",
                        'Promises to Pay': data['promises_to_pay'],
                        'Avg Promise Amount': f"KES {data.get('average_promise_amount', 0):,.0f}"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 🏆 Top Performers")
                
                top_performers = agent_analysis.get('top_performers', [])
                if top_performers:
                    for agent_id, data in top_performers:
                        with st.expander(f"⭐ {agent_id} - Success Rate: {data['success_rate'] * 100:.1f}%"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Actions", data['total_actions'])
                                st.metric("Successful Actions", data['successful_actions'])
                            with col2:
                                st.metric("Promises to Pay", data['promises_to_pay'])
                                st.metric("Promise Amount", f"KES {data.get('total_promise_amount', 0):,.0f}")
            
            # Action type analysis
            st.markdown("#### 📈 Action Type Effectiveness")
            all_actions = {}
            for agent_data in agent_performance.values():
                for action_type, count in agent_data.get('actions_by_type', {}).items():
                    all_actions[action_type] = all_actions.get(action_type, 0) + count
            
            if all_actions:
                action_types = list(all_actions.keys())
                action_counts = list(all_actions.values())
                
                fig = px.bar(
                    x=action_types,
                    y=action_counts,
                    title="Collection Actions by Type",
                    labels={'x': 'Action Type', 'y': 'Number of Actions'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent performance data available")
    
    def render_workflow_optimization(self, analysis):
        """Render workflow optimization recommendations"""
        st.markdown("---")
        st.subheader("⚙️ Workflow Optimization")
        
        workflow_optimization = analysis.get('workflow_optimization', {})
        
        if workflow_optimization:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Resource Allocation")
                
                resource_allocation = workflow_optimization.get('resource_allocation', {})
                if resource_allocation:
                    allocation_data = []
                    for strategy, allocation in resource_allocation.items():
                        allocation_data.append({
                            'Strategy': strategy,
                            'Agents Needed': allocation['agents_needed'],
                            'Priority': allocation['priority']
                        })
                    
                    allocation_df = pd.DataFrame(allocation_data)
                    st.dataframe(allocation_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 🤖 Automation Opportunities")
                
                automation_opportunities = workflow_optimization.get('automation_opportunities', [])
                if automation_opportunities:
                    for opportunity in automation_opportunities:
                        with st.expander(f"🔄 {opportunity['process']}"):
                            st.write(f"**Automation Type:** {opportunity['automation_type']}")
                            st.write(f"**Estimated Savings:** KES {opportunity['estimated_savings']:,.0f}")
                            st.write(f"**Timeline:** {opportunity['implementation_timeline']}")
                            
                            if st.button("Implement Automation", key=f"auto_{opportunity['process']}"):
                                st.success(f"Automation implementation started for {opportunity['process']}")
                else:
                    st.info("No automation opportunities identified")
            
            # Workload suggestions
            st.markdown("#### ⚖️ Workload Management")
            workload_suggestions = workflow_optimization.get('workload_suggestions', [])
            if workload_suggestions:
                for suggestion in workload_suggestions:
                    priority_color = {
                        'High': 'red',
                        'Medium': 'orange',
                        'Low': 'green'
                    }.get(suggestion['priority'], 'gray')
                    
                    st.warning(f"**{suggestion['type']}** - {suggestion['description']}")
            else:
                st.success("✅ Workload is well balanced")
        else:
            st.info("No workflow optimization data available")
    
    def render_performance_analytics(self, analysis):
        """Render performance analytics and trends"""
        st.markdown("---")
        st.subheader("📈 Performance Analytics & Trends")
        
        performance_metrics = analysis.get('performance_metrics', {})
        recovery_trends = analysis.get('recovery_trends', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Key Performance Indicators")
            
            kpi_data = {
                'Metric': [
                    'Recovery Rate',
                    'Promise-to-Pay Keep Rate',
                    'Average Days to Recovery',
                    'Cost of Collections',
                    'Efficiency Score'
                ],
                'Value': [
                    f"{performance_metrics.get('recovery_rate', 0) * 100:.1f}%",
                    f"{performance_metrics.get('promise_to_pay_keep_rate', 0) * 100:.1f}%",
                    f"{performance_metrics.get('average_days_to_recovery', 0):.0f} days",
                    f"KES {performance_metrics.get('cost_of_collections', 0):,.0f}",
                    f"{performance_metrics.get('collection_efficiency_score', 0) * 100:.1f}%"
                ]
            }
            
            kpi_df = pd.DataFrame(kpi_data)
            st.dataframe(kpi_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 📅 Recovery Trends")
            
            monthly_rates = recovery_trends.get('monthly_recovery_rates', {})
            if monthly_rates:
                months = list(monthly_rates.keys())
                rates = [rate * 100 for rate in monthly_rates.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months,
                    y=rates,
                    name='Recovery Rate',
                    line=dict(color='green', width=3),
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title="Monthly Recovery Rate Trend",
                    xaxis_title="Month",
                    yaxis_title="Recovery Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Strategy effectiveness
        st.markdown("#### 🎯 Strategy Effectiveness")
        strategy_effectiveness = recovery_trends.get('strategy_effectiveness', {})
        if strategy_effectiveness:
            strategies = list(strategy_effectiveness.keys())
            effectiveness = [eff * 100 for eff in strategy_effectiveness.values()]
            
            fig = px.bar(
                x=strategies,
                y=effectiveness,
                title="Strategy Effectiveness (%)",
                labels={'x': 'Collection Strategy', 'y': 'Effectiveness (%)'},
                color=effectiveness,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the collections and recovery page"""
        st.title("📞 Collections & Recovery Management")
        
        st.markdown("""
        Comprehensive delinquency management, recovery optimization, and workflow automation 
        to maximize collections effectiveness, minimize losses, and optimize resource allocation 
        through data-driven strategy recommendations and performance monitoring.
        """)
        
        try:
            self.render_collections_dashboard()
        except Exception as e:
            st.error(f"Error running collections page: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    page = CollectionsRecoveryPage()
    page.run()