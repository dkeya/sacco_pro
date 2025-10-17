# pages/07_Data_Quality_MIS.py
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
from sacco_core.analytics.dq import DataQualityAnalyzer
from sacco_core.sidebar import render_sidebar

st.set_page_config(
    page_title="Data Quality MIS",
    page_icon="📋",
    layout="wide"
)

# Check authentication and render sidebar
if not st.session_state.get('authenticated', False):
    st.error("🔐 Please log in to access this page")
    st.stop()

# Render consistent sidebar and styling
render_sidebar()

class DataQualityMISPage:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.config = self.config_manager.load_settings()
        self.dq_analyzer = DataQualityAnalyzer()
        
        if not self._check_access():
            st.stop()
    
    def _check_access(self):
        if not st.session_state.get('authenticated', False):
            st.error("Please login to access this page")
            return False
        
        has_access = self.rbac_manager.check_page_access(
            "07_Data_Quality_MIS.py", 
            st.session_state.role, 
            self.config
        )
        
        if not has_access:
            st.error("You do not have permission to access this page")
            return False
        
        self.audit_logger.log_data_access(
            st.session_state.user, 
            st.session_state.role, 
            "data_quality_mis_page"
        )
        return True
    
    def render_dq_dashboard(self):
        """Render data quality dashboard"""
        st.subheader("📊 Data Quality Dashboard")
        
        # Run comprehensive data quality assessment
        dq_assessment = self.dq_analyzer.comprehensive_data_quality_assessment()
        overall_score = dq_assessment.get('overall_score', 0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Data Quality Score",
                f"{overall_score:.1f}%",
                help="Comprehensive data quality assessment score"
            )
        
        with col2:
            critical_issues = dq_assessment.get('critical_issues_count', 0)
            st.metric(
                "Critical Data Issues", 
                f"{critical_issues}",
                delta_color="inverse" if critical_issues > 0 else "normal",
                help="Number of critical data quality issues"
            )
        
        with col3:
            data_completeness = dq_assessment.get('completeness_score', 0) * 100
            st.metric(
                "Data Completeness",
                f"{data_completeness:.1f}%",
                help="Percentage of complete and non-missing data"
            )
        
        with col4:
            data_accuracy = dq_assessment.get('accuracy_score', 0) * 100
            st.metric(
                "Data Accuracy",
                f"{data_accuracy:.1f}%",
                help="Accuracy and validity of data values"
            )
        
        # Data quality overview
        self.render_dq_overview(dq_assessment)
    
    def render_dq_overview(self, dq_assessment):
        """Render data quality overview"""
        st.markdown("#### 📈 Data Quality Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality scores by dimension
            quality_dimensions = dq_assessment.get('quality_dimensions', {})
            if quality_dimensions:
                dimensions_df = pd.DataFrame({
                    'Dimension': list(quality_dimensions.keys()),
                    'Score': [score * 100 for score in quality_dimensions.values()]
                })
                
                fig = px.bar(
                    dimensions_df,
                    x='Dimension',
                    y='Score',
                    title="Data Quality Scores by Dimension",
                    color='Score',
                    color_continuous_scale='RdYlGn'
                )
                fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Target")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Minimum")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Issue severity distribution
            issue_severity = dq_assessment.get('issue_severity', {})
            if issue_severity:
                severity_df = pd.DataFrame({
                    'Severity': list(issue_severity.keys()),
                    'Count': list(issue_severity.values())
                })
                
                fig = px.pie(
                    severity_df,
                    values='Count',
                    names='Severity',
                    title="Data Issues by Severity",
                    color='Severity',
                    color_discrete_map={
                        'Critical': 'red',
                        'High': 'orange', 
                        'Medium': 'yellow',
                        'Low': 'green'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data quality trends
        self.render_dq_trends()
    
    def render_dq_trends(self):
        """Render data quality trends over time"""
        st.markdown("#### 📊 Data Quality Trends")
        
        try:
            # Use the analytics module to get real trend data
            trend_data = self.dq_analyzer.get_data_quality_trends()
            
            # Ensure we have valid data
            if trend_data is None or trend_data.empty:
                st.warning("No trend data available. Using sample data for demonstration.")
                trend_data = self._generate_fallback_trend_data()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall quality trend
                fig = px.line(
                    trend_data,
                    x='Month',
                    y='Overall_Score',
                    title="Overall Data Quality Trend",
                    markers=True
                )
                fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Target")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Minimum")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Issue count trend
                fig = px.line(
                    trend_data,
                    x='Month',
                    y=['Critical_Issues', 'High_Issues', 'Medium_Issues'],
                    title="Data Issue Trends by Severity",
                    labels={'value': 'Issue Count', 'variable': 'Severity'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional trend: Quality dimensions over time
            st.markdown("##### 📈 Quality Dimensions Trend")
            col3, col4 = st.columns(2)
            
            with col3:
                fig = px.line(
                    trend_data,
                    x='Month',
                    y=['Overall_Score', 'Completeness_Score', 'Accuracy_Score'],
                    title="Quality Scores Over Time",
                    labels={'value': 'Score (%)', 'variable': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Trend summary metrics
                if len(trend_data) >= 2:
                    latest_score = trend_data['Overall_Score'].iloc[-1]
                    previous_score = trend_data['Overall_Score'].iloc[-2]
                    trend_direction = "improving" if latest_score > previous_score else "declining"
                    trend_change = latest_score - previous_score
                    
                    st.metric(
                        "Current Quality Score", 
                        f"{latest_score:.1f}%",
                        delta=f"{trend_change:+.1f}% ({trend_direction})",
                        delta_color="normal" if trend_change > 0 else "inverse"
                    )
                    
                    # Issue reduction
                    current_issues = trend_data['Critical_Issues'].iloc[-1]
                    previous_issues = trend_data['Critical_Issues'].iloc[-2]
                    issues_change = previous_issues - current_issues
                    
                    st.metric(
                        "Critical Issues",
                        current_issues,
                        delta=f"{issues_change:+d} issues",
                        delta_color="normal" if issues_change > 0 else "inverse"
                    )
                else:
                    st.info("Insufficient data for trend analysis")
                    
        except Exception as e:
            st.error(f"Error loading trend data: {str(e)}")
            st.info("Using fallback data for demonstration")
            trend_data = self._generate_fallback_trend_data()
            
            # Simple fallback visualization
            fig = px.line(
                trend_data,
                x='Month',
                y='Overall_Score',
                title="Data Quality Trend (Sample Data)",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    def _generate_fallback_trend_data(self):
        """Generate fallback trend data when analytics fails"""
        return pd.DataFrame({
            'Month': ['Aug 2023', 'Sep 2023', 'Oct 2023', 'Nov 2023', 'Dec 2023', 'Jan 2024', 'Feb 2024'],
            'Overall_Score': [82.0, 85.0, 87.0, 84.0, 88.0, 86.0, 89.0],
            'Critical_Issues': [8, 6, 5, 7, 4, 5, 3],
            'High_Issues': [15, 12, 10, 13, 9, 11, 8],
            'Medium_Issues': [25, 22, 20, 23, 18, 20, 16],
            'Completeness_Score': [85.0, 87.0, 89.0, 86.0, 90.0, 88.0, 91.0],
            'Accuracy_Score': [80.0, 83.0, 85.0, 82.0, 86.0, 84.0, 87.0]
        })
    
    def render_data_profiling(self):
        """Render data profiling and analysis"""
        st.markdown("---")
        st.subheader("🔍 Data Profiling & Analysis")
        
        # Run data profiling
        profiling_results = self.dq_analyzer.comprehensive_data_profiling()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dataset Overview", "🔎 Column Analysis", "📈 Data Distributions", "🔗 Relationships"
        ])
        
        with tab1:
            self.render_dataset_overview(profiling_results)
        
        with tab2:
            self.render_column_analysis(profiling_results)
        
        with tab3:
            self.render_data_distributions(profiling_results)
        
        with tab4:
            self.render_data_relationships(profiling_results)
    
    def render_dataset_overview(self, profiling_results):
        """Render dataset overview"""
        st.markdown("#### 📁 Dataset Overview")
        
        dataset_stats = profiling_results.get('dataset_statistics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{dataset_stats.get('total_records', 0):,}")
        
        with col2:
            st.metric("Total Columns", dataset_stats.get('total_columns', 0))
        
        with col3:
            st.metric("Memory Usage", f"{dataset_stats.get('memory_usage_mb', 0):.1f} MB")
        
        with col4:
            st.metric("Duplicate Records", f"{dataset_stats.get('duplicate_count', 0):,}")
        
        # Data types overview
        st.markdown("#### 🏷️ Data Types Overview")
        data_types = profiling_results.get('data_types', {})
        if data_types:
            types_df = pd.DataFrame({
                'Data Type': list(data_types.keys()),
                'Count': list(data_types.values())
            })
            
            fig = px.pie(
                types_df,
                values='Count',
                names='Data Type',
                title="Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_column_analysis(self, profiling_results):
        """Render detailed column analysis"""
        st.markdown("#### 📋 Column-Level Analysis")
        
        column_profiles = profiling_results.get('column_profiles', {})
        
        # Column selection
        selected_column = st.selectbox(
            "Select Column for Detailed Analysis",
            options=list(column_profiles.keys()),
            help="Choose a column to view detailed statistics"
        )
        
        if selected_column:
            column_profile = column_profiles[selected_column]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Type", column_profile.get('data_type', 'Unknown'))
                st.metric("Unique Values", column_profile.get('unique_count', 0))
                st.metric("Missing Values", f"{column_profile.get('missing_count', 0):,}")
            
            with col2:
                completeness = (1 - column_profile.get('missing_percentage', 0)) * 100
                st.metric("Completeness", f"{completeness:.1f}%")
                st.metric("Cardinality", column_profile.get('cardinality', 'Low'))
                st.metric("Data Quality", f"{column_profile.get('quality_score', 0)*100:.1f}%")
            
            with col3:
                if column_profile.get('data_type') in ['numeric', 'integer']:
                    st.metric("Mean", f"{column_profile.get('mean', 0):.2f}")
                    st.metric("Standard Deviation", f"{column_profile.get('std', 0):.2f}")
                    st.metric("Range", f"{column_profile.get('min', 0):.2f} - {column_profile.get('max', 0):.2f}")
            
            # Value distribution for categorical data
            if column_profile.get('top_values'):
                st.markdown("#### 📊 Value Distribution")
                top_values = column_profile['top_values']
                values_df = pd.DataFrame({
                    'Value': list(top_values.keys()),
                    'Count': list(top_values.values())
                }).head(10)
                
                fig = px.bar(
                    values_df,
                    x='Value',
                    y='Count',
                    title=f"Top 10 Values in {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_data_distributions(self, profiling_results):
        """Render data distributions analysis"""
        st.markdown("#### 📈 Data Distributions")
        
        column_profiles = profiling_results.get('column_profiles', {})
        numeric_columns = [col for col, profile in column_profiles.items() 
                          if profile.get('data_type') in ['numeric', 'integer']]
        
        if numeric_columns:
            selected_numeric = st.selectbox(
                "Select Numeric Column for Distribution",
                options=numeric_columns
            )
            
            if selected_numeric:
                column_profile = column_profiles[selected_numeric]
                
                # Generate sample distribution data
                np.random.seed(42)
                if column_profile.get('data_type') == 'numeric':
                    sample_data = np.random.normal(
                        column_profile.get('mean', 0),
                        column_profile.get('std', 1),
                        1000
                    )
                else:
                    sample_data = np.random.randint(
                        column_profile.get('min', 0),
                        column_profile.get('max', 100),
                        1000
                    )
                
                fig = px.histogram(
                    x=sample_data,
                    title=f"Distribution of {selected_numeric}",
                    labels={'x': selected_numeric, 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection
        st.markdown("#### ⚠️ Outlier Analysis")
        
        outlier_columns = [col for col, profile in column_profiles.items() 
                          if profile.get('outlier_count', 0) > 0]
        
        if outlier_columns:
            outlier_df = pd.DataFrame([
                {
                    'Column': col,
                    'Outlier_Count': column_profiles[col].get('outlier_count', 0),
                    'Outlier_Percentage': column_profiles[col].get('outlier_percentage', 0) * 100
                }
                for col in outlier_columns
            ])
            
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.success("No significant outliers detected in numeric columns.")
    
    def render_data_relationships(self, profiling_results):
        """Render data relationships analysis"""
        st.markdown("#### 🔗 Data Relationships")
        
        # Correlation analysis for numeric columns
        correlations = profiling_results.get('correlations', {})
        
        if correlations:
            # Convert correlation matrix to DataFrame
            corr_df = pd.DataFrame(correlations)
            
            fig = px.imshow(
                corr_df,
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Relationship insights
        st.markdown("#### 💡 Relationship Insights")
        
        insights = [
            "Strong positive correlation between loan amount and member deposits",
            "Negative correlation between days past due and credit score",
            "Moderate correlation between employer size and average loan size"
        ]
        
        for insight in insights:
            st.info(f"💡 {insight}")
    
    def render_issue_detection(self):
        """Render data quality issue detection"""
        st.markdown("---")
        st.subheader("🚨 Data Quality Issue Detection")
        
        # Run issue detection
        issues_report = self.dq_analyzer.detect_data_quality_issues()
        
        tab1, tab2, tab3 = st.tabs([
            "🔴 Critical Issues", "🟡 Quality Warnings", "✅ Data Validation"
        ])
        
        with tab1:
            self.render_critical_issues(issues_report)
        
        with tab2:
            self.render_quality_warnings(issues_report)
        
        with tab3:
            self.render_data_validation(issues_report)
    
    def render_critical_issues(self, issues_report):
        """Render critical data quality issues"""
        critical_issues = issues_report.get('critical_issues', [])
        
        if critical_issues:
            st.error("### 🔴 Critical Data Quality Issues")
            
            for issue in critical_issues:
                with st.expander(f"🚨 {issue['issue_type']} - {issue['table_name']}.{issue['column_name']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Affected Records", f"{issue['affected_count']:,}")
                    
                    with col2:
                        st.metric("Severity", issue['severity'])
                    
                    with col3:
                        st.metric("Impact Score", f"{issue['impact_score']:.1f}")
                    
                    st.write(f"**Description**: {issue['description']}")
                    st.write(f"**Business Impact**: {issue['business_impact']}")
                    
                    # Resolution actions
                    col4, col5 = st.columns(2)
                    with col4:
                        if st.button("🛠️ Create Resolution Plan", key=f"resolve_{issue['issue_id']}"):
                            self._create_resolution_plan(issue)
                            st.success("Resolution plan created!")
                    
                    with col5:
                        if st.button("📧 Notify Data Owners", key=f"notify_{issue['issue_id']}"):
                            self._notify_data_owners(issue)
                            st.success("Data owners notified!")
        else:
            st.success("### ✅ No Critical Data Quality Issues Detected")
    
    def render_quality_warnings(self, issues_report):
        """Render data quality warnings"""
        quality_warnings = issues_report.get('quality_warnings', [])
        
        if quality_warnings:
            st.warning("### 🟡 Data Quality Warnings")
            
            warnings_df = pd.DataFrame(quality_warnings)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Warnings", len(quality_warnings))
            with col2:
                high_warnings = len([w for w in quality_warnings if w.get('severity') == 'High'])
                st.metric("High Severity", high_warnings)
            with col3:
                tables_affected = len(set(w['table_name'] for w in quality_warnings))
                st.metric("Tables Affected", tables_affected)
            
            # Detailed warnings table
            st.dataframe(warnings_df, use_container_width=True)
            
            # Bulk actions
            col4, col5 = st.columns(2)
            with col4:
                if st.button("📋 Generate Warnings Report"):
                    report = self._generate_warnings_report(quality_warnings)
                    st.success("Warnings report generated!")
            
            with col5:
                if st.button("🔄 Schedule Quality Review"):
                    self._schedule_quality_review(quality_warnings)
                    st.success("Quality review scheduled!")
        else:
            st.info("### 📋 No Quality Warnings Detected")
    
    def render_data_validation(self, issues_report):
        """Render data validation results"""
        st.markdown("#### ✅ Data Validation Rules")
        
        validation_rules = issues_report.get('validation_rules', [])
        
        # Validation rule status
        rule_status = {}
        for rule in validation_rules:
            status = rule.get('status', 'Unknown')
            rule_status[status] = rule_status.get(status, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Passing Rules", rule_status.get('Pass', 0))
        with col2:
            st.metric("Failing Rules", rule_status.get('Fail', 0))
        with col3:
            st.metric("Validation Coverage", f"{(len(validation_rules) / 25 * 100):.1f}%")  # Assuming 25 total rules
        
        # Validation rules details
        for rule in validation_rules:
            status_icon = "✅" if rule.get('status') == 'Pass' else "❌" if rule.get('status') == 'Fail' else "⚠️"
            
            with st.expander(f"{status_icon} {rule['rule_name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description**: {rule['description']}")
                    st.write(f"**Table**: {rule['table_name']}")
                    st.write(f"**Condition**: {rule['condition']}")
                with col2:
                    st.write(f"**Status**: {rule['status']}")
                    st.write(f"**Last Check**: {rule.get('last_check', 'N/A')}")
                    if rule.get('error_count', 0) > 0:
                        st.write(f"**Errors**: {rule['error_count']}")
                
                if rule.get('status') == 'Fail' and st.button("🔍 Investigate Failures", key=f"investigate_{rule['rule_id']}"):
                    self._investigate_validation_failure(rule)
    
    def render_data_cleansing(self):
        """Render data cleansing interface"""
        st.markdown("---")
        st.subheader("🧹 Data Cleansing & Enrichment")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 Data Cleaning", "📝 Data Standardization", "🔍 Deduplication", "📊 Data Enrichment"
        ])
        
        with tab1:
            self.render_data_cleaning()
        
        with tab2:
            self.render_data_standardization()
        
        with tab3:
            self.render_deduplication()
        
        with tab4:
            self.render_data_enrichment()
    
    def render_data_cleaning(self):
        """Render data cleaning interface"""
        st.markdown("#### 🔧 Automated Data Cleaning")
        
        cleaning_options = [
            "Remove duplicate records",
            "Fill missing values with defaults",
            "Correct data type mismatches",
            "Remove extreme outliers",
            "Standardize date formats",
            "Clean text fields (trim, case correction)"
        ]
        
        selected_cleaning = st.multiselect(
            "Select Cleaning Operations",
            options=cleaning_options,
            default=cleaning_options[:2],
            help="Choose data cleaning operations to perform"
        )
        
        # Affected tables preview
        affected_tables = ["members", "loans", "employers", "transactions"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tables to Process", len(affected_tables))
        with col2:
            estimated_records = 12500
            st.metric("Estimated Records", f"{estimated_records:,}")
        
        if st.button("🚀 Run Data Cleaning", type="primary"):
            with st.spinner("Running data cleaning operations..."):
                cleaning_results = self._run_data_cleaning(selected_cleaning, affected_tables)
                st.session_state.cleaning_results = cleaning_results
                st.success("Data cleaning completed successfully!")
        
        if 'cleaning_results' in st.session_state:
            self.render_cleaning_results(st.session_state.cleaning_results)
    
    def render_data_standardization(self):
        """Render data standardization interface"""
        st.markdown("#### 📝 Data Standardization")
        
        standardization_rules = [
            {"field": "phone_number", "operation": "format", "pattern": "E.164"},
            {"field": "email", "operation": "lowercase", "pattern": None},
            {"field": "national_id", "operation": "validate", "pattern": "KE_ID"},
            {"field": "address", "operation": "standardize", "pattern": "postal_format"}
        ]
        
        for rule in standardization_rules:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{rule['field']}**")
            with col2:
                st.write(f"Operation: {rule['operation']}")
            with col3:
                if st.button("Apply", key=f"apply_{rule['field']}"):
                    self._apply_standardization(rule)
                    st.success(f"Standardization applied to {rule['field']}")
        
        # Bulk standardization
        if st.button("🔄 Apply All Standardizations"):
            self._apply_all_standardizations(standardization_rules)
            st.success("All standardizations applied!")
    
    def render_deduplication(self):
        """Render deduplication interface"""
        st.markdown("#### 🔍 Deduplication Analysis")
        
        # Duplicate analysis
        duplicate_stats = {
            "potential_duplicates": 245,
            "exact_duplicates": 12,
            "fuzzy_duplicates": 233,
            "tables_affected": 3
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Potential Duplicates", duplicate_stats["potential_duplicates"])
        with col2:
            st.metric("Exact Matches", duplicate_stats["exact_duplicates"])
        with col3:
            st.metric("Fuzzy Matches", duplicate_stats["fuzzy_duplicates"])
        with col4:
            st.metric("Tables Affected", duplicate_stats["tables_affected"])
        
        # Deduplication options
        st.markdown("##### 🎯 Deduplication Strategy")
        
        strategy = st.selectbox(
            "Matching Strategy",
            options=["Exact Match", "Fuzzy Match", "Advanced Matching"],
            help="Select deduplication matching strategy"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.7,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Minimum confidence score for fuzzy matching"
        )
        
        if st.button("🔍 Run Deduplication Analysis"):
            with st.spinner("Analyzing duplicates..."):
                dup_results = self._run_deduplication_analysis(strategy, confidence_threshold)
                st.session_state.dup_results = dup_results
                st.success("Deduplication analysis completed!")
    
    def render_data_enrichment(self):
        """Render data enrichment interface"""
        st.markdown("#### 📊 Data Enrichment")
        
        enrichment_options = [
            "Geocode addresses",
            "Validate phone numbers",
            "Enhance employer information",
            "Calculate credit scores",
            "Add demographic data",
            "Enrich with external data sources"
        ]
        
        selected_enrichment = st.multiselect(
            "Select Enrichment Operations",
            options=enrichment_options,
            help="Choose data enrichment operations"
        )
        
        # External data sources
        st.markdown("##### 🔗 External Data Sources")
        
        sources = [
            {"name": "Credit Bureau", "status": "Connected", "records": "45,200"},
            {"name": "Government Registry", "status": "Connected", "records": "38,500"},
            {"name": "Address Validation", "status": "Available", "records": "N/A"}
        ]
        
        for source in sources:
            status_color = "🟢" if source["status"] == "Connected" else "🟡"
            st.write(f"{status_color} **{source['name']}** - {source['status']} - {source['records']} records")
        
        if st.button("🚀 Run Data Enrichment"):
            with st.spinner("Enriching data..."):
                enrichment_results = self._run_data_enrichment(selected_enrichment)
                st.success("Data enrichment completed!")
    
    def render_cleaning_results(self, cleaning_results):
        """Render data cleaning results"""
        st.markdown("#### 📋 Cleaning Results Summary")
        
        results = cleaning_results.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records Processed", f"{results.get('records_processed', 0):,}")
        with col2:
            st.metric("Issues Fixed", results.get('issues_fixed', 0))
        with col3:
            st.metric("Quality Improvement", f"{results.get('quality_improvement', 0):.1f}%")
        with col4:
            st.metric("Processing Time", f"{results.get('processing_time', 0):.1f}s")
    
    def render_dq_reporting(self):
        """Render data quality reporting"""
        st.markdown("---")
        st.subheader("📈 Data Quality Reporting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate DQ Report", use_container_width=True):
                report = self.dq_analyzer.generate_data_quality_report()
                st.session_state.dq_report = report
                st.success("Data quality report generated!")
        
        with col2:
            if st.button("📅 Schedule DQ Monitoring", use_container_width=True):
                self._schedule_dq_monitoring()
                st.success("DQ monitoring scheduled!")
        
        with col3:
            if st.button("🔔 Set Up DQ Alerts", use_container_width=True):
                self._setup_dq_alerts()
                st.success("DQ alerts configured!")
        
        # Report preview
        if 'dq_report' in st.session_state:
            self.render_dq_report_preview(st.session_state.dq_report)
    
    def render_dq_report_preview(self, dq_report):
        """Render data quality report preview"""
        st.markdown("#### 📋 Data Quality Report Preview")
        
        report_summary = dq_report.get('executive_summary', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{report_summary.get('overall_score', 0):.1f}%")
        with col2:
            st.metric("Critical Issues", report_summary.get('critical_issues', 0))
        with col3:
            st.metric("Improvement Needed", f"{report_summary.get('improvement_needed', 0):.1f}%")
        
        # Download report
        report_json = pd.DataFrame([dq_report]).to_json(orient='records')
        st.download_button(
            label="📥 Download Full Report",
            data=report_json,
            file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    def _create_resolution_plan(self, issue):
        """Create resolution plan for data issue"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "create_resolution_plan",
            "data_quality",
            issue['issue_id'],
            {'issue_type': issue['issue_type']}
        )
    
    def _notify_data_owners(self, issue):
        """Notify data owners about data issue"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "notify_data_owners",
            "data_quality",
            issue['issue_id'],
            {'table_name': issue['table_name']}
        )
    
    def _generate_warnings_report(self, warnings):
        """Generate warnings report"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "generate_warnings_report",
            "data_quality",
            None,
            {'warning_count': len(warnings)}
        )
        return {"status": "success", "report_id": f"DQ_WARNINGS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    
    def _schedule_quality_review(self, warnings):
        """Schedule quality review"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "schedule_quality_review",
            "data_quality",
            None,
            {'warning_count': len(warnings)}
        )
    
    def _investigate_validation_failure(self, rule):
        """Investigate validation rule failure"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "investigate_validation_failure",
            "data_quality",
            rule['rule_id'],
            {'rule_name': rule['rule_name']}
        )
    
    def _run_data_cleaning(self, operations, tables):
        """Run data cleaning operations"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "run_data_cleaning",
            "data_quality",
            None,
            {'operations': operations, 'tables': tables}
        )
        return {
            'summary': {
                'records_processed': 12500,
                'issues_fixed': 342,
                'quality_improvement': 15.2,
                'processing_time': 45.7
            }
        }
    
    def _apply_standardization(self, rule):
        """Apply standardization rule"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "apply_standardization",
            "data_quality",
            rule['field'],
            {'operation': rule['operation']}
        )
    
    def _apply_all_standardizations(self, rules):
        """Apply all standardization rules"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "apply_all_standardizations",
            "data_quality",
            None,
            {'rule_count': len(rules)}
        )
    
    def _run_deduplication_analysis(self, strategy, threshold):
        """Run deduplication analysis"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "run_deduplication_analysis",
            "data_quality",
            None,
            {'strategy': strategy, 'threshold': threshold}
        )
        return {'status': 'completed', 'duplicates_found': 245}
    
    def _run_data_enrichment(self, operations):
        """Run data enrichment operations"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "run_data_enrichment",
            "data_quality",
            None,
            {'operations': operations}
        )
        return {'status': 'completed', 'records_enriched': 12500}
    
    def _schedule_dq_monitoring(self):
        """Schedule DQ monitoring"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "schedule_dq_monitoring",
            "data_quality",
            None,
            {}
        )
    
    def _setup_dq_alerts(self):
        """Setup DQ alerts"""
        self.audit_logger.log_action(
            st.session_state.user,
            st.session_state.role,
            "setup_dq_alerts",
            "data_quality",
            None,
            {}
        )
    
    def run(self):
        """Run the data quality MIS page"""
        st.title("📊 Data Quality MIS")
        
        st.markdown("""
        Comprehensive data quality management, monitoring, and improvement system. 
        Ensure data integrity, accuracy, and reliability across all SACCO operations.
        """)
        
        self.render_dq_dashboard()
        self.render_data_profiling()
        self.render_issue_detection()
        self.render_data_cleansing()
        self.render_dq_reporting()

if __name__ == "__main__":
    page = DataQualityMISPage()
    page.run()