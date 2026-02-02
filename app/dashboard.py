import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_and_prepare_data
from src.features import FeatureEngineer
from src.evaluation import calculate_metrics
from src.utils import load_model


# Page configuration
st.set_page_config(
    page_title="Learning Progress Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Learning Progress Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Dashboard Settings")
view_option = st.sidebar.selectbox(
    "Select View",
    ["Overview", "Student Analysis", "Model Performance", "Risk Assessment"]
)

# Load data
@st.cache_data
def load_data():
    """Load and cache data"""
    train_df, valid_df, test_df = load_and_prepare_data()
    return train_df, valid_df, test_df

try:
    train_df, valid_df, test_df = load_data()
    from src.utils import get_semester_order
    train_df['semester_order']=train_df['HOC_KY'].apply(get_semester_order)

    # Create features
    engineer = FeatureEngineer()
    valid_df = engineer.create_features(valid_df)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# Overview Page
if view_option == "Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(valid_df['MA_SO_SV'].unique()))
    
    with col2:
        avg_credits = valid_df['TC_DANGKY'].mean()
        st.metric("Avg Credits Registered", f"{avg_credits:.1f}")
    
    with col3:
        avg_completed = valid_df['TC_HOANTHANH'].mean()
        st.metric("Avg Credits Completed", f"{avg_completed:.1f}")
    
    with col4:
        completion_rate = (valid_df['TC_HOANTHANH'].sum() / valid_df['TC_DANGKY'].sum()) * 100
        st.metric("Overall Completion Rate", f"{completion_rate:.1f}%")
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Credits Registered")
        fig1 = px.histogram(
            valid_df, x='TC_DANGKY',
            nbins=30,
            title="Credits Registered Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Distribution of Credits Completed")
        fig2 = px.histogram(
            valid_df, x='TC_HOANTHANH',
            nbins=30,
            title="Credits Completed Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # GPA and CPA distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("GPA Distribution")
        fig3 = px.histogram(
            valid_df, x='GPA',
            nbins=30,
            title="GPA Distribution"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("CPA Distribution")
        fig4 = px.histogram(
            valid_df, x='CPA',
            nbins=30,
            title="CPA Distribution"
        )
        st.plotly_chart(fig4, use_container_width=True)


# Student Analysis Page
elif view_option == "Student Analysis":
    st.header("Student Analysis")
    
    # Student search
    student_id = st.text_input("Enter Student ID (MA_SO_SV):")
    
    if student_id:
        student_data = valid_df[valid_df['MA_SO_SV'] == student_id]
        
        if len(student_data) > 0:
            st.success(f"Student {student_id} found!")
            
            student_info = student_data.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Admission Year", student_info.get('NAM_TUYENSINH', 'N/A'))
                st.metric("Admission Score", f"{student_info.get('DIEM_TRUNGTUYEN', 0):.2f}")
            
            with col2:
                st.metric("Current GPA", f"{student_info.get('GPA', 0):.2f}")
                st.metric("Current CPA", f"{student_info.get('CPA', 0):.2f}")
            
            with col3:
                st.metric("Credits Registered", student_info.get('TC_DANGKY', 0))
                st.metric("Credits Completed", student_info.get('TC_HOANTHANH', 0))
            
            # Historical performance
            st.subheader("Historical Performance")
            student_history = train_df[train_df['MA_SO_SV'] == student_id].sort_values('semester_order')
            
            if len(student_history) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=student_history['HOC_KY'],
                    y=student_history['GPA'],
                    mode='lines+markers',
                    name='GPA',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=student_history['HOC_KY'],
                    y=student_history['CPA'],
                    mode='lines+markers',
                    name='CPA',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title="GPA and CPA Trend",
                    xaxis_title="Semester",
                    yaxis_title="Score",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for this student.")
        else:
            st.warning(f"Student {student_id} not found in validation set.")
    
    # Segment analysis
    st.markdown("---")
    st.subheader("Analysis by Segments")
    
    segment_option = st.selectbox(
        "Select Segment",
        ["Admission Method (PTXT)", "Admission Year", "GPA Level"]
    )
    
    if segment_option == "Admission Method (PTXT)":
        segment_col = 'PTXT'
    elif segment_option == "Admission Year":
        segment_col = 'NAM_TUYENSINH'
    else:
        valid_df['GPA_Level'] = pd.cut(
            valid_df['GPA'],
            bins=[0, 2.5, 3.2, 3.6, 4.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        segment_col = 'GPA_Level'
    
    segment_stats = valid_df.groupby(segment_col).agg({
        'TC_DANGKY': 'mean',
        'TC_HOANTHANH': 'mean',
        'MA_SO_SV': 'count'
    }).reset_index()
    segment_stats.columns = [segment_col, 'Avg Credits Registered', 'Avg Credits Completed', 'Count']
    segment_stats['Completion Rate'] = (segment_stats['Avg Credits Completed'] / segment_stats['Avg Credits Registered'] * 100).round(2)
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Visualization
    fig = px.bar(
        segment_stats,
        x=segment_col,
        y='Completion Rate',
        title=f"Completion Rate by {segment_option}",
        color='Completion Rate',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)


# Model Performance Page
elif view_option == "Model Performance":
    st.header("Model Performance")
    
    st.info("Upload predictions to see model performance metrics")
    
    uploaded_file = st.file_uploader("Upload predictions CSV file", type=['csv'])
    
    if uploaded_file is not None:
        predictions_df = pd.read_csv(uploaded_file)
        
        if 'MA_SO_SV' in predictions_df.columns and 'PRED_TC_HOANTHANH' in predictions_df.columns:
            # Merge with actual values
            eval_df = valid_df[['MA_SO_SV', 'TC_HOANTHANH']].merge(
                predictions_df,
                on='MA_SO_SV',
                how='inner'
            )
            
            if len(eval_df) > 0:
                y_true = eval_df['TC_HOANTHANH']
                y_pred = eval_df['PRED_TC_HOANTHANH']
                
                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                
                with col3:
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                
                with col4:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                
                # Scatter plot
                st.subheader("Predictions vs Actual")
                fig = px.scatter(
                    x=y_true,
                    y=y_pred,
                    labels={'x': 'Actual TC_HOANTHANH', 'y': 'Predicted TC_HOANTHANH'},
                    title="Predictions vs Actual Values"
                )
                
                # Add perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Error distribution
                errors = y_true - y_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Error Distribution")
                    fig = px.histogram(
                        x=errors,
                        nbins=50,
                        title="Distribution of Prediction Errors"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Error Statistics")
                    error_stats = pd.DataFrame({
                        'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error'],
                        'Value': [errors.mean(), errors.std(), errors.min(), errors.max()]
                    })
                    st.dataframe(error_stats, use_container_width=True)
            else:
                st.warning("No matching student IDs found between predictions and validation set.")
        else:
            st.error("CSV file must contain 'MA_SO_SV' and 'PRED_TC_HOANTHANH' columns.")


# Risk Assessment Page
elif view_option == "Risk Assessment":
    st.header("Risk Assessment")
    
    # Define at-risk students
    valid_df['completion_rate'] = valid_df['TC_HOANTHANH'] / valid_df['TC_DANGKY'] * 100
    valid_df['risk_level'] = pd.cut(
        valid_df['completion_rate'],
        bins=[0, 50, 75, 90, 100],
        labels=['High Risk', 'Medium Risk', 'Low Risk', 'On Track']
    )
    
    # Risk distribution
    risk_counts = valid_df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Distribution")
        st.dataframe(risk_counts, use_container_width=True)
        
        # Calculate percentages
        total = risk_counts['Count'].sum()
        high_risk_pct = (risk_counts[risk_counts['Risk Level'] == 'High Risk']['Count'].values[0] / total * 100) if len(risk_counts[risk_counts['Risk Level'] == 'High Risk']) > 0 else 0
        st.metric("High Risk Students", f"{high_risk_pct:.1f}%")
    
    with col2:
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk Level',
            title="Student Risk Distribution",
            color='Risk Level',
            color_discrete_map={
                'High Risk': 'red',
                'Medium Risk': 'orange',
                'Low Risk': 'yellow',
                'On Track': 'green'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High risk students
    st.markdown("---")
    st.subheader("High Risk Students")
    
    high_risk_students = valid_df[valid_df['risk_level'] == 'High Risk'][
        ['MA_SO_SV', 'GPA', 'CPA', 'TC_DANGKY', 'TC_HOANTHANH', 'completion_rate']
    ].sort_values('completion_rate')
    
    st.dataframe(high_risk_students, use_container_width=True)
    
    # Download high risk list
    csv = high_risk_students.to_csv(index=False)
    st.download_button(
        label="Download High Risk Students List",
        data=csv,
        file_name="high_risk_students.csv",
        mime="text/csv"
    )
    
    # Intervention recommendations
    st.markdown("---")
    st.subheader("Recommended Interventions")
    
    st.markdown("""
    ### For High Risk Students:
    1. **Academic Support**: Provide tutoring and study groups
    2. **Counseling**: Offer academic counseling sessions
    3. **Course Load**: Recommend reducing credit load next semester
    4. **Monitoring**: Increase check-ins with academic advisors
    
    ### For Medium Risk Students:
    1. **Early Warning**: Send early warning notifications
    2. **Resources**: Share study resources and time management tools
    3. **Peer Support**: Connect with peer mentors
    
    ### Preventive Measures:
    1. **Registration Guidance**: Help students choose appropriate course loads
    2. **Academic Planning**: Assist with semester planning
    3. **Support Services**: Promote available campus support services
    """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Learning Progress Prediction Dashboard | Multour | DATAFLOW 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)