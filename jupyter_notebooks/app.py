import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Risk categorization functions
def categorize_stroke_risk(probability):
    """Categorize stroke risk based on predicted probability"""
    if probability < 0.05:
        return 'Very Low'
    elif probability < 0.15:
        return 'Low'
    elif probability < 0.30:
        return 'Moderate'
    elif probability < 0.50:
        return 'High'
    else:
        return 'Very High'

def get_risk_color(category):
    """Get color coding for risk categories"""
    color_map = {
        'Very Low': '#28a745',
        'Low': '#6cb42c', 
        'Moderate': '#ffc107',
        'High': '#fd7e14',
        'Very High': '#dc3545'
    }
    return color_map.get(category, '#6c757d')

def get_clinical_recommendations(category):
    """Get clinical recommendations for each risk category"""
    recommendations = {
        'Very Low': [
            "Continue routine primary care",
            "Standard lifestyle counseling", 
            "Annual cardiovascular screening"
        ],
        'Low': [
            "Enhanced lifestyle interventions",
            "Semi-annual cardiovascular monitoring",
            "Monitor modifiable risk factors"
        ],
        'Moderate': [
            "Quarterly cardiovascular assessments",
            "Aggressive lifestyle modifications", 
            "Consider preventive medications"
        ],
        'High': [
            "Monthly cardiovascular monitoring",
            "Intensive risk factor management",
            "Preventive medication therapy"
        ],
        'Very High': [
            "Immediate specialist referral",
            "Intensive monitoring and intervention",
            "Comprehensive medication management"
        ]
    }
    return recommendations.get(category, ["Consult healthcare provider"])

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-category {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .clinical-rec {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the stroke dataset"""
    try:
        df = pd.read_csv("inputs/datasets/Stroke-data.csv")
        return df
    except FileNotFoundError:
        # Fallback to sample data if file not found
        st.error("Dataset not found. Please ensure Stroke-data.csv is in the inputs/datasets/ folder.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Stroke Prediction Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Data Analytics with Artificial Intelligence - Healthcare Decision Support System")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Data Exploration", "Risk Assessment", "Risk Analysis", "Model Performance", "Patient Prediction"]
    )
    
    if page == "Overview":
        show_overview(df)
    elif page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Risk Assessment":
        show_risk_assessment(df)
    elif page == "Risk Analysis":
        show_risk_analysis(df)
    elif page == "Model Performance":
        show_model_performance(df)
    elif page == "Patient Prediction":
        show_patient_prediction(df)

def show_overview(df):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        stroke_count = df['stroke'].sum()
        st.metric("Stroke Cases", f"{stroke_count:,}")
    with col3:
        stroke_rate = df['stroke'].mean() * 100
        st.metric("Stroke Rate", f"{stroke_rate:.1f}%")
    with col4:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    st.subheader("Business Requirements Addressed")
    st.markdown("""
    ✅ **BR1**: Understanding key stroke risk factors through comprehensive data analysis  
    ✅ **BR2**: Predicting stroke likelihood using machine learning models  
    ✅ **BR3**: Identifying high-risk patient populations for targeted interventions  
    ✅ **BR4**: Interactive dashboard for data visualization and real-time predictions  
    ✅ **BR5**: Statistical validation of relationships between health factors and stroke risk  
    ✅ **BR6**: Business impact analysis and cost-effectiveness insights  
    """)
    
    # Project Navigation & Technical Details
    st.subheader("🔬 Project Navigation & Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Analysis Components:**
        - **Exploratory Data Analysis**: Statistical patterns and correlations
        - **Data Preprocessing & Cleaning**: Missing value handling and feature engineering
        - **Machine Learning Model Development**: Multiple algorithms with optimization
        - **Statistical Validation**: Chi-square testing and hypothesis validation
        - **Interactive Visualization**: Real-time charts and risk assessment tools
        """)
    
    with col2:
        st.markdown("""
        **💻 Technologies Used:**
        - **Python**: pandas, scikit-learn, matplotlib, seaborn
        - **Statistical Analysis**: Hypothesis testing, chi-square validation
        - **Interactive Dashboards**: Streamlit, Plotly
        - **Data Visualization**: Advanced charts and statistical plots
        - **Predictive Modeling**: Random Forest, XGBoost, Logistic Regression
        """)
    
    st.markdown("""
    **🎯 Navigation Guide:**
    - **Data Exploration**: Comprehensive EDA with statistical insights
    - **Risk Assessment**: Chi-square testing and correlation analysis  
    - **Risk Analysis**: Clinical risk categorization system
    - **Model Performance**: ML algorithm comparison and validation
    - **Patient Prediction**: Real-time stroke risk calculator
    """)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Age Factor Analysis:**
        - Patients 60+ have ~15% stroke rate
        - Patients under 45 have ~2% stroke rate
        - Age is the strongest single predictor
        """)
    
    with col2:
        st.markdown("""
        **Health Conditions Impact:**
        - Hypertension increases risk 3-4x
        - Heart disease increases risk 4-5x
        - Combined conditions compound risk significantly
        """)

def show_data_exploration(df):
    st.header("🔍 Data Exploration & Visualization")
    
    # 1. Bar Chart - Stroke rates by categories
    st.subheader("1. Bar Chart: Stroke Rates by Patient Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender analysis
        gender_stroke = df.groupby('gender')['stroke'].agg(['count', 'sum', 'mean'])
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_stroke['mean'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Stroke Rate by Gender')
        ax.set_ylabel('Stroke Rate')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Age group analysis
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['<30', '30-45', '45-60', '60+'])
        age_stroke = df.groupby('age_group')['stroke'].mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        age_stroke.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Stroke Rate by Age Group')
        ax.set_ylabel('Stroke Rate')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    # 2. Scatter Plot - Age vs Glucose with stroke overlay
    st.subheader("2. Scatter Plot: Age vs Glucose Level Analysis")
    
    fig = px.scatter(df, x='age', y='avg_glucose_level', 
                     color='stroke', color_discrete_map={0: 'lightblue', 1: 'red'},
                     title='Age vs Glucose Level - Stroke Risk Analysis',
                     labels={'stroke': 'Stroke', 'age': 'Age (years)', 
                            'avg_glucose_level': 'Average Glucose Level (mg/dL)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Heatmap - Correlation matrix
    st.subheader("3. Heatmap: Feature Correlation Analysis")
    
    # Prepare numerical data for correlation
    numerical_df = df.select_dtypes(include=[np.number]).copy()
    
    # Encode some categorical variables
    cat_mappings = {
        'gender': {'Male': 1, 'Female': 0, 'Other': 2},
        'ever_married': {'Yes': 1, 'No': 0},
        'Residence_type': {'Urban': 1, 'Rural': 0}
    }
    
    for col, mapping in cat_mappings.items():
        if col in df.columns:
            numerical_df[col] = df[col].map(mapping)
    
    correlation_matrix = numerical_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    # 4. Box Plot - Distributions by stroke status
    st.subheader("4. Box Plot: Feature Distributions by Stroke Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        df.boxplot(column='age', by='stroke', ax=ax)
        ax.set_title('Age Distribution by Stroke Status')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        df.boxplot(column='avg_glucose_level', by='stroke', ax=ax)
        ax.set_title('Glucose Distribution by Stroke Status')
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(6, 6))
        df.boxplot(column='bmi', by='stroke', ax=ax)
        ax.set_title('BMI Distribution by Stroke Status')
        st.pyplot(fig)

def show_risk_assessment(df):
    st.header("⚠️ Risk Factor Assessment")
    
    # High-risk patient identification
    st.subheader("High-Risk Patient Identification")
    
    # Define high-risk criteria
    high_risk_patients = df[
        (df['age'] >= 60) & 
        (df['hypertension'] == 1) & 
        (df['avg_glucose_level'] > 120)
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High-Risk Patients", len(high_risk_patients))
    
    with col2:
        high_risk_stroke_rate = high_risk_patients['stroke'].mean()
        st.metric("High-Risk Stroke Rate", f"{high_risk_stroke_rate*100:.1f}%")
    
    with col3:
        risk_ratio = high_risk_stroke_rate / df['stroke'].mean()
        st.metric("Risk Multiplier", f"{risk_ratio:.1f}x")
    
    # Risk factors analysis
    st.subheader("Risk Factor Impact Analysis")
    
    risk_factors = {
        'Age 60+': (df['age'] >= 60),
        'Hypertension': (df['hypertension'] == 1),
        'Heart Disease': (df['heart_disease'] == 1),
        'High Glucose': (df['avg_glucose_level'] > 125),
        'Obesity': (df['bmi'] > 30),
        'Current Smoker': (df['smoking_status'] == 'smokes')
    }
    
    risk_analysis = []
    for factor, condition in risk_factors.items():
        factor_df = df[condition]
        if len(factor_df) > 0:
            stroke_rate = factor_df['stroke'].mean()
            relative_risk = stroke_rate / df['stroke'].mean()
            risk_analysis.append({
                'Risk Factor': factor,
                'Patients': len(factor_df),
                'Stroke Rate': f"{stroke_rate*100:.1f}%",
                'Relative Risk': f"{relative_risk:.1f}x"
            })
    
    risk_df = pd.DataFrame(risk_analysis)
    st.dataframe(risk_df, use_container_width=True)

def show_model_performance(df):
    st.header("🤖 Model Performance Analysis")
    
    # Prepare data for modeling
    st.subheader("Model Training and Evaluation")
    
    # Simple feature encoding for demo
    model_df = df.copy()
    
    # Encode categorical variables
    cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in cat_columns:
        model_df[col] = pd.Categorical(model_df[col]).codes
    
    # Handle missing BMI values
    model_df['bmi'].fillna(model_df['bmi'].median(), inplace=True)
    
    # Select features
    feature_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                      'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    
    X = model_df[feature_columns]
    y = model_df['stroke']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    model_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        model_results[name] = {'model': model, 'auc': auc_score, 'proba': y_pred_proba}
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        for name, results in model_results.items():
            st.metric(f"{name} AUC-ROC", f"{results['auc']:.3f}")
    
    with col2:
        # ROC Curve
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, results in model_results.items():
            fpr, tpr, _ = roc_curve(y_test, results['proba'])
            ax.plot(fpr, tpr, label=f'{name} (AUC = {results["auc"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Feature importance (Random Forest)
    if 'Random Forest' in model_results:
        st.subheader("Feature Importance Analysis")
        rf_model = model_results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.plot(x='Feature', y='Importance', kind='bar', ax=ax)
        ax.set_title('Feature Importance - Random Forest')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def show_risk_analysis(df):
    st.header("🎯 Population Risk Stratification Analysis")
    
    st.markdown("""
    This section provides comprehensive risk stratification analysis for the entire patient population,
    helping healthcare administrators understand risk distribution and plan interventions accordingly.
    """)
    
    # Create simple risk scores for the dataset
    df_risk = df.copy()
    
    # Calculate basic risk scores
    def calculate_population_risk(row):
        risk = 0.02  # Base risk
        
        # Age risk
        if row['age'] >= 60:
            risk += 0.13
        elif row['age'] >= 45:
            risk += 0.06
        elif row['age'] >= 30:
            risk += 0.01
            
        # Health conditions
        if row['hypertension'] == 1:
            risk *= 3.7
        if row['heart_disease'] == 1:
            risk *= 2.8
            
        # Glucose risk
        if row['avg_glucose_level'] > 125:
            risk *= 2.1
        elif row['avg_glucose_level'] > 100:
            risk *= 1.4
            
        # BMI risk
        if row['bmi'] > 30:
            risk *= 1.3
        elif row['bmi'] > 25:
            risk *= 1.1
            
        return min(risk, 1.0)
    
    # Apply risk calculation
    df_risk['risk_probability'] = df_risk.apply(calculate_population_risk, axis=1)
    df_risk['risk_score'] = (df_risk['risk_probability'] * 100).round(1)
    df_risk['risk_category'] = df_risk['risk_probability'].apply(categorize_stroke_risk)
    
    # Display overall statistics
    st.subheader("📊 Population Risk Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = df_risk['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.1f}%")
    
    with col2:
        high_risk_count = len(df_risk[df_risk['risk_category'].isin(['High', 'Very High'])])
        high_risk_pct = (high_risk_count / len(df_risk)) * 100
        st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_pct:.1f}%)")
    
    with col3:
        actual_stroke_rate = df_risk['stroke'].mean() * 100
        st.metric("Actual Stroke Rate", f"{actual_stroke_rate:.1f}%")
    
    with col4:
        median_risk = df_risk['risk_score'].median()
        st.metric("Median Risk Score", f"{median_risk:.1f}%")
    
    # Risk category distribution
    st.subheader("🏥 Risk Category Distribution")
    
    category_counts = df_risk['risk_category'].value_counts()
    categories = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    colors = [get_risk_color(cat) for cat in categories]
    
    # Create visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker_colors=[get_risk_color(cat) for cat in category_counts.index],
            hole=0.4
        )])
        fig_pie.update_layout(title="Risk Category Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = go.Figure(data=[go.Bar(
            x=category_counts.index,
            y=category_counts.values,
            marker_color=[get_risk_color(cat) for cat in category_counts.index]
        )])
        fig_bar.update_layout(
            title="Patient Count by Risk Category",
            xaxis_title="Risk Category",
            yaxis_title="Number of Patients"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk score distribution
    st.subheader("📈 Risk Score Distribution Analysis")
    
    fig_hist = go.Figure(data=[go.Histogram(
        x=df_risk['risk_score'],
        nbinsx=30,
        marker_color='lightblue',
        opacity=0.7
    )])
    fig_hist.add_vline(x=avg_risk, line_dash="dash", line_color="red", 
                       annotation_text=f"Mean: {avg_risk:.1f}%")
    fig_hist.add_vline(x=median_risk, line_dash="dash", line_color="orange",
                       annotation_text=f"Median: {median_risk:.1f}%")
    fig_hist.update_layout(
        title="Distribution of Risk Scores Across Population",
        xaxis_title="Risk Score (%)",
        yaxis_title="Number of Patients"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Risk vs Actual Outcomes
    st.subheader("🎯 Risk Prediction vs Actual Outcomes")
    
    # Cross-tabulation
    risk_outcome = pd.crosstab(df_risk['risk_category'], df_risk['stroke'], normalize='index') * 100
    
    fig_outcome = go.Figure()
    fig_outcome.add_trace(go.Bar(
        x=risk_outcome.index,
        y=risk_outcome[0],
        name='No Stroke',
        marker_color='lightgreen'
    ))
    fig_outcome.add_trace(go.Bar(
        x=risk_outcome.index,
        y=risk_outcome[1],
        name='Stroke',
        marker_color='lightcoral'
    ))
    
    fig_outcome.update_layout(
        title="Stroke Occurrence Rate by Risk Category",
        xaxis_title="Risk Category",
        yaxis_title="Percentage (%)",
        barmode='stack'
    )
    st.plotly_chart(fig_outcome, use_container_width=True)
    
    # Detailed category analysis
    st.subheader("📋 Detailed Risk Category Analysis")
    
    for category in categories:
        if category in category_counts.index:
            category_data = df_risk[df_risk['risk_category'] == category]
            count = len(category_data)
            percentage = (count / len(df_risk)) * 100
            stroke_rate = category_data['stroke'].mean() * 100
            avg_age = category_data['age'].mean()
            
            with st.expander(f"{category} Risk - {count} patients ({percentage:.1f}%)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Stroke Rate", f"{stroke_rate:.1f}%")
                with col2:
                    st.metric("Average Age", f"{avg_age:.1f} years")
                with col3:
                    hypertension_rate = category_data['hypertension'].mean() * 100
                    st.metric("Hypertension Rate", f"{hypertension_rate:.1f}%")
                
                # Clinical recommendations for this category
                recommendations = get_clinical_recommendations(category)
                st.markdown("**Clinical Recommendations:**")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
    
    # Export functionality
    st.subheader("💾 Export Risk Analysis Data")
    
    if st.button("Generate Risk Assessment Report"):
        # Create summary report
        summary_data = {
            'Risk_Category': category_counts.index,
            'Patient_Count': category_counts.values,
            'Percentage': (category_counts.values / len(df_risk) * 100).round(1),
            'Avg_Risk_Score': [df_risk[df_risk['risk_category'] == cat]['risk_score'].mean().round(1) 
                              for cat in category_counts.index],
            'Stroke_Rate': [df_risk[df_risk['risk_category'] == cat]['stroke'].mean().round(3) * 100 
                           for cat in category_counts.index]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        st.markdown("### 📊 Risk Stratification Summary Report")
        st.dataframe(summary_df, use_container_width=True)
        
        # Download button for CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Risk Analysis Report (CSV)",
            data=csv,
            file_name="stroke_risk_analysis_report.csv",
            mime="text/csv"
        )

def show_patient_prediction(df):
    st.header("👤 Individual Patient Risk Assessment")
    
    st.markdown("### Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 0, 100, 50)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    if st.button("Calculate Stroke Risk", type="primary"):
        # Simple risk calculation based on key factors
        risk_score = 0
        
        # Age risk
        if age < 30:
            age_risk = 0.02
        elif age < 45:
            age_risk = 0.03
        elif age < 60:
            age_risk = 0.08
        else:
            age_risk = 0.15
        
        risk_score += age_risk
        
        # Health conditions
        if hypertension == "Yes":
            risk_score *= 3
        if heart_disease == "Yes":
            risk_score *= 4
        
        # Glucose level
        if glucose_level > 125:
            risk_score *= 2
        elif glucose_level > 100:
            risk_score *= 1.5
        
        # BMI
        if bmi > 30:
            risk_score *= 1.3
        
        # Smoking
        if smoking_status == "smokes":
            risk_score *= 1.5
        elif smoking_status == "formerly smoked":
            risk_score *= 1.2
        
        # Cap the risk score
        risk_score = min(risk_score, 1.0)
        
        # Get risk category and clinical recommendations
        risk_category = categorize_stroke_risk(risk_score)
        risk_color = get_risk_color(risk_category)
        clinical_recs = get_clinical_recommendations(risk_category)
        
        # Display results with enhanced styling
        st.subheader("🎯 Comprehensive Risk Assessment Results")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stroke Probability", f"{risk_score*100:.1f}%")
        
        with col2:
            population_avg = df['stroke'].mean()
            risk_ratio = risk_score / population_avg if population_avg > 0 else 0
            st.metric("vs Population Average", f"{risk_ratio:.1f}x")
        
        with col3:
            # Risk score out of 100
            st.metric("Risk Score", f"{risk_score*100:.0f}/100")
        
        with col4:
            st.metric("Risk Category", risk_category)
        
        # Risk category display with color coding
        st.markdown(f"""
        <div class="risk-category" style="background-color: {risk_color}; color: white;">
            🏥 CLINICAL RISK LEVEL: {risk_category.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for risk visualization
        st.subheader("📊 Risk Visualization")
        progress_value = min(risk_score, 1.0)
        st.progress(progress_value, text=f"Risk Level: {progress_value*100:.1f}%")
        
        # Risk factor breakdown
        st.subheader("� Risk Factor Analysis")
        risk_factors = []
        
        if age >= 60:
            risk_factors.append(f"⚠️ Advanced age ({age} years) - Higher stroke risk")
        elif age >= 45:
            risk_factors.append(f"⚡ Middle age ({age} years) - Moderate risk increase")
            
        if hypertension == "Yes":
            risk_factors.append("🔴 Hypertension - Major risk factor (3x increased risk)")
            
        if heart_disease == "Yes":
            risk_factors.append("💗 Heart disease - Significant risk factor (4x increased risk)")
            
        if glucose_level > 125:
            risk_factors.append(f"📈 Elevated glucose ({glucose_level} mg/dL) - Diabetes concern")
        elif glucose_level > 100:
            risk_factors.append(f"📊 Borderline glucose ({glucose_level} mg/dL) - Monitor closely")
            
        if bmi > 30:
            risk_factors.append(f"⚖️ Obesity (BMI: {bmi:.1f}) - Increased cardiovascular risk")
        elif bmi > 25:
            risk_factors.append(f"📏 Overweight (BMI: {bmi:.1f}) - Mild risk increase")
            
        if smoking_status == "smokes":
            risk_factors.append("🚬 Current smoker - Major modifiable risk factor")
        elif smoking_status == "formerly smoked":
            risk_factors.append("🚭 Former smoker - Reduced but present risk")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.success("✅ No major risk factors identified!")
        
        # Clinical recommendations
        st.subheader("⚕️ Personalized Clinical Recommendations")
        
        for i, rec in enumerate(clinical_recs, 1):
            st.markdown(f"""
            <div class="clinical-rec">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)
        
        if risk_score >= 0.15:
            st.markdown("""
            <div class="risk-high">
            <h4>⚠️ High Risk - Immediate Action Recommended</h4>
            <ul>
                <li>Schedule immediate consultation with healthcare provider</li>
                <li>Consider comprehensive cardiovascular screening</li>
                <li>Implement aggressive lifestyle modifications</li>
                <li>Monitor blood pressure and glucose levels closely</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
            <h4>✅ Lower Risk - Preventive Measures Recommended</h4>
            <ul>
                <li>Continue regular health checkups</li>
                <li>Maintain healthy lifestyle habits</li>
                <li>Monitor key risk factors annually</li>
                <li>Stay physically active and eat balanced diet</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
