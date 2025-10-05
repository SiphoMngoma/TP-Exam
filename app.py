# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import io
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Retirement Fund Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class RetirementFundAnalyzer:
    def __init__(self):
        self.ppd_plan = None
        self.pension_coverage = None
        self.ppd_cleaned = None
        self.pension_cleaned = None
        self.model = None
        self.forecast = None
        self.data_loaded = False
        
    def load_uploaded_data(self, ppd_file, coverage_file):
        """Load datasets from uploaded files"""
        try:
            if ppd_file is not None:
                self.ppd_plan = pd.read_csv(ppd_file)
                st.sidebar.success(f"✅ PPD data loaded: {self.ppd_plan.shape}")
            else:
                st.sidebar.error("❌ Please upload PPD_PlanLevel.csv")
                return False
                
            if coverage_file is not None:
                self.pension_coverage = pd.read_csv(coverage_file)
                st.sidebar.success(f"✅ Coverage data loaded: {self.pension_coverage.shape}")
            else:
                st.sidebar.error("❌ Please upload pension_coverage.csv")
                return False
                
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def load_local_data(self):
        """Load datasets from local files"""
        try:
            self.ppd_plan = pd.read_csv('PPD_PlanLevel.csv')
            self.pension_coverage = pd.read_csv('pension_coverage.csv')
            return True
        except Exception as e:
            st.error(f"Error loading local data: {e}")
            return False
    
    def clean_data(self):
        """Clean and preprocess the datasets"""
        try:
            # Clean PPD data
            ppd_columns_to_keep = [
                'ppd_id', 'PlanName', 'fy', 'StateName', 'PlanType', 'EmployeeTypeCovered',
                'ActAssets_GASB', 'ActLiabilities_GASB', 'ActFundedRatio_GASB', 'UAAL_GASB',
                'payroll', 'RequiredContribution', 'PercentReqContPaid', 'TotalPensionLiability',
                'NetPensionLiability', 'InvestmentReturn_1yr', 'InvestmentReturn_5yr', 'InvestmentReturn_10yr',
                'contrib_EE_regular', 'contrib_ER_regular', 'contrib_tot', 'income_net', 'expense_net',
                'actives_tot', 'beneficiaries_tot', 'benefits_tot', 'ActiveSalary_avg', 'BeneficiaryBenefit_avg'
            ]
            
            # Only keep columns that exist in the dataset
            available_columns = [col for col in ppd_columns_to_keep if col in self.ppd_plan.columns]
            self.ppd_cleaned = self.ppd_plan[available_columns].copy()
            
            # Handle missing values for numerical columns
            numerical_cols_ppd = ['ActAssets_GASB', 'ActLiabilities_GASB', 'ActFundedRatio_GASB', 
                                 'payroll', 'InvestmentReturn_1yr', 'InvestmentReturn_5yr', 'InvestmentReturn_10yr',
                                 'contrib_tot', 'income_net', 'actives_tot', 'beneficiaries_tot']
            
            numerical_cols_ppd = [col for col in numerical_cols_ppd if col in self.ppd_cleaned.columns]
            
            for col in numerical_cols_ppd:
                self.ppd_cleaned[col].fillna(self.ppd_cleaned[col].median(), inplace=True)
            
            # Drop rows with critical missing values
            critical_cols = ['ActFundedRatio_GASB', 'fy', 'StateName']
            critical_cols = [col for col in critical_cols if col in self.ppd_cleaned.columns]
            self.ppd_cleaned = self.ppd_cleaned.dropna(subset=critical_cols)
            
            # Clean pension coverage data
            self.pension_cleaned = self.pension_coverage.copy()
            if 'year' in self.pension_cleaned.columns:
                self.pension_cleaned['year'] = pd.to_datetime(self.pension_cleaned['year'], format='%Y', errors='coerce')
                self.pension_cleaned = self.pension_cleaned.dropna(subset=['year'])
                self.pension_cleaned.fillna(method='ffill', inplace=True)
                self.pension_cleaned.fillna(method='bfill', inplace=True)
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error cleaning data: {e}")
            return False
    
    def build_prophet_model(self):
        """Build and train Prophet forecasting model"""
        try:
            if 'all' not in self.pension_cleaned.columns or 'year' not in self.pension_cleaned.columns:
                st.warning("Pension coverage data doesn't contain required columns for forecasting")
                return False
                
            # Prepare data for Prophet
            prophet_data = self.pension_cleaned[['year', 'all']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 2:
                st.warning("Not enough data for forecasting")
                return False
            
            # Initialize and train model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            self.model.fit(prophet_data)
            
            # Create future predictions
            future = self.model.make_future_dataframe(periods=5, freq='Y')
            self.forecast = self.model.predict(future)
            
            return True
        except Exception as e:
            st.error(f"Error building Prophet model: {e}")
            return False
    
    def calculate_metrics(self):
        """Calculate model evaluation metrics"""
        try:
            if self.forecast is None or self.pension_cleaned is None:
                return {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0, 'accuracy': 0}
                
            historical_forecast = self.forecast[self.forecast['ds'].isin(self.pension_cleaned['year'])]
            actual_values = self.pension_cleaned['all'].values
            predicted_values = historical_forecast['yhat'].values
            
            if len(actual_values) == 0 or len(predicted_values) == 0:
                return {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0, 'accuracy': 0}
            
            mae = mean_absolute_error(actual_values, predicted_values)
            mse = mean_squared_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_values, predicted_values)
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            accuracy = max(0, 100 - mape)
            
            return {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'accuracy': accuracy
            }
        except:
            return {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0, 'accuracy': 0}
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        if not self.data_loaded:
            return None
            
        try:
            # Financial Health Analysis
            financial_metrics = self.ppd_cleaned.groupby('StateName').agg({
                'ActFundedRatio_GASB': 'mean',
                'UAAL_GASB': 'mean',
                'ActAssets_GASB': 'sum',
                'TotalPensionLiability': 'sum',
                'payroll': 'mean'
            }).round(2)
            
            financial_metrics['Funding_Status'] = pd.cut(
                financial_metrics['ActFundedRatio_GASB'],
                bins=[0, 60, 80, 100, 200],
                labels=['Critical', 'At Risk', 'Healthy', 'Well-Funded']
            )
            
            # Risk Analysis
            np.random.seed(42)
            self.ppd_cleaned['withdrawal_risk'] = np.where(
                (self.ppd_cleaned['ActFundedRatio_GASB'] < 70) & 
                (self.ppd_cleaned['InvestmentReturn_1yr'] < 0) &
                (self.ppd_cleaned['UAAL_GASB'] > self.ppd_cleaned['payroll']),
                1, 0
            )
            
            # Two-Pot System Simulation
            self.ppd_cleaned['accessible_pot_ratio'] = np.random.uniform(0.1, 0.3, len(self.ppd_cleaned))
            self.ppd_cleaned['locked_pot_ratio'] = 1 - self.ppd_cleaned['accessible_pot_ratio']
            self.ppd_cleaned['accessible_contrib'] = self.ppd_cleaned['contrib_tot'] * self.ppd_cleaned['accessible_pot_ratio']
            self.ppd_cleaned['locked_contrib'] = self.ppd_cleaned['contrib_tot'] * self.ppd_cleaned['locked_pot_ratio']
            
            return financial_metrics
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def main():
    st.markdown('<h1 class="main-header">🏦 Retirement Fund Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Analysis of Two-Pot Retirement System")
    
    # Initialize analyzer
    analyzer = RetirementFundAnalyzer()
    
    # File Upload Section in Sidebar
    st.sidebar.title("📁 Data Upload")
    st.sidebar.markdown("""
    <div class="upload-section">
        <h4>Upload Your Datasets</h4>
        <p>Please upload both CSV files to proceed with analysis:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploaders
    ppd_file = st.sidebar.file_uploader(
        "Upload PPD_PlanLevel.csv", 
        type=['csv'],
        help="Upload the PPD Plan Level dataset"
    )
    
    coverage_file = st.sidebar.file_uploader(
        "Upload pension_coverage.csv", 
        type=['csv'],
        help="Upload the Pension Coverage dataset"
    )
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source:",
        ["Upload Files", "Use Local Files"],
        help="Choose to upload files or use local files"
    )
    
    data_loaded = False
    
    if data_source == "Upload Files":
        if ppd_file is not None and coverage_file is not None:
            if st.sidebar.button("🚀 Load and Analyze Data", type="primary"):
                with st.spinner("Loading and analyzing uploaded data..."):
                    if analyzer.load_uploaded_data(ppd_file, coverage_file):
                        if analyzer.clean_data():
                            data_loaded = True
                            st.sidebar.success("✅ Data loaded and cleaned successfully!")
        else:
            st.sidebar.warning("⚠️ Please upload both CSV files")
    else:
        if st.sidebar.button("🚀 Load Local Data", type="primary"):
            with st.spinner("Loading and analyzing local data..."):
                if analyzer.load_local_data():
                    if analyzer.clean_data():
                        data_loaded = True
                        st.sidebar.success("✅ Local data loaded successfully!")
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    sections = [
        "Overview",
        "Data Exploration", 
        "Financial Health",
        "Forecasting",
        "Risk Analysis",
        "Two-Pot System",
        "Demographic Analysis"
    ]
    
    if not data_loaded:
        # Show data upload instructions on main page
        st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #f8f9fa; border-radius: 10px;">
            <h2>📊 Welcome to Retirement Fund Analysis Dashboard</h2>
            <p style="font-size: 1.2em;">To get started, please upload your datasets using the sidebar.</p>
            <div style="margin: 30px 0;">
                <h4>Required Files:</h4>
                <ul style="text-align: left; display: inline-block;">
                    <li><strong>PPD_PlanLevel.csv</strong> - Pension plan level data</li>
                    <li><strong>pension_coverage.csv</strong> - Pension coverage statistics</li>
                </ul>
            </div>
            <p>Choose your data source in the sidebar and click the load button to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample of what the dashboard will display
        st.markdown("---")
        st.subheader("📋 What you'll get:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Financial Analysis**
            - Funded ratios
            - Asset analysis
            - Risk assessment
            """)
            
        with col2:
            st.markdown("""
            **Forecasting**
            - Coverage predictions
            - Trend analysis
            - Model accuracy
            """)
            
        with col3:
            st.markdown("""
            **Demographic Insights**
            - Coverage disparities
            - Gender analysis
            - Education impact
            """)
        
        return
    
    # If data is loaded, proceed with analysis
    selected_section = st.sidebar.radio("Go to", sections)
    
    # Run analysis
    with st.spinner("Running analysis..."):
        financial_metrics = analyzer.run_analysis()
        
        # Only build Prophet model if we have the required data
        prophet_success = False
        if 'all' in analyzer.pension_cleaned.columns and 'year' in analyzer.pension_cleaned.columns:
            prophet_success = analyzer.build_prophet_model()
        
        metrics = analyzer.calculate_metrics() if prophet_success else {}
    
    # OVERVIEW SECTION
    if selected_section == "Overview":
        st.markdown('<h2 class="section-header">📊 Executive Summary</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Pension Plans Analyzed",
                f"{len(analyzer.ppd_cleaned):,}",
                "Plans"
            )
        
        with col2:
            avg_funding = analyzer.ppd_cleaned['ActFundedRatio_GASB'].mean()
            st.metric(
                "Average Funded Ratio",
                f"{avg_funding:.1f}%",
                f"{'Healthy' if avg_funding > 80 else 'At Risk'}"
            )
        
        with col3:
            withdrawal_risk_rate = analyzer.ppd_cleaned['withdrawal_risk'].mean() * 100
            st.metric(
                "High Withdrawal Risk Funds",
                f"{withdrawal_risk_rate:.1f}%",
                "Of total plans"
            )
        
        with col4:
            if 'all' in analyzer.pension_cleaned.columns:
                current_coverage = analyzer.pension_cleaned['all'].iloc[-1]
                st.metric(
                    "Current Pension Coverage",
                    f"{current_coverage:.1f}%",
                    "Overall population"
                )
            else:
                st.metric("Current Pension Coverage", "N/A", "Data not available")
        
        # Rest of your existing Overview section code remains the same...
        # [Keep all the existing Overview section code here]
        
    # DATA EXPLORATION SECTION
    elif selected_section == "Data Exploration":
        st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
        
        # Show data source info
        st.info(f"📁 **Data Source:** {data_source} | 📊 **PPD Data:** {analyzer.ppd_cleaned.shape} | 📈 **Coverage Data:** {analyzer.pension_cleaned.shape}")
        
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Statistical Summary", "Data Quality"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PPD PlanLevel Data")
                st.dataframe(analyzer.ppd_cleaned.head(10), use_container_width=True)
                st.write(f"**Shape:** {analyzer.ppd_cleaned.shape}")
                st.write(f"**Columns:** {len(analyzer.ppd_cleaned.columns)}")
            
            with col2:
                st.subheader("Pension Coverage Data")
                st.dataframe(analyzer.pension_cleaned.head(10), use_container_width=True)
                st.write(f"**Shape:** {analyzer.pension_cleaned.shape}")
                st.write(f"**Columns:** {len(analyzer.pension_cleaned.columns)}")
        
        # Rest of your existing Data Exploration section code remains the same...
        # [Keep all the existing Data Exploration section code here]

    # [Keep all other sections exactly as they are in your original code]
    # FINANCIAL HEALTH, FORECASTING, RISK ANALYSIS, TWO-POT SYSTEM, DEMOGRAPHIC ANALYSIS sections remain unchanged

    # Footer
    st.markdown("---")
    st.markdown(
        "**Retirement Fund Analysis Dashboard** | "
        "Built with Streamlit | "
        f"Data Source: {data_source} | "
        f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()