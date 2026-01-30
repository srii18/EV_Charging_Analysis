# EV Charging Infrastructure Strategic Analysis

A comprehensive data science project for electric vehicle charging infrastructure analysis in India, combining data cleaning, exploratory analysis, machine learning forecasting, and strategic recommendations through an interactive dashboard.

## 🚀 Project Overview

This project analyzes the gap between EV demand and charging infrastructure across Indian states, providing data-driven insights for infrastructure investment and planning decisions.

### Key Components
- **Data Cleaning & EDA**: Comprehensive data preprocessing and exploratory analysis
- **Time Series Forecasting**: XGBoost-based ML models for EV sales prediction
- **Gap Analysis**: Strategic scoring system for infrastructure needs
- **Interactive Dashboard**: Streamlit-based visualization and recommendation system

## 📊 Key Insights

### High Priority States (Immediate Action Required)
1. **Uttar Pradesh**: 467,843 EVs, only 3 charging stations (0.006 stations per 1000 EVs)
2. **Maharashtra**: 348,151 EVs, only 4 charging stations (0.011 stations per 1000 EVs)
3. **Karnataka**: 261,095 EVs, only 4 charging stations (0.015 stations per 1000 EVs)
4. **Rajasthan**: 179,351 EVs, 0 charging stations
5. **Gujarat**: 163,429 EVs, 0 charging stations

### ML Model Performance
- **Algorithm**: XGBoost Regressor with 200 estimators
- **Test R² Score**: 0.9035 (Excellent predictive power)
- **MAPE**: 28.40% on test set
- **Key Features**: 3-month moving average (48.1%), 6-month moving average (37.5%), 1-month lag (9.2%)

### Market Growth Predictions
- **Top Growth States**: Chhattisgarh (+15.4%), Gujarat (+12.4%), Bihar (+8.7%)
- **Forecast Period**: 3-12 month projections
- **Infrastructure Impact**: States with no infrastructure show highest growth potential

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Internet connection (for initial data download)

### Setup Steps

1. **Clone/Download the repository**
2. **Navigate to project directory**:
   ```bash
   cd EV_Charging_Analysis
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis notebooks** (optional):
   ```bash
   jupyter notebook
   # Open and run: data_cleaning_EDA.ipynb, time_series_implementation.ipynb, ML_model.ipynb
   ```

5. **Launch the dashboard**:
   ```bash
   streamlit run strategic_recommendation_dashboard.py
   ```

### Dependencies
```
pandas==2.1.4          # Data manipulation
numpy==1.24.3           # Numerical operations
matplotlib==3.8.2       # Static visualizations
seaborn==0.13.0         # Statistical plots
plotly==5.17.0          # Interactive visualizations
streamlit==1.29.0       # Dashboard framework
scikit-learn==1.3.2     # ML preprocessing
xgboost==1.7.0          # ML modeling
kagglehub==0.2.0        # Data download
jupyter==1.0.0          # Notebook support
```

## 📁 Project Structure

```
EV_Charging_Analysis/
├── Analysis Notebooks
│   ├── data_cleaning_EDA.ipynb           # Data cleaning and exploratory analysis
│   ├── time_series_implementation.ipynb  # Time series forecasting implementation
│   └── ML_model.ipynb                    # Machine learning model development
├── Dashboard & Applications
│   ├── strategic_recommendation_dashboard.py  # Main Streamlit dashboard
│   └── app/                                 # Additional application modules
├── Data Directory
│   ├── data/
│   │   ├── raw/                            # Raw datasets
│   │   ├── processed/                      # Cleaned and processed data
│   │   └── india_states.geojson           # Geographic data
│   └── detailed_gap_analysis.csv           # Generated analysis results
├── Configuration
│   ├── config/                             # Configuration files
│   ├── utils/                              # Utility functions
│   └── models/                             # Saved ML models
├── Project Files
│   ├── requirements.txt                     # Python dependencies
│   ├── README.md                           # This file
│   └── .gitignore                          # Git ignore rules
└── Testing
    └── tests/                              # Test files
```

## 🔄 Analysis Workflow

### Phase 1: Data Cleaning & EDA (`data_cleaning_EDA.ipynb`)
1. **Data Ingestion**: Download EV charging stations and sales data from Kaggle
2. **Data Cleaning**: 
   - Remove duplicates (66 duplicate station records removed)
   - Standardize state names (fixed Kerala variations: 'Keraka', 'Keral', 'Lerala')
   - Handle missing values and data type conversions
3. **Exploratory Analysis**:
   - Temporal trends (2014-2024 EV sales data)
   - Geographic distribution analysis
   - Vehicle category breakdown (2-wheelers, 3-wheelers, 4-wheelers, buses)
4. **Gap Analysis**: Calculate stations per 1000 EVs and charging gap scores

### Phase 2: Time Series Implementation (`time_series_implementation.ipynb`)
1. **Feature Engineering**:
   - Lag features (1, 3, 6, 12 months)
   - Rolling statistics (moving averages, standard deviation)
   - Growth rates (MoM, YoY, 6-month)
   - Seasonality features (month/quarter encoding)
2. **Model Training**:
   - XGBoost Regressor with optimized hyperparameters
   - Time-based train/test split (last 12 months for testing)
3. **Performance Evaluation**:
   - R² Score: 0.9035 (excellent predictive power)
   - MAPE: 28.40% on test set
   - Feature importance analysis

### Phase 3: ML Model Development (`ML_model.ipynb`)
1. **Enhanced Modeling**: Scenario-based growth simulation
2. **Infrastructure Impact Analysis**: Different growth scenarios based on current infrastructure levels
3. **Adoption Spike Potential**: Calculate growth potential for meaningful market sizes
4. **Investment Recommendations**: Quantified infrastructure requirements

### Phase 4: Strategic Dashboard (`strategic_recommendation_dashboard.py`)
1. **Interactive Visualizations**: Real-time filtering and analysis
2. **Geographic Mapping**: Current stations, recommended locations, heat maps
3. **Strategic Recommendations**: Priority-based investment guidance
4. **Executive Summary**: High-level insights for decision makers

## 🎯 How to Use

### Running the Analysis
1. **Complete Analysis Pipeline**:
   ```bash
   # Step 1: Data cleaning and EDA
   jupyter notebook data_cleaning_EDA.ipynb
   
   # Step 2: Time series forecasting
   jupyter notebook time_series_implementation.ipynb
   
   # Step 3: ML model development
   jupyter notebook ML_model.ipynb
   
   # Step 4: Launch interactive dashboard
   streamlit run strategic_recommendation_dashboard.py
   ```

2. **Quick Start - Dashboard Only**:
   ```bash
   streamlit run strategic_recommendation_dashboard.py
   ```

### Dashboard Features
- **Time Period Selection**: Current month, 3/6/12 month projections
- **Growth Filtering**: Filter by growth categories (Declining, Stable, Growing, High Growth)
- **Priority Filtering**: Filter by priority levels (High, Medium, Low)
- **Interactive Maps**: Current stations, recommended locations, heat analysis
- **Strategic Insights**: Investment priorities and executive summary

## 📈 Data Sources

### Primary Datasets
1. **EV Charging Stations**: 
   - Source: Kaggle - "EV Charging Stations in India Simplified 2025"
   - Records: 855 stations (cleaned to 786 unique stations)
   - Coverage: 14 states with charging infrastructure
   - Features: Location, power capacity, connector types, operators

2. **EV Sales by State**:
   - Source: Kaggle - "Electric Vehicle Sales by State in India"
   - Records: 96,845 sales records (2014-2024)
   - Coverage: 34 states + union territories
   - Features: Monthly sales by vehicle category and type

### Data Processing Pipeline
- **Cleaning**: Removed duplicates, standardized state names, handled missing values
- **Validation**: Cross-referenced geographic boundaries and state names
- **Aggregation**: Combined datasets for comprehensive gap analysis
- **Feature Engineering**: Created time-based features for ML modeling

## 🔧 Technical Details

### Machine Learning Model Architecture
```python
# XGBoost Regressor Configuration
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)
```

### Feature Engineering Pipeline
- **Lag Features**: sales_lag_1m, sales_lag_3m, sales_lag_6m, sales_lag_12m
- **Rolling Statistics**: sales_ma_3m, sales_ma_6m, sales_ma_12m, sales_std_6m
- **Growth Rates**: growth_mom, growth_yoy, growth_6m, growth_acceleration
- **Trend Indicators**: months_in_market, cumulative_sales, market_stage_numeric
- **Seasonality**: Month and quarter one-hot encoded features

### Strategic Scoring System
```python
# Gap Analysis Scoring
Charging_Gap_Score = (EV_Sales_Normalized × 0.6) + 
                    ((1 - Station_Density_Normalized) × 0.4)

# Investment Priority Scoring
Investment_Priority_Score = (Infrastructure_Need × 0.5) + 
                           (Projected_Sales_Normalized × 0.3) + 
                           (Growth_Rate_Normalized × 0.2)
```

## 🎨 Visualizations & Outputs

### Analysis Notebooks Generate:
- **Temporal trend plots**: EV sales evolution (2014-2024)
- **Geographic distributions**: State-wise charging station coverage
- **Gap analysis charts**: Infrastructure demand vs supply
- **Feature importance plots**: ML model interpretability
- **Forecast visualizations**: Predicted vs actual sales

### Interactive Dashboard Provides:
- **Real-time KPIs**: Total EV sales, charging stations, high-priority states
- **Strategic maps**: Current infrastructure and recommended locations
- **Growth analysis**: Current vs projected sales comparisons
- **Investment recommendations**: Priority-based actionable insights
- **Executive summary**: High-level strategic overview

## 📋 Key Findings & Recommendations

### Critical Infrastructure Gaps
- **Uttar Pradesh**: Highest demand (467,843 EVs) with minimal infrastructure (3 stations)
- **Maharashtra**: Second-largest market (348,151 EVs) severely underserved (4 stations)
- **Karnataka**: Tech hub with 261,095 EVs but only 4 charging stations
- **Zero Infrastructure States**: Rajasthan, Gujarat, and 20+ other states with no charging stations

### Strategic Recommendations
1. **Immediate Priority**: Focus on top 5 high-gap states (Uttar Pradesh, Maharashtra, Karnataka, Rajasthan, Gujarat)
2. **Infrastructure Expansion**: Minimum 240 new charging stations required to meet current demand
3. **Growth Markets**: Invest in states with high growth potential but low current infrastructure
4. **Monitoring Framework**: Track monthly adoption trends and adjust infrastructure plans

### Market Opportunities
- **Underserved Markets**: 23 states with <1 station per 1000 EVs
- **Growth Leaders**: Chhattisgarh, Gujarat, Bihar showing >8% projected growth
- **Infrastructure Impact**: States with no infrastructure show highest adoption spike potential

## 🤝 Contributing & Future Enhancements

### Potential Improvements
- **Real-time Data Integration**: API connections for live charging station data
- **Advanced ML Models**: LSTM/Transformer models for improved time series forecasting
- **Mobile Interface**: Responsive design for mobile accessibility
- **Geospatial Analysis**: Advanced location optimization algorithms
- **Economic Modeling**: ROI analysis and investment optimization

### Development Setup
```bash
# For development contributions
git clone <repository-url>
cd EV_Charging_Analysis
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
pre-commit install  # If pre-commit hooks configured
```

## 📞 Support & Troubleshooting

### Common Issues
1. **Data Download Failures**: Ensure internet connection and kagglehub authentication
2. **Memory Issues**: Use smaller datasets or cloud computing resources
3. **Dashboard Not Loading**: Check Streamlit version and port availability
4. **Model Training Slow**: Consider reducing dataset size or using GPU acceleration

### Performance Optimization
- **Data Processing**: Use chunked processing for large datasets
- **Model Training**: Enable GPU support with XGBoost
- **Dashboard**: Cache data loading and computations
- **Memory Management**: Clear unused variables and use efficient data types

---

**Project Status**: ✅ Complete - Production Ready  
**Last Updated**: January 2026  
**Data Freshness**: 2014-2024 (with 12-24 month forecasts)  
**Model Performance**: R² = 0.9035 (Excellent)  

**Note**: This project automatically downloads required datasets from Kaggle on first run. Internet connection is required for initial setup. All generated analysis files are saved locally for offline use.
