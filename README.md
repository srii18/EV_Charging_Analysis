# EV Charging Infrastructure Strategic Dashboard

A comprehensive analysis dashboard for electric vehicle charging infrastructure in India, combining data analysis, machine learning predictions, and strategic recommendations.

## 🚀 Features

### Section 1: Comprehensive Analysis
- **29 States + 5 Union Territories Analysis**: Complete coverage of Indian geography
- **Priority Classification**: 82.4% of states identified as medium priority for infrastructure development
- **Infrastructure Gap Analysis**: Current charging stations vs EV demand analysis
- **Temporal Trends**: EV sales patterns over time
- **Vehicle Type Distribution**: Breakdown by 2-wheelers, 3-wheelers, 4-wheelers, and buses

### Section 2: ML Predictive Modeling
- **XGBoost Model**: Predicts future EV sales trends by state
- **12-24 Month Forecasts**: Identifies states with adoption spike potential
- **Growth Potential Analysis**: Quantifies expected market expansion
- **Feature Importance**: Understands key drivers of EV adoption

### Section 3: Strategic Recommendation System
- **Interactive Maps**: Visual representation of current and recommended charging infrastructure
- **Priority Rankings**: Strategic scoring system for infrastructure investment
- **Implementation Timeline**: Phased rollout plan (Phase 1: 0-6 months, Phase 2: 6-12 months, Phase 3: 12-24 months)
- **Investment Calculations**: Quantified infrastructure requirements

## 📊 Key Insights

### High Priority States (Immediate Action Required)
1. **Uttar Pradesh**: 467,843 EVs, only 3 charging stations
2. **Maharashtra**: 348,151 EVs, only 4 charging stations  
3. **Karnataka**: 261,095 EVs, only 4 charging stations
4. **Rajasthan**: 179,351 EVs, 0 charging stations
5. **Gujarat**: 163,429 EVs, 0 charging stations

### Market Growth Predictions
- **Average Expected Growth**: 23.8% across all states
- **Highest Growth Potential**: Telangana, Nagaland, Arunachal Pradesh
- **Infrastructure Impact**: 50% increase in stations could drive significant adoption

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone/Download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   python strategic_dashboard.py
   ```

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- scikit-learn >= 1.1.0
- xgboost >= 1.7.0
- kagglehub >= 0.2.0

## 📁 Project Structure

```
EV_Charging_Analysis/
├── strategic_dashboard.py      # Main dashboard application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── main.ipynb                # Original analysis notebook
├── ML_model.ipynb            # ML modeling notebook
└── detailed_gap_analysis.csv # Generated analysis results
```

## 🎯 How to Use

### Running the Dashboard
1. Open terminal/command prompt
2. Navigate to the project directory
3. Run: `python strategic_dashboard.py`
4. The dashboard will display three sections sequentially

### Understanding the Output
- **Interactive Visualizations**: All charts are interactive using Plotly
- **Strategic Recommendations**: Actionable insights for infrastructure planning
- **Executive Summary**: High-level overview for decision makers

## 📈 Data Sources

### Primary Datasets
1. **EV Charging Stations**: Indian EV charging stations data (Kaggle)
2. **EV Sales by State**: Electric vehicle sales data by Indian states (Kaggle)

### Data Processing
- **Cleaning**: Removed duplicates, standardized state names
- **Validation**: Cross-referenced state boundaries
- **Aggregation**: Combined datasets for comprehensive analysis

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm**: XGBoost Regressor
- **Features**: Station Count, Stations per 1000 EV, Charging Gap Score
- **Target**: EV Sales Quantity
- **Validation**: Train-test split with performance metrics

### Strategic Scoring
- **Gap Score**: 60% demand weight + 40% infrastructure deficit
- **Priority Classification**: Low (0-0.3), Medium (0.3-0.6), High (0.6-1.0)
- **Strategic Priority**: Combines current gap with growth predictions

## 🎨 Visualizations

### Section 1: Analysis
- Priority distribution pie chart
- Top states by EV sales
- Infrastructure density analysis
- Vehicle type distribution
- Charging gap rankings

### Section 2: ML Predictions
- Adoption spike potential rankings
- Current vs predicted sales comparison
- Infrastructure impact analysis
- Feature importance visualization

### Section 3: Strategic Recommendations
- Interactive strategic map
- Priority rankings
- Investment recommendations
- Implementation timeline

## 📋 Executive Summary

### Current Situation
- **Total States Analyzed**: 29 states + 5 union territories
- **Medium Priority States**: 82.4% require balanced infrastructure growth
- **High Priority States**: 5 states need immediate infrastructure investment

### Key Recommendations
1. **Immediate Focus**: Uttar Pradesh, Maharashtra, Karnataka
2. **Infrastructure Expansion**: 240 new charging stations required
3. **Investment Priority**: High-gap states with high growth potential
4. **Monitoring**: Track adoption trends and adjust plans accordingly

## 🤝 Contributing

### Future Enhancements
- Real-time data integration
- Additional ML models for refined predictions
- Mobile-friendly interface
- API integration for live charging station data

## 📞 Support

For questions or issues:
1. Check the requirements are properly installed
2. Ensure internet connectivity for data downloads
3. Review the error messages for specific issues

---

**Note**: The dashboard automatically downloads required datasets from Kaggle on first run. Internet connection is required for initial setup.
