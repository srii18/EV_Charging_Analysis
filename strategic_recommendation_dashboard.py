import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
import os

# Set page configuration
st.set_page_config(
    page_title="EV Charging Infrastructure Recommendation System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .high-priority {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .medium-priority {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    }
    .low-priority {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CONFIG = {
    'data_paths': {
        'forecast': 'data/processed/ev_sales_forecast.csv',
        'stations': 'data/raw/Indian_EV_Stations_Simplified.csv',
        'gap_analysis': 'data/processed/detailed_gap_analysis.csv',
        'priority': 'data/processed/priority_states.csv',
        'geojson': 'data/india_states.geojson'
    },
    'scoring_weights': {
        'sales': 0.4,
        'growth': 0.3,
        'infrastructure': 0.3
    },
    'investment_weights': {
        'infrastructure_need': 0.5,
        'projected_sales': 0.3,
        'projected_growth': 0.2
    }
}

# Load data with error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load all required data files with comprehensive error handling"""
    try:
        data = {}
        
        # Check if files exist
        for key, path in CONFIG['data_paths'].items():
            if not os.path.exists(path):
                st.error(f"❌ Missing file: `{path}`")
                st.info("Please ensure all data files are in the correct directory structure.")
                st.stop()
        
        # Load CSV files
        data['forecast'] = pd.read_csv(CONFIG['data_paths']['forecast'])
        data['stations'] = pd.read_csv(CONFIG['data_paths']['stations'])
        data['gap_analysis'] = pd.read_csv(CONFIG['data_paths']['gap_analysis'])
        data['priority'] = pd.read_csv(CONFIG['data_paths']['priority'])
        
        # Load GeoJSON
        with open(CONFIG['data_paths']['geojson'], 'r') as f:
            data['geojson'] = json.load(f)
        
        # Validate required columns
        required_cols = {
            'forecast': ['State', 'Current_Monthly_Sales', 'Predicted_Monthly_Sales', 'Growth_Rate_%'],
            'stations': ['State', 'Station Name'],
            'gap_analysis': ['State', 'Station_Count', 'EV_Sales_Quantity'],
            'priority': ['State', 'Recommended_New_Stations']
        }
        
        for key, cols in required_cols.items():
            missing = [col for col in cols if col not in data[key].columns]
            if missing:
                st.error(f"❌ Missing columns in {key}: {missing}")
                st.stop()
        
        return data
        
    except pd.errors.EmptyDataError:
        st.error("❌ One or more data files are empty")
        st.stop()
    except json.JSONDecodeError:
        st.error("❌ Invalid GeoJSON file format")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
with st.spinner('Loading data...'):
    data = load_data()
    forecast_df = data['forecast']
    stations_df = data['stations']
    gap_analysis_df = data['gap_analysis']
    priority_df = data['priority']
    geojson_data = data['geojson']

# Data preprocessing
@st.cache_data
def preprocess_data(forecast, gap_analysis, priority, stations, weights):
    """Preprocess and merge all data sources"""
    try:
        # Clean state names for consistency
        forecast['State'] = forecast['State'].str.strip().str.title()
        gap_analysis['State'] = gap_analysis['State'].str.strip().str.title()
        priority['State'] = priority['State'].str.strip().str.title()
        
        # Ensure numeric columns
        forecast['Growth_Rate_%'] = pd.to_numeric(forecast['Growth_Rate_%'], errors='coerce')
        forecast['Predicted_Monthly_Sales'] = pd.to_numeric(forecast['Predicted_Monthly_Sales'], errors='coerce')
        forecast['Current_Monthly_Sales'] = pd.to_numeric(forecast['Current_Monthly_Sales'], errors='coerce')
        
        # Calculate growth category
        forecast['Growth_Category'] = pd.cut(
            forecast['Growth_Rate_%'].fillna(0),
            bins=[-np.inf, -5, 5, 20, np.inf],
            labels=['Declining', 'Stable', 'Growing', 'High Growth']
        )
        
        # Merge datasets
        combined = forecast.merge(gap_analysis, on='State', how='left')
        combined = combined.merge(priority[['State', 'Recommended_New_Stations']], on='State', how='left')
        
        # Fill missing values
        combined['Stations_per_1000_EV'] = combined['Stations_per_1000_EV'].fillna(0)
        combined['Station_Count'] = combined['Station_Count'].fillna(0)
        combined['Recommended_New_Stations'] = combined['Recommended_New_Stations'].fillna(0)
        
        # Calculate infrastructure need score
        combined['Infrastructure_Need_Score'] = (
            combined['Predicted_Monthly_Sales'].fillna(0) * weights['sales'] +
            combined['Growth_Rate_%'].fillna(0) * weights['growth'] +
            (1 / (combined['Stations_per_1000_EV'] + 0.001)) * weights['infrastructure']
        )
        
        # Normalize the score
        scaler = MinMaxScaler()
        combined['Infrastructure_Need_Score'] = scaler.fit_transform(
            combined[['Infrastructure_Need_Score']]
        )
        
        # Add priority if missing
        if 'Priority' not in combined.columns:
            combined['Priority'] = pd.cut(
                combined['Infrastructure_Need_Score'],
                bins=[0, 0.33, 0.67, 1],
                labels=['Low', 'Medium', 'High']
            )
        
        # Drop rows with all NaN values
        combined = combined.dropna(how='all')
        
        return combined
        
    except Exception as e:
        st.error(f"❌ Error preprocessing data: {str(e)}")
        st.stop()

combined_df = preprocess_data(
    forecast_df, 
    gap_analysis_df, 
    priority_df, 
    stations_df,
    CONFIG['scoring_weights']
)

# Sidebar
st.sidebar.markdown("## 📊 Dashboard Controls")
st.sidebar.markdown("---")

# Time period selector
time_period = st.sidebar.selectbox(
    "Select Analysis Period",
    ["Current Month", "3 Months Projection", "6 Months Projection", "12 Months Projection"],
    index=0
)

# Growth filter
available_growth = combined_df['Growth_Category'].dropna().unique()
growth_filter = st.sidebar.multiselect(
    "Filter by Growth Category",
    options=available_growth,
    default=available_growth
)

# Priority filter
available_priority = combined_df['Priority'].dropna().unique()
priority_filter = st.sidebar.multiselect(
    "Filter by Priority Level",
    options=available_priority,
    default=available_priority
)

# Apply filters
filtered_df = combined_df[
    (combined_df['Growth_Category'].isin(growth_filter)) &
    (combined_df['Priority'].isin(priority_filter))
].copy()

# Check for empty results
if filtered_df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust your selection.")
    st.stop()

# Apply time period calculations with compound growth
time_multipliers = {
    "Current Month": 1,
    "3 Months Projection": 3,
    "6 Months Projection": 6,
    "12 Months Projection": 12
}

multiplier = time_multipliers[time_period]

# Fixed projection logic with improved growth rate handling
def calculate_projected_sales(row, multiplier):
    """Calculate projected sales with robust growth rate handling"""
    current_sales = row['Current_Monthly_Sales']
    predicted_sales = row['Predicted_Monthly_Sales']
    growth_rate = row['Growth_Rate_%']
    
    # Handle zero and very low baseline sales to prevent mathematical artifacts
    if current_sales <= 0:
        # For states with zero current sales, use absolute growth instead of percentage
        absolute_growth = predicted_sales - current_sales
        if absolute_growth <= 0:
            # No growth scenario
            return predicted_sales
        else:
            # Apply conservative growth factor for zero-baseline states
            # Cap the effective growth rate to prevent extreme projections
            conservative_growth_rate = min(growth_rate, 50)  # Cap at 50%
            return predicted_sales * (1 + conservative_growth_rate / 100 * (multiplier - 1))
    elif current_sales <= 10:
        # For states with very low baseline sales (<=10), use conservative growth
        # This prevents small baselines from creating extreme compound growth effects
        conservative_growth_rate = min(growth_rate, 30)  # More conservative cap for low baselines
        return predicted_sales * (1 + conservative_growth_rate / 100 * (multiplier - 1))
    else:
        # Normal compound growth for states with meaningful existing sales
        return predicted_sales * ((1 + growth_rate / 100) ** multiplier)

filtered_df['Projected_Sales'] = filtered_df.apply(
    lambda row: calculate_projected_sales(row, multiplier), 
    axis=1
)

# Recalculate projected growth with improved handling
def calculate_projected_growth(row, multiplier):
    """Calculate projected growth rate with robust handling"""
    current_sales = row['Current_Monthly_Sales']
    projected_sales = row['Projected_Sales']
    
    if current_sales <= 0:
        # For zero baseline, use very conservative growth estimate
        # These states already get max infrastructure need score, so growth should be minimal
        return min(row['Growth_Rate_%'], 20) * (multiplier / 12)  # Much more conservative
    elif current_sales <= 10:
        # For very low baseline, use more conservative estimate
        return min(row['Growth_Rate_%'], 30) * multiplier  # Cap at 30% total
    else:
        return ((projected_sales / current_sales) - 1) * 100

filtered_df['Projected_Growth'] = filtered_df.apply(
    lambda row: calculate_projected_growth(row, multiplier), 
    axis=1
)

# Recalculate adjusted infrastructure need
filtered_df['Adjusted_Infrastructure_Need'] = (
    filtered_df['Projected_Sales'] / filtered_df['Projected_Sales'].max() * CONFIG['scoring_weights']['sales'] +
    filtered_df['Projected_Growth'] / (filtered_df['Projected_Growth'].max() + 0.001) * CONFIG['scoring_weights']['growth'] +
    (1 / (filtered_df['Stations_per_1000_EV'] + 0.001)) / 
    (1 / (filtered_df['Stations_per_1000_EV'] + 0.001)).max() * CONFIG['scoring_weights']['infrastructure']
)

# Normalize
scaler = MinMaxScaler()
filtered_df['Adjusted_Infrastructure_Need'] = scaler.fit_transform(
    filtered_df[['Adjusted_Infrastructure_Need']]
)

# Main header
st.markdown('<h1 class="main-header">⚡ EV Charging Infrastructure Recommendation System</h1>', unsafe_allow_html=True)

# Key Metrics Section
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ev_sales = filtered_df['Projected_Sales'].sum()
    period_text = time_period.lower().replace(' projection', '')
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total EV Sales</h3>
        <h2>{total_ev_sales:,.0f}</h2>
        <p>Projected for {period_text}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_stations = stations_df['Station Name'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <h3>Charging Stations</h3>
        <h2>{total_stations:,}</h2>
        <p>Currently operational</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    high_priority_states = filtered_df[filtered_df['Priority'] == 'High'].shape[0]
    st.markdown(f"""
    <div class="metric-card high-priority">
        <h3>High Priority States</h3>
        <h2>{high_priority_states}</h2>
        <p>Need immediate attention</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_growth = filtered_df['Projected_Growth'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h3>Avg Growth Rate</h3>
        <h2>{avg_growth:.1f}%</h2>
        <p>For {period_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Strategic Recommendations Section
st.markdown("## 🎯 Strategic Recommendations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 States by Infrastructure Need")
    top_states = filtered_df.nlargest(10, 'Adjusted_Infrastructure_Need')
    
    fig_top_states = px.bar(
        top_states,
        x='Adjusted_Infrastructure_Need',
        y='State',
        orientation='h',
        title=f'Infrastructure Need Score ({time_period})',
        color='Adjusted_Infrastructure_Need',
        color_continuous_scale='viridis',
        labels={'Adjusted_Infrastructure_Need': 'Need Score'}
    )
    fig_top_states.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_top_states, width='stretch')

with col2:
    st.markdown("### Priority Distribution")
    priority_counts = filtered_df['Priority'].value_counts()
    
    colors = {'High': '#ee5a24', 'Medium': '#feca57', 'Low': '#48dbfb'}
    fig_priority = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title='States by Priority Level',
        color=priority_counts.index,
        color_discrete_map=colors
    )
    fig_priority.update_layout(height=400)
    st.plotly_chart(fig_priority, width='stretch')

# Geographic Visualization
st.markdown("## 🗺️ Geographic Analysis")

tab1, tab2, tab3 = st.tabs(["Current Stations", "Priority Recommendations", "Heat Map"])

with tab1:
    st.markdown("### Current Charging Station Locations")
    try:
        m_current = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        for _, station in stations_df.iterrows():
            if pd.notna(station.get('Latitude')) and pd.notna(station.get('Longitude')):
                folium.CircleMarker(
                    location=[station['Latitude'], station['Longitude']],
                    radius=3,
                    popup=f"{station['Station Name']}<br>State: {station['State']}",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.6
                ).add_to(m_current)
        
        st_folium(m_current, width=700, height=500)
    except Exception as e:
        st.warning(f"⚠️ Could not display station map: {str(e)}")

with tab2:
    st.markdown("### Infrastructure Priority by State")
    try:
        m_rec = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        priority_colors = {'High': '#ee5a24', 'Medium': '#feca57', 'Low': '#48dbfb'}
        
        for _, rec in filtered_df.iterrows():
            state = rec['State']
            color = priority_colors.get(rec['Priority'], 'gray')
            
            # Find state geometry in geojson
            for feature in geojson_data['features']:
                if feature['properties']['name'] == state:
                    folium.GeoJson(
                        feature,
                        style_function=lambda x, color=color: {
                            'fillColor': color,
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.5
                        },
                        tooltip=f"{state} - Priority: {rec['Priority']}"
                    ).add_to(m_rec)
                    break
        
        st_folium(m_rec, width=700, height=500)
    except Exception as e:
        st.warning(f"⚠️ Could not display priority map: {str(e)}")

with tab3:
    st.markdown("### Infrastructure Need Heat Map")
    try:
        fig_choropleth = px.choropleth(
            filtered_df,
            geojson=geojson_data,
            featureidkey="properties.name",
            locations='State',
            color='Adjusted_Infrastructure_Need',
            color_continuous_scale="Viridis",
            title=f"Infrastructure Need Score by State ({time_period})",
            hover_data={
                'State': True,
                'Projected_Sales': ':,.0f',
                'Projected_Growth': ':.1f',
                'Stations_per_1000_EV': ':.2f',
                'Priority': True
            }
        )
        
        fig_choropleth.update_geos(fitbounds="locations", visible=False)
        fig_choropleth.update_layout(height=600)
        st.plotly_chart(fig_choropleth, width='stretch')
    except Exception as e:
        st.warning(f"⚠️ Could not display heat map: {str(e)}")

# Detailed Analysis Section
st.markdown("## 📊 Detailed Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Growth Analysis", "Station Distribution", "Gap Analysis", "Investment Priorities"])

with tab1:
    st.markdown("### EV Sales Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth rate distribution
        fig_growth_dist = px.histogram(
            filtered_df,
            x='Projected_Growth',
            nbins=20,
            title=f'Growth Rate Distribution ({time_period})',
            color='Priority',
            labels={'Projected_Growth': 'Growth Rate (%)'}
        )
        fig_growth_dist.update_layout(height=400)
        st.plotly_chart(fig_growth_dist, width='stretch')
    
    with col2:
        # Current vs Predicted sales
        fig_sales_comparison = px.scatter(
            filtered_df,
            x='Current_Monthly_Sales',
            y='Projected_Sales',
            color='Priority',
            size='Adjusted_Infrastructure_Need',
            hover_data=['State', 'Projected_Growth'],
            title=f'Current vs Projected Sales ({time_period})',
            labels={
                'Current_Monthly_Sales': 'Current Sales',
                'Projected_Sales': 'Projected Sales'
            }
        )
        
        # Add reference line
        max_val = filtered_df['Current_Monthly_Sales'].max()
        fig_sales_comparison.add_shape(
            type='line',
            x0=0, y0=0,
            x1=max_val,
            y1=max_val * ((1 + filtered_df['Growth_Rate_%'].mean() / 100) ** multiplier),
            line=dict(color='red', dash='dash')
        )
        
        fig_sales_comparison.update_layout(height=400)
        st.plotly_chart(fig_sales_comparison, width='stretch')

with tab2:
    st.markdown("### Charging Station Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Station count by state
        station_by_state = stations_df['State'].value_counts().head(10)
        fig_station_count = px.bar(
            x=station_by_state.values,
            y=station_by_state.index,
            orientation='h',
            title='Top 10 States by Station Count',
            labels={'x': 'Number of Stations', 'y': 'State'}
        )
        fig_station_count.update_layout(height=400)
        st.plotly_chart(fig_station_count, width='stretch')
    
    with col2:
        # Power distribution
        if 'Power (kW)' in stations_df.columns:
            fig_power_dist = px.histogram(
                stations_df[stations_df['Power (kW)'] > 0],
                x='Power (kW)',
                nbins=20,
                title='Charging Station Power Distribution',
                color='Connector Type' if 'Connector Type' in stations_df.columns else None,
                labels={'Power (kW)': 'Power (kW)'}
            )
            fig_power_dist.update_layout(height=400)
            st.plotly_chart(fig_power_dist, width='stretch')
        else:
            st.info("Power distribution data not available")

with tab3:
    st.markdown("### Infrastructure Gap Analysis")
    
    # Gap analysis table
    gap_cols = ['State', 'Station_Count', 'EV_Sales_Quantity', 
                'Stations_per_1000_EV', 'Priority']
    
    # Only include columns that exist
    available_cols = [col for col in gap_cols if col in filtered_df.columns]
    gap_table = filtered_df[available_cols].copy()
    
    if 'Charging_Gap_Score' in filtered_df.columns:
        gap_table['Charging_Gap_Score'] = filtered_df['Charging_Gap_Score']
        gap_table = gap_table.sort_values('Charging_Gap_Score', ascending=False).head(15)
        gap_table['Charging_Gap_Score'] = gap_table['Charging_Gap_Score'].round(3)
    
    gap_table['Station_Count'] = gap_table['Station_Count'].fillna(0).astype(int)
    gap_table['Stations_per_1000_EV'] = gap_table['Stations_per_1000_EV'].fillna(0).round(2)
    
    st.dataframe(gap_table, width='stretch', hide_index=True)
    
    # Gap visualization
    fig_gap = px.scatter(
        filtered_df,
        x='Stations_per_1000_EV',
        y='Projected_Sales',
        color='Priority',
        size='Adjusted_Infrastructure_Need',
        hover_data=['State'],
        title=f'Infrastructure Gap: Projected EV Sales vs Station Availability ({time_period})',
        labels={
            'Stations_per_1000_EV': 'Stations per 1000 EVs',
            'Projected_Sales': 'Projected Sales'
        }
    )
    fig_gap.update_layout(height=500)
    st.plotly_chart(fig_gap, width='stretch')

with tab4:
    st.markdown("### Investment Priority Recommendations")
    
    # Create investment recommendations
    investment_df = filtered_df.copy()
    investment_df['Investment_Priority_Score'] = (
        investment_df['Adjusted_Infrastructure_Need'] * CONFIG['investment_weights']['infrastructure_need'] +
        investment_df['Projected_Sales'] / (investment_df['Projected_Sales'].max() + 0.001) * CONFIG['investment_weights']['projected_sales'] +
        (investment_df['Projected_Growth'] + 100) / 200 * CONFIG['investment_weights']['projected_growth']
    )
    
    investment_df = investment_df.sort_values('Investment_Priority_Score', ascending=False)
    
    # Top investment recommendations
    top_investments = investment_df.head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Investment priority chart
        fig_investment = px.bar(
            top_investments,
            x='Investment_Priority_Score',
            y='State',
            orientation='h',
            title=f'Top 10 Investment Priorities ({time_period})',
            color='Investment_Priority_Score',
            color_continuous_scale='plasma',
            labels={'Investment_Priority_Score': 'Priority Score'}
        )
        fig_investment.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_investment, width='stretch')
    
    with col2:
        # Detailed recommendations table
        rec_cols = ['State', 'Priority', 'Projected_Sales', 
                    'Projected_Growth', 'Station_Count', 'Investment_Priority_Score']
        rec_table = top_investments[rec_cols].copy()
        rec_table['Station_Count'] = rec_table['Station_Count'].fillna(0).astype(int)
        rec_table['Investment_Priority_Score'] = rec_table['Investment_Priority_Score'].round(3)
        rec_table['Projected_Sales'] = rec_table['Projected_Sales'].round(0)
        rec_table['Projected_Growth'] = rec_table['Projected_Growth'].round(1)
        
        st.dataframe(rec_table, width='stretch', hide_index=True)

# Executive Summary Section
st.markdown("## 📋 Executive Summary")

col1, col2, col3 = st.columns(3)

with col1:
    high_priority_count = filtered_df[filtered_df['Priority'] == 'High'].shape[0]
    total_recommended = filtered_df['Recommended_New_Stations'].sum()
    high_growth_count = filtered_df[filtered_df['Projected_Growth'] > 10].shape[0]
    
    st.markdown(f"""
    ### Critical Findings
    - **High Priority States**: {high_priority_count} require immediate infrastructure investment
    - **Total Gap**: {total_recommended:.0f} additional stations needed to meet demand
    - **Growth Leaders**: {high_growth_count} states showing >10% projected growth
    """)

with col2:
    fastest_growing_idx = filtered_df['Projected_Growth'].idxmax()
    fastest_state = filtered_df.loc[fastest_growing_idx, 'State']
    fastest_growth = filtered_df.loc[fastest_growing_idx, 'Projected_Growth']
    underserved_count = filtered_df[filtered_df['Stations_per_1000_EV'] < 1].shape[0]
    high_potential = filtered_df[
        (filtered_df['Projected_Growth'] > 5) & 
        (filtered_df['Stations_per_1000_EV'] < 0.1)
    ].shape[0]
    
    st.markdown(f"""
    ### Market Opportunities
    - **Fastest Growing**: {fastest_state} with {fastest_growth:.1f}% projected growth rate
    - **Underserved**: {underserved_count} states with <1 station per 1000 EVs
    - **High Potential**: {high_potential} states with growing EV sales but limited infrastructure
    """)

with col3:
    st.markdown("""
    ### Strategic Recommendations
    1. Focus on top 5 high-priority states
    2. Expand to medium-priority growth markets
    3. Strengthen infrastructure in stable markets
    4. Monitor monthly growth indicators
    """)

# Data Export
st.markdown("## 💾 Export Data")
col1, col2, col3 = st.columns(3)

with col1:
    csv_filtered = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv_filtered,
        file_name=f"ev_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    if not investment_df.empty:
        csv_investment = investment_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Investment Priorities",
            data=csv_investment,
            file_name=f"investment_priorities_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if not top_states.empty:
        csv_top_states = top_states.to_csv(index=False)
        st.download_button(
            label="📥 Download Top States",
            data=csv_top_states,
            file_name=f"top_states_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>⚡ EV Charging Infrastructure Recommendation System</p>
    <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Data Source: EV Sales Forecast & Charging Station Analysis</p>
</div>
""", unsafe_allow_html=True)
