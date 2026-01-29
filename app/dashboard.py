"""
Streamlit Dashboard for ChargeSmart India
Interactive visualization of EV charging infrastructure analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import STATE_CAPITALS, TIER_COLORS, PRIORITY_THRESHOLDS
from models.analysis import EVInfrastructureAnalyzer
from utils.data_validation import clean_and_validate_data
from utils.logger import setup_logger

# Page configuration
st.set_page_config(
    page_title="ChargeSmart India - EV Infrastructure Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card h3 {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    .metric-card p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
        margin: 0 !important;
    }
    .priority-critical { color: #d62728; font-weight: bold; }
    .priority-high { color: #ff7f0e; font-weight: bold; }
    .priority-medium { color: #2ca02c; font-weight: bold; }
    .priority-low { color: #1f77b4; font-weight: bold; }
    .stDataFrame {
        background-color: white;
        border-radius: 0.5rem;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and prepare data for the dashboard"""
    try:
        # Try to load processed data first
        gap_analysis_path = Path(__file__).parent.parent / "data" / "processed" / "gap_analysis_complete.csv"
        
        if gap_analysis_path.exists():
            gap_analysis = pd.read_csv(gap_analysis_path)
            return gap_analysis, None, None
        else:
            # If processed data doesn't exist, process from raw data
            st.warning("Processed data not found. Loading and processing raw data...")
            
            # Load raw data (you'll need to adjust these paths)
            ev_sales_path = Path(__file__).parent.parent / "data" / "raw" / "EV_Dataset.csv"
            stations_path = Path(__file__).parent.parent / "data" / "raw" / "Indian_EV_Stations_Simplified.csv"
            
            if ev_sales_path.exists() and stations_path.exists():
                ev_sales = pd.read_csv(ev_sales_path)
                stations = pd.read_csv(stations_path)
                
                # Clean and validate data
                ev_sales_clean, stations_clean = clean_and_validate_data(ev_sales, stations)
                
                # Perform analysis
                analyzer = EVInfrastructureAnalyzer()
                gap_analysis = analyzer.perform_gap_analysis(ev_sales_clean, stations_clean)
                
                return gap_analysis, ev_sales_clean, stations_clean
            else:
                st.error("Raw data files not found. Please ensure data files are in the correct location.")
                return None, None, None
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def create_map_visualization(gap_analysis):
    """Create interactive map visualization with accurate Indian boundaries"""
    
    # Add coordinates to gap analysis
    gap_analysis['Latitude'] = gap_analysis['State'].map(lambda x: STATE_CAPITALS.get(x, [20, 77])[0])
    gap_analysis['Longitude'] = gap_analysis['State'].map(lambda x: STATE_CAPITALS.get(x, [20, 77])[1])
    
    # Create map with accurate Indian boundaries
    fig = px.scatter_geo(
        gap_analysis,
        lat='Latitude',
        lon='Longitude',
        size='Recommended_New_Stations',
        color='Recommendation_Tier',
        hover_name='State',
        hover_data={
            'Total_EV_Sales': ':,.0f',
            'Total_Stations': ':,.0f',
            'EVs_per_Station': ':,.1f',
            'Priority_Score': ':.1f',
            'Recommended_New_Stations': ':,.0f'
        },
        size_max=60,
        projection="natural earth",
        scope="asia",
        center={'lat': 20.5937, 'lon': 78.9629},  # India center coordinates
        color_discrete_map=TIER_COLORS,
        title="EV Charging Infrastructure Gap Analysis - India",
        fitbounds="locations"
    )
    
    # Update layout for better Indian map visualization
    fig.update_layout(
        height=700,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2E86AB",
            coastlinewidth=2,
            showland=True,
            landcolor="#F8F9FA",
            showocean=True,
            oceancolor="#E3F2FD",
            showlakes=True,
            lakecolor="#BBDEFB",
            showrivers=True,
            rivercolor="#64B5F6",
            riverwidth=1,
            showcountries=True,
            countrycolor="#37474F",
            countrywidth=1.5,
            # Focus on India and neighboring countries
            resolution=50,
            lonaxis=dict(range=[68, 98]),  # India longitude bounds
            lataxis=dict(range=[6, 38])    # India latitude bounds
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Add India-specific annotations
    fig.add_annotation(
        text="India",
        x=78.9629,
        y=20.5937,
        showarrow=False,
        font=dict(size=16, color="#37474F", family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#37474F",
        borderwidth=2,
        borderpad=4
    )
    
    return fig

def create_choropleth_map(gap_analysis):
    """Create proper choropleth map of India with accurate state boundaries"""
    
    try:
        # Load India state boundaries from GeoJSON
        geojson_path = Path(__file__).parent.parent / "data" / "india_states.geojson"
        
        if geojson_path.exists():
            with open(geojson_path, 'r') as f:
                india_geojson = json.load(f)
        else:
            # Fallback to simple visualization
            return create_fallback_map(gap_analysis)
        
        # Create choropleth map with GeoJSON
        fig = go.Figure(go.Choroplethmapbox(
            geojson=india_geojson,
            locations=gap_analysis['State'],
            z=gap_analysis['Priority_Score'],
            colorscale=[
                [0, '#2E86AB'],    # Low - Blue
                [0.25, '#2CA02C'], # Medium - Green  
                [0.5, '#FF7F0E'],  # High - Orange
                [0.75, '#D62728'], # Critical - Red
                [1, '#8B0000']     # Very Critical - Dark Red
            ],
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='black',
            text=gap_analysis['State'],
            hovertemplate='<b>%{location}</b><br>' +
                         'Priority Score: %{z:.1f}<br>' +
                         'EV Sales: %{customdata[0]:,.0f}<br>' +
                         'Current Stations: %{customdata[1]:,.0f}<br>' +
                         'Recommended: %{customdata[2]:,.0f}<br>' +
                         '<extra></extra>',
            customdata=gap_analysis[['Total_EV_Sales', 'Total_Stations', 'Recommended_New_Stations']].values
        ))
        
        # Update layout for mapbox
        fig.update_layout(
            title='India EV Infrastructure Gap Analysis - State Boundaries',
            height=700,
            mapbox=dict(
                style='carto-positron',
                center=dict(lat=20.5937, lon=78.9629),
                zoom=4.5
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
        
    except Exception as e:
        # Fallback to simple visualization if GeoJSON fails
        return create_fallback_map(gap_analysis)

def create_fallback_map(gap_analysis):
    """Create fallback map with simple state representation"""
    
    fig = go.Figure()
    
    # Add India outline
    fig.add_trace(go.Scattergeo(
        lon=[68.2, 97.4, 97.4, 68.2, 68.2],
        lat=[6.8, 6.8, 37.1, 37.1, 6.8],
        mode='lines',
        line=dict(width=2, color='black'),
        fill='toself',
        fillcolor='lightgray',
        name='India',
        hoverinfo='skip'
    ))
    
    # Add state data
    for _, row in gap_analysis.iterrows():
        color = TIER_COLORS.get(row['Recommendation_Tier'], '#808080')
        lat, lon = STATE_CAPITALS.get(row['State'], [20, 77])
        
        fig.add_trace(go.Scattergeo(
            lon=[lon],
            lat=[lat],
            mode='markers',
            marker=dict(
                size=25,
                color=color,
                line=dict(width=2, color='black'),
                symbol='circle',
                opacity=0.8
            ),
            text=f"<b>{row['State']}</b><br>"
                 f"Priority: {row['Recommendation_Tier']}<br>"
                 f"Score: {row['Priority_Score']:.1f}<br>"
                 f"EV Sales: {row['Total_EV_Sales']:,.0f}<br>"
                 f"Current Stations: {row['Total_Stations']:,.0f}<br>"
                 f"Recommended: {row['Recommended_New_Stations']:,.0f}",
            hoverinfo='text',
            name=row['State']
        ))
    
    fig.update_layout(
        title='India EV Infrastructure Gap Analysis - State View',
        height=700,
        geo=dict(
            scope='asia',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='black',
            coastlinewidth=2,
            showland=True,
            landcolor='white',
            showocean=True,
            oceancolor='#E3F2FD',
            showcountries=True,
            countrycolor='black',
            countrywidth=2,
            showsubunits=True,
            subunitcolor='black',
            subunitwidth=1,
            resolution=50,
            lonaxis=dict(range=[68, 98]),
            lataxis=dict(range=[6, 38]),
            center=dict(lat=20.5937, lon=78.9629)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_priority_chart(gap_analysis):
    
    priority_counts = gap_analysis['Recommendation_Tier'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=priority_counts.index,
            y=priority_counts.values,
            marker_color=[TIER_COLORS.get(tier, '#808080') for tier in priority_counts.index]
        )
    ])
    
    fig.update_layout(
        title="Distribution of States by Priority Level",
        xaxis_title="Priority Tier",
        yaxis_title="Number of States",
        height=400
    )
    
    return fig

def create_gap_analysis_chart(gap_analysis):
    """Create gap analysis comparison chart"""
    
    # Get top 15 states by priority score
    top_states = gap_analysis.nlargest(15, 'Priority_Score').sort_values('Priority_Score', ascending=True)
    
    fig = go.Figure()
    
    # Add bars for EV sales
    fig.add_trace(go.Bar(
        y=top_states['State'],
        x=top_states['Total_EV_Sales'],
        name='EV Sales',
        orientation='h',
        marker_color='lightblue'
    ))
    
    # Add bars for current stations (scaled)
    fig.add_trace(go.Bar(
        y=top_states['State'],
        x=top_states['Total_Stations'] * 1000,  # Scale to make visible
        name='Current Stations (×1000)',
        orientation='h',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title="Top 15 States: EV Sales vs Current Infrastructure",
        xaxis_title="Count",
        yaxis_title="State",
        height=600,
        barmode='group'
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ ChargeSmart India</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">EV Charging Infrastructure Strategic Analysis</h3>', unsafe_allow_html=True)
    
    # Load data
    gap_analysis, ev_sales, stations = load_data()
    
    if gap_analysis is None:
        st.error("Unable to load data. Please check data files and try again.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters & Options")
    
    # Priority tier filter
    selected_tiers = st.sidebar.multiselect(
        "Select Priority Tiers",
        options=gap_analysis['Recommendation_Tier'].unique(),
        default=gap_analysis['Recommendation_Tier'].unique()
    )
    
    # Filter data
    filtered_data = gap_analysis[gap_analysis['Recommendation_Tier'].isin(selected_tiers)]
    
    # Metrics section
    st.header("📊 Key Metrics - India EV Infrastructure Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_states = len(filtered_data)
        # Count Union Territories (common UTs in India)
        union_territories = ['Delhi', 'Goa', 'Puducherry', 'Chandigarh', 'Ladakh', 'Jammu & Kashmir']
        ut_count = len(filtered_data[filtered_data['State'].isin(union_territories)])
        states_count = total_states - ut_count
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{states_count} States</h3>
            <p>{ut_count} Union Territories</p>
            <p style="font-size: 0.8em; opacity: 0.8;">Total: {total_states} Regions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_ev_sales = filtered_data['Total_EV_Sales'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_ev_sales:,.0f}</h3>
            <p>Total EV Sales (2022-24)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_stations = filtered_data['Total_Stations'].sum()
        states_with_infrastructure = len(filtered_data[filtered_data['Total_Stations'] > 0])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_stations:,.0f}</h3>
            <p>Current Charging Stations</p>
            <p style="font-size: 0.8em; opacity: 0.8;">In {states_with_infrastructure} regions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        recommended_stations = filtered_data['Recommended_New_Stations'].sum()
        high_priority_states = len(filtered_data[filtered_data['Recommendation_Tier'] == 'HIGH - Priority Investment'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{recommended_stations:,.0f}</h3>
            <p>Recommended Stations</p>
            <p style="font-size: 0.8em; opacity: 0.8;">{high_priority_states} High Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Priority distribution
    st.header("🎯 Priority Distribution")
    priority_chart = create_priority_chart(filtered_data)
    st.plotly_chart(priority_chart, use_container_width=True)
    
    # Map visualization
    st.header("🗺️ Geographic Analysis")
    
    # Add map type selector
    map_type = st.radio(
        "Select Map Type",
        ["Interactive Bubble Map", "State Boundary Map"],
        horizontal=True
    )
    
    if map_type == "Interactive Bubble Map":
        map_fig = create_map_visualization(filtered_data)
    else:
        map_fig = create_choropleth_map(filtered_data)
    
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Gap analysis
    st.header("📈 Infrastructure Gap Analysis")
    gap_chart = create_gap_analysis_chart(filtered_data)
    st.plotly_chart(gap_chart, use_container_width=True)
    
    # Detailed data table
    st.header("📋 Detailed Analysis")
    
    # Show/hide columns option
    available_columns = ['State', 'Priority_Score', 'Recommendation_Tier', 'Total_EV_Sales', 
                        'Total_Stations', 'EVs_per_Station', 'Recommended_New_Stations']
    
    selected_columns = st.multiselect(
        "Select columns to display",
        options=available_columns,
        default=['State', 'Priority_Score', 'Recommendation_Tier', 'Total_EV_Sales', 'Total_Stations']
    )
    
    if selected_columns:
        # Format the data for display
        display_data = filtered_data[selected_columns].copy()
        
        # Format numeric columns
        if 'Priority_Score' in display_data.columns:
            display_data['Priority_Score'] = display_data['Priority_Score'].round(1)
        if 'Total_EV_Sales' in display_data.columns:
            display_data['Total_EV_Sales'] = display_data['Total_EV_Sales'].map('{:,}'.format)
        if 'Total_Stations' in display_data.columns:
            display_data['Total_Stations'] = display_data['Total_Stations'].map('{:,}'.format)
        if 'Recommended_New_Stations' in display_data.columns:
            display_data['Recommended_New_Stations'] = display_data['Recommended_New_Stations'].map('{:,}'.format)
        
        st.dataframe(display_data, use_container_width=True)
    
    # Download section
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Filtered Data (CSV)"):
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="ev_infrastructure_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Download Summary Report"):
            # Create summary report
            summary = {
                'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'States Analyzed': len(filtered_data),
                'Total EV Sales': filtered_data['Total_EV_Sales'].sum(),
                'Current Stations': filtered_data['Total_Stations'].sum(),
                'Recommended Stations': filtered_data['Recommended_New_Stations'].sum(),
                'Critical Priority States': len(filtered_data[filtered_data['Recommendation_Tier'] == 'CRITICAL - Immediate Action']),
                'High Priority States': len(filtered_data[filtered_data['Recommendation_Tier'] == 'HIGH - Priority Investment'])
            }
            
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download Summary",
                data=summary_csv,
                file_name="ev_infrastructure_summary.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
