import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EV Charging Dashboard", page_icon="⚡", layout="wide")

class EVChargingDashboard:
    def __init__(self):
        self.gap_analysis = None
        self.ml_predictions = None
        
    def load_data(self):
        try:
            self.gap_analysis = pd.read_csv('detailed_gap_analysis.csv')
            st.success("✓ Data loaded successfully")
        except:
            self._create_sample_data()
            
    def _create_sample_data(self):
        states = ['Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Rajasthan', 'Gujarat',
                 'Tamil Nadu', 'Bihar', 'Delhi', 'Madhya Pradesh', 'Assam']
        
        sample_data = {
            'State': states,
            'Station_Count': [3, 4, 4, 0, 0, 40, 0, 1, 1, 0],
            'EV_Sales_Quantity': [467843, 348151, 261095, 179351, 163429, 165261, 152279, 142454, 110517, 105754],
            'Charging_Gap_Score': [0.999, 0.846, 0.734, 0.630, 0.610, 0.596, 0.595, 0.582, 0.541, 0.536],
            'Priority': ['High', 'High', 'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium']
        }
        
        self.gap_analysis = pd.DataFrame(sample_data)
        self.ml_predictions = self.gap_analysis.copy()
        self.ml_predictions['Predicted_Sales_24m'] = self.ml_predictions['EV_Sales_Quantity'] * np.random.uniform(1.1, 1.5, len(self.ml_predictions))
        self.ml_predictions['Adoption_Spike_Potential'] = ((self.ml_predictions['Predicted_Sales_24m'] - self.ml_predictions['EV_Sales_Quantity']) / self.ml_predictions['EV_Sales_Quantity']) * 100
        st.success("✓ Sample data created")

def main():
    dashboard = EVChargingDashboard()
    
    st.title("⚡ EV Charging Infrastructure Strategic Dashboard")
    st.markdown("---")
    
    # Load data
    if dashboard.gap_analysis is None:
        dashboard.load_data()
    
    # Sidebar navigation
    section = st.sidebar.selectbox(
        "Choose Section:",
        ["📊 Infrastructure Analysis", "🤖 ML Predictions", "🗺️ Strategic Recommendations"]
    )
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("States Analyzed", dashboard.gap_analysis['State'].nunique())
    with col2:
        priority_counts = dashboard.gap_analysis['Priority'].value_counts()
        st.metric("High Priority", priority_counts.get('High', 0))
    with col3:
        st.metric("Total EV Sales", f"{dashboard.gap_analysis['EV_Sales_Quantity'].sum():,.0f}")
    with col4:
        st.metric("Charging Stations", f"{dashboard.gap_analysis['Station_Count'].sum():,.0f}")
    
    st.markdown("---")
    
    if section == "📊 Infrastructure Analysis":
        st.header("📊 Infrastructure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Priority distribution
            priority_counts = dashboard.gap_analysis['Priority'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                hole=0.3
            )])
            fig_pie.update_layout(title="Priority Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Top states by EV sales
            top_states = dashboard.gap_analysis.nlargest(10, 'EV_Sales_Quantity')
            fig_bar = go.Figure(data=[go.Bar(
                x=top_states['EV_Sales_Quantity'],
                y=top_states['State'],
                orientation='h'
            )])
            fig_bar.update_layout(title="Top States by EV Sales")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Gap analysis
        st.subheader("Charging Gap Analysis")
        gap_data = dashboard.gap_analysis.nlargest(10, 'Charging_Gap_Score')
        fig_gap = go.Figure(data=[go.Bar(
            x=gap_data['Charging_Gap_Score'],
            y=gap_data['State'],
            orientation='h',
            marker_color='red'
        )])
        fig_gap.update_layout(title="States with Highest Charging Gap")
        st.plotly_chart(fig_gap, use_container_width=True)
        
        # Data table
        st.subheader("Detailed Analysis")
        st.dataframe(dashboard.gap_analysis.round(3), use_container_width=True)
    
    elif section == "🤖 ML Predictions":
        st.header("🤖 ML Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Adoption spike potential
            top_spikes = dashboard.ml_predictions.nlargest(5, 'Adoption_Spike_Potential')
            fig_spikes = go.Figure(data=[go.Bar(
                x=top_spikes['Adoption_Spike_Potential'],
                y=top_spikes['State'],
                orientation='h',
                marker_color='orange'
            )])
            fig_spikes.update_layout(title="Top 5 - Adoption Spike Potential")
            st.plotly_chart(fig_spikes, use_container_width=True)
        
        with col2:
            # Current vs predicted
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=dashboard.ml_predictions['EV_Sales_Quantity'],
                y=dashboard.ml_predictions['Predicted_Sales_24m'],
                mode='markers',
                text=dashboard.ml_predictions['State'],
                marker=dict(size=10)
            ))
            fig_comparison.update_layout(title="Current vs Predicted Sales")
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # ML insights
        st.subheader("ML Insights")
        avg_growth = dashboard.ml_predictions['Adoption_Spike_Potential'].mean()
        st.metric("Average Growth Potential", f"{avg_growth:.1f}%")
        
        st.dataframe(
            dashboard.ml_predictions[['State', 'EV_Sales_Quantity', 'Predicted_Sales_24m', 'Adoption_Spike_Potential']].round(1),
            use_container_width=True
        )
    
    elif section == "🗺️ Strategic Recommendations":
        st.header("🗺️ Strategic Recommendations")
        
        # Strategic priority
        strategic_data = dashboard.gap_analysis.copy()
        strategic_data['Strategic_Priority_Score'] = (
            strategic_data['Charging_Gap_Score'] * 0.6 + 
            dashboard.ml_predictions['Adoption_Spike_Potential'] * 0.4
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategic rankings
            top_strategic = strategic_data.nlargest(10, 'Strategic_Priority_Score')
            fig_strategic = go.Figure(data=[go.Bar(
                x=top_strategic['Strategic_Priority_Score'],
                y=top_strategic['State'],
                orientation='h',
                marker_color='green'
            )])
            fig_strategic.update_layout(title="Strategic Priority Rankings")
            st.plotly_chart(fig_strategic, use_container_width=True)
        
        with col2:
            # Investment recommendations
            investment_data = strategic_data.copy()
            investment_data['Recommended_Stations'] = np.where(
                investment_data['Priority'] == 'High',
                investment_data['Station_Count'] * 5,
                investment_data['Station_Count'] * 2
            )
            
            fig_investment = go.Figure(data=[go.Bar(
                x=investment_data['Recommended_Stations'],
                y=investment_data['State'],
                orientation='h',
                marker_color='blue'
            )])
            fig_investment.update_layout(title="Recommended New Stations")
            st.plotly_chart(fig_investment, use_container_width=True)
        
        # Action plan
        st.subheader("📋 Immediate Action Plan")
        high_priority = strategic_data[strategic_data['Priority'] == 'High'].head(3)
        
        for i, (_, row) in enumerate(high_priority.iterrows(), 1):
            st.write(f"**{i}. {row['State']}**")
            st.write(f"   • Current: {row['Station_Count']} stations")
            st.write(f"   • Market: {row['EV_Sales_Quantity']:,.0f} EVs")
            st.write(f"   • Recommended: {int(row['Station_Count'] * 5)} new stations")
            st.write("")
        
        # Summary metrics
        st.subheader("💡 Investment Summary")
        current_stations = strategic_data['Station_Count'].sum()
        recommended_stations = investment_data['Recommended_Stations'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Infrastructure", f"{current_stations:.0f}")
        with col2:
            st.metric("Recommended Expansion", f"{recommended_stations:.0f}")
        with col3:
            st.metric("New Stations Required", f"{recommended_stations - current_stations:.0f}")

if __name__ == "__main__":
    main()
