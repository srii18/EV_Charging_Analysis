"""
Core analysis models for ChargeSmart India
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from config.config import SCORING_WEIGHTS, PRIORITY_THRESHOLDS, STATE_CORRECTIONS

class EVInfrastructureAnalyzer:
    """
    Main analyzer for EV charging infrastructure gaps
    """
    
    def __init__(self, logger=None):
        """
        Initialize the analyzer
        
        Args:
            logger: Logger instance for logging
        """
        self.logger = logger
        self.scaler = MinMaxScaler()
        self.gap_analysis = None
        
    def prepare_demand_metrics(self, ev_sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate demand metrics from EV sales data
        
        Args:
            ev_sales_df: Cleaned EV sales DataFrame
            
        Returns:
            DataFrame with demand metrics by state
        """
        if self.logger:
            self.logger.info("Calculating demand metrics...")
        
        # Total EV sales by state (cumulative demand)
        ev_demand = ev_sales_df.groupby('State').agg({
            'EV_Sales_Quantity': 'sum'
        }).reset_index()
        ev_demand.columns = ['State', 'Total_EV_Sales']
        
        # Recent sales (2023-2024) - indicates momentum
        recent_years = [2023, 2024]
        recent_sales = ev_sales_df[ev_sales_df['Year'].isin(recent_years)]
        recent_growth = recent_sales.groupby('State')['EV_Sales_Quantity'].sum().reset_index()
        recent_growth.columns = ['State', 'Recent_Sales_2023_24']
        
        # Calculate year-over-year growth rate
        yearly_sales = ev_sales_df[ev_sales_df['Year'].isin([2022, 2023, 2024])].groupby(['State', 'Year'])['EV_Sales_Quantity'].sum().reset_index()
        growth_rates = []
        
        for state in yearly_sales['State'].unique():
            state_data = yearly_sales[yearly_sales['State'] == state].sort_values('Year')
            if len(state_data) >= 2:
                # Calculate growth rate
                recent = state_data['EV_Sales_Quantity'].iloc[-1]
                older = state_data['EV_Sales_Quantity'].iloc[-2]
                growth_rate = ((recent - older) / (older + 1)) * 100  # +1 to avoid division by zero
            else:
                growth_rate = 0
            growth_rates.append({'State': state, 'Growth_Rate_%': growth_rate})
        
        growth_df = pd.DataFrame(growth_rates)
        
        # Merge all demand metrics
        demand_metrics = ev_demand.merge(recent_growth, on='State', how='left')
        demand_metrics = demand_metrics.merge(growth_df, on='State', how='left')
        demand_metrics['Recent_Sales_2023_24'] = demand_metrics['Recent_Sales_2023_24'].fillna(0)
        demand_metrics['Growth_Rate_%'] = demand_metrics['Growth_Rate_%'].fillna(0)
        
        if self.logger:
            self.logger.info(f"Calculated demand metrics for {len(demand_metrics)} states")
        
        return demand_metrics
    
    def prepare_supply_metrics(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate supply metrics from charging stations data
        
        Args:
            stations_df: Cleaned charging stations DataFrame
            
        Returns:
            DataFrame with supply metrics by state
        """
        if self.logger:
            self.logger.info("Calculating supply metrics...")
        
        # Station count and capacity by state
        supply_metrics = stations_df.groupby('State').agg({
            'Station Name': 'count',
            'Power (kW)': 'sum',
            'City': 'nunique'
        }).reset_index()
        supply_metrics.columns = ['State', 'Total_Stations', 'Total_Capacity_kW', 'Cities_Covered']
        
        # Average power per station
        supply_metrics['Avg_Power_per_Station'] = supply_metrics['Total_Capacity_kW'] / supply_metrics['Total_Stations']
        
        # Count stations by connector type
        connector_counts = stations_df.groupby(['State', 'Connector Type']).size().reset_index(name='count')
        fast_charging = connector_counts[connector_counts['Connector Type'].str.contains('CCS|CHAdeMO', na=False, case=False)]
        fast_charging_by_state = fast_charging.groupby('State')['count'].sum().reset_index()
        fast_charging_by_state.columns = ['State', 'Fast_Charging_Stations']
        
        supply_metrics = supply_metrics.merge(fast_charging_by_state, on='State', how='left')
        supply_metrics['Fast_Charging_Stations'] = supply_metrics['Fast_Charging_Stations'].fillna(0)
        
        if self.logger:
            self.logger.info(f"Calculated supply metrics for {len(supply_metrics)} states")
        
        return supply_metrics
    
    def normalize_score(self, series: pd.Series, reverse: bool = False, clip_outliers: bool = True) -> pd.Series:
        """
        Normalize series to 0-100 scale
        
        Args:
            series: Series to normalize
            reverse: If True, reverse the score (higher original = lower normalized)
            clip_outliers: If True, clip extreme outliers at 95th percentile
            
        Returns:
            Normalized series
        """
        if clip_outliers:
            # Clip extreme outliers at 95th percentile
            series = series.clip(upper=series.quantile(0.95))
        
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series([50.0] * len(series), index=series.index)
        
        normalized = ((series - min_val) / (max_val - min_val)) * 100
        
        if reverse:
            normalized = 100 - normalized
        
        return normalized
    
    def calculate_priority_scores(self, gap_analysis: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate priority scores based on gap analysis using notebook methodology
        
        Args:
            gap_analysis: DataFrame with demand and supply metrics
            
        Returns:
            DataFrame with priority scores and recommendations
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Create a copy for scoring
        scoring_data = gap_analysis.copy()
        
        # Handle missing values and infinities
        scoring_data = scoring_data.fillna(0)
        scoring_data = scoring_data.replace([np.inf, -np.inf], 0)
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Normalize EV sales (demand)
        if scoring_data['Total_EV_Sales'].std() > 0:
            scoring_data['EV_Sales_Normalized'] = scaler.fit_transform(scoring_data[['Total_EV_Sales']])
        else:
            scoring_data['EV_Sales_Normalized'] = 0
            
        # Normalize station density (inverse for infrastructure deficit)
        if scoring_data['EVs_per_Station'].std() > 0:
            # Higher EVs per Station = worse infrastructure, so we normalize and invert
            scoring_data['Station_Density_Normalized'] = scaler.fit_transform(scoring_data[['EVs_per_Station']])
        else:
            scoring_data['Station_Density_Normalized'] = 0
        
        # Calculate charging gap score using notebook methodology (60-40 split)
        scoring_data['Priority_Score'] = (
            scoring_data['EV_Sales_Normalized'] * 0.6 +  # 60% weight to demand
            scoring_data['Station_Density_Normalized'] * 0.4  # 40% weight to infrastructure deficit
        )
        
        # Add priority categories using notebook thresholds (3 categories)
        scoring_data['Recommendation_Tier'] = pd.cut(
            scoring_data['Priority_Score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW - Monitor & Plan', 'MEDIUM - Strategic Development', 'HIGH - Priority Investment']
        )
        
        # Add interpretation
        def get_interpretation(row):
            score = row['Priority_Score']
            if score >= 0.6:
                return "Urgent: High demand, low infrastructure"
            elif score >= 0.3:
                return "Moderate: Balanced growth needed"
            else:
                return "Stable: Adequate infrastructure"
        
        scoring_data['Interpretation'] = scoring_data.apply(get_interpretation, axis=1)
        
        # Calculate recommended stations (simplified approach)
        scoring_data['Recommended_New_Stations'] = np.where(
            scoring_data['Total_Stations'] == 0,
            scoring_data['Total_EV_Sales'] / 1000,  # 1 station per 1000 EVs for states with no infrastructure
            scoring_data['Total_EV_Sales'] / 2000   # 1 station per 2000 EVs for states with some infrastructure
        ).round(0).astype(int)
        
        # Ensure minimum recommendations
        scoring_data['Recommended_New_Stations'] = scoring_data['Recommended_New_Stations'].clip(lower=1)
        
        return scoring_data
    
    def perform_gap_analysis(self, ev_sales_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete demand-supply gap analysis
        
        Args:
            ev_sales_df: Cleaned EV sales DataFrame
            stations_df: Cleaned charging stations DataFrame
            
        Returns:
            Complete gap analysis DataFrame
        """
        if self.logger:
            self.logger.info("Starting gap analysis...")
        
        # Apply state name corrections
        stations_df['State'] = stations_df['State'].replace(STATE_CORRECTIONS)
        
        # Calculate demand and supply metrics
        demand_metrics = self.prepare_demand_metrics(ev_sales_df)
        supply_metrics = self.prepare_supply_metrics(stations_df)
        
        # Merge demand and supply
        gap_analysis = demand_metrics.merge(supply_metrics, on='State', how='left')
        
        # Fill NaN for states with no infrastructure
        infrastructure_cols = ['Total_Stations', 'Total_Capacity_kW', 'Cities_Covered', 'Fast_Charging_Stations']
        for col in infrastructure_cols:
            gap_analysis[col] = gap_analysis[col].fillna(0)
        
        # Calculate key gap metrics
        gap_analysis['EVs_per_Station'] = gap_analysis.apply(
            lambda row: row['Total_EV_Sales'] / row['Total_Stations'] if row['Total_Stations'] > 0 
            else row['Total_EV_Sales'], axis=1
        )
        
        gap_analysis['Station_Density_per_1000_EVs'] = gap_analysis.apply(
            lambda row: (row['Total_Stations'] / row['Total_EV_Sales']) * 1000 if row['Total_EV_Sales'] > 0 
            else 0, axis=1
        )
        
        gap_analysis['Has_Infrastructure'] = (gap_analysis['Total_Stations'] > 0).astype(int)
        
        # Calculate priority scores
        gap_analysis = self.calculate_priority_scores(gap_analysis)
        
        # Sort by priority
        gap_analysis = gap_analysis.sort_values('Priority_Score', ascending=False).reset_index(drop=True)
        
        # Calculate recommended new stations
        def calculate_recommended_stations(row):
            if row['Priority_Score'] >= PRIORITY_THRESHOLDS['critical']:
                # Critical: aim for 1 station per 1000 EVs
                target_ratio = 1000
            elif row['Priority_Score'] >= PRIORITY_THRESHOLDS['high']:
                # High: aim for 1 station per 1500 EVs
                target_ratio = 1500
            elif row['Priority_Score'] >= PRIORITY_THRESHOLDS['medium']:
                # Medium: aim for 1 station per 2000 EVs
                target_ratio = 2000
            else:
                # Low: aim for 1 station per 3000 EVs
                target_ratio = 3000
            
            current_stations = row['Total_Stations']
            needed_stations = max(0, (row['Total_EV_Sales'] / target_ratio) - current_stations)
            
            return int(needed_stations)
        
        gap_analysis['Recommended_New_Stations'] = gap_analysis.apply(calculate_recommended_stations, axis=1)
        
        self.gap_analysis = gap_analysis
        
        if self.logger:
            self.logger.info(f"Gap analysis complete for {len(gap_analysis)} states")
            self.logger.info(f"Total recommended new stations: {gap_analysis['Recommended_New_Stations'].sum()}")
        
        return gap_analysis
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the analysis
        
        Returns:
            Dictionary with summary statistics
        """
        if self.gap_analysis is None:
            return {}
        
        summary = {
            'total_states_analyzed': len(self.gap_analysis),
            'states_with_infrastructure': (self.gap_analysis['Total_Stations'] > 0).sum(),
            'states_without_infrastructure': (self.gap_analysis['Total_Stations'] == 0).sum(),
            'total_ev_sales': self.gap_analysis['Total_EV_Sales'].sum(),
            'total_current_stations': self.gap_analysis['Total_Stations'].sum(),
            'total_recommended_stations': self.gap_analysis['Recommended_New_Stations'].sum(),
            'priority_distribution': self.gap_analysis['Recommendation_Tier'].value_counts().to_dict(),
            'top_5_states': self.gap_analysis.nlargest(5, 'Priority_Score')['State'].tolist()
        }
        
        return summary
