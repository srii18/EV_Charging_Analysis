"""
Test suite for ChargeSmart India Analysis
"""

import pytest
import pandas as pd
import numpy as np
from models.analysis import EVInfrastructureAnalyzer
from utils.data_validation import clean_and_validate_data, validate_ev_sales_data, validate_charging_stations_data

class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_ev_sales_data_valid(self):
        """Test validation with valid EV sales data"""
        valid_data = pd.DataFrame({
            'Year': [2022, 2023, 2024],
            'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu'],
            'EV_Sales_Quantity': [1000, 1500, 2000],
            'Vehicle_Category': ['2-Wheelers', '3-Wheelers', '4-Wheelers']
        })
        
        is_valid, errors = validate_ev_sales_data(valid_data)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_ev_sales_data_invalid(self):
        """Test validation with invalid EV sales data"""
        invalid_data = pd.DataFrame({
            'Year': [2022, 2030, 2024],  # Invalid year
            'State': ['Maharashtra', None, 'Tamil Nadu'],  # Missing state
            'EV_Sales_Quantity': [1000, -500, 2000],  # Negative sales
            'Vehicle_Category': ['2-Wheelers', 'Invalid', '4-Wheelers']  # Invalid category
        })
        
        is_valid, errors = validate_ev_sales_data(invalid_data)
        assert is_valid == False
        assert len(errors) > 0
    
    def test_validate_charging_stations_data_valid(self):
        """Test validation with valid charging stations data"""
        valid_data = pd.DataFrame({
            'State': ['Maharashtra', 'Karnataka'],
            'Latitude': [19.0760, 12.9716],
            'Longitude': [72.8777, 77.5946],
            'Power (kW)': [50.0, 60.0]
        })
        
        is_valid, errors = validate_charging_stations_data(valid_data)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_charging_stations_data_invalid(self):
        """Test validation with invalid charging stations data"""
        invalid_data = pd.DataFrame({
            'State': ['Maharashtra', None],  # Missing state
            'Latitude': [19.0760, 100.0],  # Invalid latitude
            'Longitude': [72.8777, 77.5946],
            'Power (kW)': [50.0, -10.0]  # Negative power
        })
        
        is_valid, errors = validate_charging_stations_data(invalid_data)
        assert is_valid == False
        assert len(errors) > 0

class TestEVInfrastructureAnalyzer:
    """Test EVInfrastructureAnalyzer class"""
    
    @pytest.fixture
    def sample_ev_sales(self):
        """Sample EV sales data for testing"""
        return pd.DataFrame({
            'Year': [2022, 2023, 2024] * 3,
            'State': ['Maharashtra'] * 3 + ['Karnataka'] * 3 + ['Tamil Nadu'] * 3,
            'EV_Sales_Quantity': [1000, 1200, 1500, 800, 1000, 1300, 600, 800, 1100],
            'Vehicle_Category': ['2-Wheelers'] * 9
        })
    
    @pytest.fixture
    def sample_stations(self):
        """Sample charging stations data for testing"""
        return pd.DataFrame({
            'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu'],
            'Station Name': ['Station 1', 'Station 2', 'Station 3'],
            'Latitude': [19.0760, 12.9716, 13.0827],
            'Longitude': [72.8777, 77.5946, 80.2707],
            'Power (kW)': [50.0, 60.0, 70.0],
            'Connector Type': ['CCS', 'Type 2', 'CHAdeMO'],
            'City': ['Mumbai', 'Bangalore', 'Chennai']
        })
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = EVInfrastructureAnalyzer()
        assert analyzer.scaler is not None
        assert analyzer.gap_analysis is None
    
    def test_prepare_demand_metrics(self, sample_ev_sales):
        """Test demand metrics calculation"""
        analyzer = EVInfrastructureAnalyzer()
        demand_metrics = analyzer.prepare_demand_metrics(sample_ev_sales)
        
        assert len(demand_metrics) == 3  # 3 states
        assert 'Total_EV_Sales' in demand_metrics.columns
        assert 'Recent_Sales_2023_24' in demand_metrics.columns
        assert 'Growth_Rate_%' in demand_metrics.columns
        
        # Check calculations
        maharashtra_data = demand_metrics[demand_metrics['State'] == 'Maharashtra'].iloc[0]
        assert maharashtra_data['Total_EV_Sales'] == 3700  # 1000 + 1200 + 1500
        assert maharashtra_data['Recent_Sales_2023_24'] == 2700  # 1200 + 1500
    
    def test_prepare_supply_metrics(self, sample_stations):
        """Test supply metrics calculation"""
        analyzer = EVInfrastructureAnalyzer()
        supply_metrics = analyzer.prepare_supply_metrics(sample_stations)
        
        assert len(supply_metrics) == 3  # 3 states
        assert 'Total_Stations' in supply_metrics.columns
        assert 'Total_Capacity_kW' in supply_metrics.columns
        assert 'Cities_Covered' in supply_metrics.columns
        
        # Check calculations
        maharashtra_data = supply_metrics[supply_metrics['State'] == 'Maharashtra'].iloc[0]
        assert maharashtra_data['Total_Stations'] == 1
        assert maharashtra_data['Total_Capacity_kW'] == 50.0
        assert maharashtra_data['Cities_Covered'] == 1
    
    def test_normalize_score(self):
        """Test score normalization"""
        analyzer = EVInfrastructureAnalyzer()
        
        # Test normal case
        series = pd.Series([10, 20, 30, 40, 50])
        normalized = analyzer.normalize_score(series)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 100.0
        assert len(normalized) == len(series)
        
        # Test reverse case
        normalized_reverse = analyzer.normalize_score(series, reverse=True)
        assert normalized_reverse.iloc[0] == 100.0  # Highest original becomes lowest normalized
        assert normalized_reverse.iloc[-1] == 0.0  # Lowest original becomes highest normalized
        
        # Test uniform case
        uniform_series = pd.Series([10, 10, 10, 10, 10])
        normalized_uniform = analyzer.normalize_score(uniform_series)
        assert all(normalized_uniform == 50.0)
    
    def test_perform_gap_analysis(self, sample_ev_sales, sample_stations):
        """Test complete gap analysis"""
        analyzer = EVInfrastructureAnalyzer()
        gap_analysis = analyzer.perform_gap_analysis(sample_ev_sales, sample_stations)
        
        assert len(gap_analysis) == 3  # 3 states
        assert 'Priority_Score' in gap_analysis.columns
        assert 'Recommendation_Tier' in gap_analysis.columns
        assert 'Recommended_New_Stations' in gap_analysis.columns
        
        # Check that scores are in valid range
        assert gap_analysis['Priority_Score'].between(0, 100).all()
        
        # Check that states are sorted by priority
        assert gap_analysis.iloc[0]['Priority_Score'] >= gap_analysis.iloc[1]['Priority_Score']
    
    def test_get_summary_statistics(self, sample_ev_sales, sample_stations):
        """Test summary statistics"""
        analyzer = EVInfrastructureAnalyzer()
        analyzer.perform_gap_analysis(sample_ev_sales, sample_stations)
        
        summary = analyzer.get_summary_statistics()
        
        assert 'total_states_analyzed' in summary
        assert 'states_with_infrastructure' in summary
        assert 'total_ev_sales' in summary
        assert 'total_current_stations' in summary
        assert 'total_recommended_stations' in summary
        assert 'priority_distribution' in summary
        assert 'top_5_states' in summary
        
        assert summary['total_states_analyzed'] == 3
        assert summary['states_with_infrastructure'] == 3
        assert summary['total_ev_sales'] > 0

class TestIntegration:
    """Integration tests"""
    
    def test_clean_and_validate_data(self):
        """Test data cleaning and validation integration"""
        # Sample data with some issues
        ev_sales = pd.DataFrame({
            'Year': [2022, 2023, 2024, 2022, 2023],
            'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Maharashtra', 'Karnataka'],
            'EV_Sales_Quantity': [1000, 1500, 2000, -500, 0],  # Include negative and zero
            'Vehicle_Category': ['2-Wheelers', '3-Wheelers', '4-Wheelers', '2-Wheelers', 'Others']
        })
        
        stations = pd.DataFrame({
            'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Maharashtra'],  # Duplicate
            'Station Name': ['Station 1', 'Station 2', 'Station 3', 'Station 1'],  # Duplicate
            'Latitude': [19.0760, 12.9716, 13.0827, 19.0760],
            'Longitude': [72.8777, 77.5946, 80.2707, 72.8777],
            'Power (kW)': [50.0, 60.0, 70.0, 50.0],
            'Connector Type': ['CCS', 'Type 2', 'CHAdeMO', 'CCS'],
            'City': ['Mumbai', 'Bangalore', 'Chennai', 'Mumbai']
        })
        
        ev_clean, stations_clean = clean_and_validate_data(ev_sales, stations)
        
        # Check that negative sales were removed
        assert (ev_clean['EV_Sales_Quantity'] >= 0).all()
        
        # Check that zero sales were removed (should be filtered out by validation)
        assert (ev_clean['EV_Sales_Quantity'] > 0).all()
        
        # Check that duplicates were removed from stations
        assert len(stations_clean) < len(stations)
        
        # Check that state names are properly formatted
        assert all(ev_clean['State'].str.istitle())
        assert all(stations_clean['State'].str.istitle())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
