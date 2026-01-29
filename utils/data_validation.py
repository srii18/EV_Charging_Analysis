"""
Data validation utilities for ChargeSmart India
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config.config import YEARS_FOR_ANALYSIS, MAIN_VEHICLE_CATEGORIES

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def validate_ev_sales_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate EV sales dataset
    
    Args:
        df: DataFrame containing EV sales data
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_columns = ['Year', 'State', 'EV_Sales_Quantity', 'Vehicle_Category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'Year' in df.columns and not pd.api.types.is_numeric_dtype(df['Year']):
        errors.append("Year column should be numeric")
    
    if 'EV_Sales_Quantity' in df.columns and not pd.api.types.is_numeric_dtype(df['EV_Sales_Quantity']):
        errors.append("EV_Sales_Quantity column should be numeric")
    
    # Check for reasonable values
    if 'Year' in df.columns:
        invalid_years = df[~df['Year'].isin(range(2010, 2030))]
        if len(invalid_years) > 0:
            errors.append(f"Found {len(invalid_years)} records with invalid years")
    
    if 'EV_Sales_Quantity' in df.columns:
        negative_sales = df[df['EV_Sales_Quantity'] < 0]
        if len(negative_sales) > 0:
            errors.append(f"Found {len(negative_sales)} records with negative sales quantities")
    
    # Check for empty states
    if 'State' in df.columns:
        empty_states = df[df['State'].isna() | (df['State'] == '')]
        if len(empty_states) > 0:
            errors.append(f"Found {len(empty_states)} records with empty state names")
    
    # Check vehicle categories
    if 'Vehicle_Category' in df.columns:
        invalid_categories = df[~df['Vehicle_Category'].isin(MAIN_VEHICLE_CATEGORIES + ['Others'])]
        if len(invalid_categories) > 0:
            errors.append(f"Found {len(invalid_categories)} records with invalid vehicle categories")
    
    return len(errors) == 0, errors

def validate_charging_stations_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate charging stations dataset
    
    Args:
        df: DataFrame containing charging stations data
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_columns = ['State', 'Latitude', 'Longitude', 'Power (kW)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'Latitude' in df.columns and not pd.api.types.is_numeric_dtype(df['Latitude']):
        errors.append("Latitude column should be numeric")
    
    if 'Longitude' in df.columns and not pd.api.types.is_numeric_dtype(df['Longitude']):
        errors.append("Longitude column should be numeric")
    
    if 'Power (kW)' in df.columns and not pd.api.types.is_numeric_dtype(df['Power (kW)']):
        errors.append("Power (kW) column should be numeric")
    
    # Check for reasonable coordinate values
    if 'Latitude' in df.columns:
        invalid_lat = df[(df['Latitude'] < 6) | (df['Latitude'] > 38)]
        if len(invalid_lat) > 0:
            errors.append(f"Found {len(invalid_lat)} records with invalid latitude values (should be 6-38 for India)")
    
    if 'Longitude' in df.columns:
        invalid_lon = df[(df['Longitude'] < 68) | (df['Longitude'] > 98)]
        if len(invalid_lon) > 0:
            errors.append(f"Found {len(invalid_lon)} records with invalid longitude values (should be 68-98 for India)")
    
    if 'Power (kW)' in df.columns:
        negative_power = df[df['Power (kW)'] < 0]
        if len(negative_power) > 0:
            errors.append(f"Found {len(negative_power)} records with negative power values")
        
        zero_power = df[df['Power (kW)'] == 0]
        if len(zero_power) > 0:
            errors.append(f"Found {len(zero_power)} records with zero power values")
    
    # Check for empty states
    if 'State' in df.columns:
        empty_states = df[df['State'].isna() | (df['State'] == '')]
        if len(empty_states) > 0:
            errors.append(f"Found {len(empty_states)} records with empty state names")
    
    return len(errors) == 0, errors

def clean_and_validate_data(ev_sales_df: pd.DataFrame, stations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and validate both datasets
    
    Args:
        ev_sales_df: EV sales DataFrame
        stations_df: Charging stations DataFrame
    
    Returns:
        Tuple of cleaned (ev_sales_df, stations_df)
    
    Raises:
        DataValidationError: If validation fails
    """
    # Pre-clean the data first
    ev_sales_clean = ev_sales_df.copy()
    stations_clean = stations_df.copy()
    
    # Remove rows with missing critical data
    ev_sales_clean = ev_sales_clean.dropna(subset=['State', 'EV_Sales_Quantity'])
    stations_clean = stations_clean.dropna(subset=['State', 'Latitude', 'Longitude'])
    
    # Remove empty state names
    ev_sales_clean = ev_sales_clean[ev_sales_clean['State'].str.strip() != '']
    stations_clean = stations_clean[stations_clean['State'].str.strip() != '']
    
    # Validate EV sales data
    is_valid, errors = validate_ev_sales_data(ev_sales_clean)
    if not is_valid:
        raise DataValidationError(f"EV sales data validation failed: {errors}")
    
    # Validate charging stations data
    is_valid, errors = validate_charging_stations_data(stations_clean)
    if not is_valid:
        raise DataValidationError(f"Charging stations data validation failed: {errors}")
    
    # Clean EV sales data
    # Remove negative sales
    ev_sales_clean = ev_sales_clean[ev_sales_clean['EV_Sales_Quantity'] > 0]
    
    # Filter by years of interest
    ev_sales_clean = ev_sales_clean[ev_sales_clean['Year'].isin(YEARS_FOR_ANALYSIS)]
    
    # Filter by main vehicle categories
    ev_sales_clean = ev_sales_clean[ev_sales_clean['Vehicle_Category'].isin(MAIN_VEHICLE_CATEGORIES)]
    
    # Clean state names
    ev_sales_clean['State'] = ev_sales_clean['State'].str.strip().str.title()
    
    # Clean charging stations data
    stations_clean = stations_df.copy()
    
    # Remove duplicates
    stations_clean = stations_clean.drop_duplicates()
    
    # Remove rows with missing critical data
    stations_clean = stations_clean.dropna(subset=['State', 'Latitude', 'Longitude'])
    
    # Clean state names
    stations_clean['State'] = stations_clean['State'].str.strip().str.title()
    
    return ev_sales_clean, stations_clean
