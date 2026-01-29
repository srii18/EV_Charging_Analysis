"""
Configuration file for ChargeSmart India
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# File paths
EV_SALES_FILE = RAW_DATA_DIR / "EV_Dataset.csv"
CHARGING_STATIONS_FILE = RAW_DATA_DIR / "Indian_EV_Stations_Simplified.csv"

# Output files
GAP_ANALYSIS_OUTPUT = PROCESSED_DATA_DIR / "gap_analysis_complete.csv"
PRIORITY_STATES_OUTPUT = PROCESSED_DATA_DIR / "priority_states.csv"
CITY_RECOMMENDATIONS_OUTPUT = PROCESSED_DATA_DIR / "city_recommendations.csv"

# Analysis parameters
YEARS_FOR_ANALYSIS = [2022, 2023, 2024]
MAIN_VEHICLE_CATEGORIES = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus']

# Scoring weights
SCORING_WEIGHTS = {
    'demand_score': 0.25,
    'growth_score': 0.25,
    'momentum_score': 0.15,
    'infrastructure_gap_score': 0.25,
    'infrastructure_penalty': 0.10
}

# Priority thresholds
PRIORITY_THRESHOLDS = {
    'critical': 75,
    'high': 60,
    'medium': 40,
    'low': 0
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (14, 8),
    'font_size': 10,
    'style': 'seaborn-v0_8-whitegrid',
    'palette': 'Set2'
}

# State name corrections
STATE_CORRECTIONS = {
    'Keraka': 'Kerala',
    'Keral': 'Kerala', 
    'Lerala': 'Kerala',
    'uttar pradesh': 'Uttar Pradesh',
    'Tamil Nadu ': 'Tamil Nadu'
}

# State capital coordinates
STATE_CAPITALS = {
    'Uttar Pradesh': [26.8467, 80.9462],
    'Maharashtra': [19.0760, 72.8777],
    'Delhi': [28.6139, 77.2090],
    'Rajasthan': [26.9124, 75.7873],
    'Bihar': [25.5941, 85.1376],
    'Karnataka': [12.9716, 77.5946],
    'Gujarat': [23.0225, 72.5714],
    'West Bengal': [22.5726, 88.3639],
    'Tamil Nadu': [13.0827, 80.2707],
    'Madhya Pradesh': [23.2599, 77.4126],
    'Haryana': [28.4595, 77.0266],
    'Punjab': [30.7333, 76.7794],
    'Andhra Pradesh': [16.5062, 80.6480],
    'Telangana': [17.3850, 78.4867],
    'Kerala': [8.5241, 76.9366],
    'Odisha': [20.2961, 85.8245],
    'Assam': [26.2006, 92.9376],
    'Jharkhand': [23.6102, 85.2799],
    'Chhattisgarh': [21.2514, 81.6296],
    'Uttarakhand': [30.0668, 79.0193],
    'Himachal Pradesh': [31.1048, 77.1734],
    'Goa': [15.2993, 74.1240],
    'Andaman & Nicobar Island': [11.6400, 92.7300],
    'Arunachal Pradesh': [27.0844, 93.6053],
    'Assam': [26.2006, 92.9376],
    'Bihar': [25.5941, 85.1376],
    'Chandigarh': [30.7333, 76.7794],
    'Chhattisgarh': [21.2514, 81.6296],
    'Dnh And Dd': [20.1807, 73.2207],
    'Goa': [15.2993, 74.1240],
    'Gujarat': [23.0225, 72.5714],
    'Haryana': [28.4595, 77.0266],
    'Himachal Pradesh': [31.1048, 77.1734],
    'Jammu And Kashmir': [34.0837, 74.7973],
    'Jharkhand': [23.6102, 85.2799],
    'Karnataka': [12.9716, 77.5946],
    'Kerala': [8.5241, 76.9366],
    'Ladakh': [34.1526, 77.5771],
    'Madhya Pradesh': [23.2599, 77.4126],
    'Maharashtra': [19.0760, 72.8777],
    'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.5788, 91.8933],
    'Mizoram': [23.1645, 92.9376],
    'Nagaland': [26.1584, 94.5624],
    'Odisha': [20.2961, 85.8245],
    'Puducherry': [11.9416, 79.8083],
    'Punjab': [30.7333, 76.7794],
    'Rajasthan': [26.9124, 75.7873],
    'Sikkim': [27.3314, 88.6138],
    'Tamil Nadu': [13.0827, 80.2707],
    'Telangana': [17.3850, 78.4867],
    'Tripura': [23.8315, 91.2868],
    'Uttar Pradesh': [26.8467, 80.9462],
    'Uttarakhand': [30.0668, 79.0193],
    'West Bengal': [22.5726, 88.3639]
}

# Tier colors for visualization (3-tier system from notebooks)
TIER_COLORS = {
    'HIGH - Priority Investment': '#d62728',
    'MEDIUM - Strategic Development': '#ff7f0e',
    'LOW - Monitor & Plan': '#2ca02c'
}
