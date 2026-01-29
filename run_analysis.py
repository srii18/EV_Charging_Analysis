"""
Run EV Infrastructure Analysis
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import EV_SALES_FILE, CHARGING_STATIONS_FILE, GAP_ANALYSIS_OUTPUT
from utils.data_validation import clean_and_validate_data
from models.analysis import EVInfrastructureAnalyzer
from utils.logger import setup_logger

def main():
    """Run the complete EV infrastructure analysis"""
    logger = setup_logger()
    
    logger.info("🚀 Starting EV Infrastructure Analysis...")
    
    # Check if data files exist
    if not EV_SALES_FILE.exists():
        logger.error(f"EV sales data file not found: {EV_SALES_FILE}")
        return
    
    if not CHARGING_STATIONS_FILE.exists():
        logger.error(f"Charging stations data file not found: {CHARGING_STATIONS_FILE}")
        return
    
    # Load raw data
    ev_data = pd.read_csv(EV_SALES_FILE)
    stations_raw = pd.read_csv(CHARGING_STATIONS_FILE)
    
    logger.info(f"Loaded {len(ev_data):,} EV sales records")
    logger.info(f"Loaded {len(stations_raw):,} charging station records")
    
    # Clean and validate data
    try:
        ev_sales_clean, stations_clean = clean_and_validate_data(ev_data, stations_raw)
        logger.info(f"Cleaned EV sales data: {len(ev_sales_clean):,} records")
        logger.info(f"Cleaned stations data: {len(stations_clean):,} records")
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return
    
    # Perform gap analysis
    analyzer = EVInfrastructureAnalyzer(logger)
    gap_analysis = analyzer.perform_gap_analysis(ev_sales_clean, stations_clean)
    
    # Save results
    gap_analysis.to_csv(GAP_ANALYSIS_OUTPUT, index=False)
    logger.info(f"Analysis complete! Results saved to {GAP_ANALYSIS_OUTPUT}")
    
    # Print summary
    logger.info("\n📊 ANALYSIS SUMMARY:")
    logger.info(f"Total regions analyzed: {len(gap_analysis)}")
    
    priority_counts = gap_analysis['Recommendation_Tier'].value_counts()
    for tier, count in priority_counts.items():
        logger.info(f"{tier}: {count}")
    
    # Top 5 high priority regions
    high_priority = gap_analysis[gap_analysis['Recommendation_Tier'] == 'HIGH - Priority Investment']
    if len(high_priority) > 0:
        logger.info("\n🔥 TOP 5 HIGH PRIORITY REGIONS:")
        top_5 = high_priority.nlargest(5, 'Priority_Score')
        for _, row in top_5.iterrows():
            logger.info(f"  {row['State']} (Score: {row['Priority_Score']:.3f}, Recommended: {row['Recommended_New_Stations']:,.0f} stations)")
    
    logger.info("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
