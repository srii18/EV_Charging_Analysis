"""
Setup script to download and prepare data for ChargeSmart India
"""

import pandas as pd
import kagglehub
from pathlib import Path
import shutil

def setup_data():
    """Download and setup data files in the correct location"""
    
    print("🚀 Setting up data for ChargeSmart India...")
    
    # Create data directories
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download EV charging stations data
        print("📥 Downloading EV charging stations data...")
        path1 = kagglehub.dataset_download("pranjal9091/ev-charging-stations-in-india-simplified-2025")
        print(f"   Downloaded to: {path1}")
        
        # Copy to data/raw
        stations_source = Path(path1) / "Indian_EV_Stations_Simplified.csv"
        stations_dest = raw_data_dir / "Indian_EV_Stations_Simplified.csv"
        shutil.copy2(stations_source, stations_dest)
        print(f"   ✅ Copied to: {stations_dest}")
        
        # Download EV sales data
        print("📥 Downloading EV sales data...")
        path2 = kagglehub.dataset_download("mafzal19/electric-vehicle-sales-by-state-in-india")
        print(f"   Downloaded to: {path2}")
        
        # Copy to data/raw
        sales_source = Path(path2) / "EV_Dataset.csv"
        sales_dest = raw_data_dir / "EV_Dataset.csv"
        shutil.copy2(sales_source, sales_dest)
        print(f"   ✅ Copied to: {sales_dest}")
        
        print("\n🎉 Data setup complete!")
        print(f"📁 Files are now available in: {raw_data_dir.absolute()}")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Run: streamlit run app/dashboard.py")
        
    except Exception as e:
        print(f"❌ Error setting up data: {e}")
        print("\nAlternative: Manually copy the CSV files to data/raw/ directory")

if __name__ == "__main__":
    setup_data()
