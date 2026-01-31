"""
OpenChargeMap API Data Retrieval for India
Retrieves EV charging station data from OpenChargeMap API for India
"""

import requests
import json
import pandas as pd
import time
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenChargeMapRetriever:
    """Class to retrieve charging station data from OpenChargeMap API"""
    
    def __init__(self, base_url: str = "https://api.openchargemap.org/v3", api_key: Optional[str] = None):
        self.base_url = base_url
        self.session = requests.Session()
        headers = {
            'User-Agent': 'EV-Charging-Analysis-Script/1.0'
        }
        if api_key:
            headers['X-API-Key'] = api_key
        self.session.headers.update(headers)
    
    def get_india_charging_stations(
        self,
        max_results: int = 1000,
        compact: bool = True,
        verbose: bool = False,
        level_id: Optional[int] = None,
        connection_type_id: Optional[int] = None,
        status_type_id: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        distance: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve charging station data for India from OpenChargeMap API
        
        Args:
            max_results: Maximum number of results to retrieve (max 10,000)
            compact: Return smaller response with just IDs
            verbose: Include full reference data objects
            level_id: Filter by charging level (1=slow, 2=fast, 3=rapid)
            connection_type_id: Filter by connector type
            status_type_id: Filter by operational status
            latitude: Latitude for location-based search
            longitude: Longitude for location-based search
            distance: Search radius in km/miles
            
        Returns:
            List of charging station dictionaries
        """
        # Build query parameters
        params = {
            'output': 'json',
            'countryid': '106',  # India's country ID (corrected from 100)
            'maxresults': min(max_results, 10000),  # API limit is 10,000
            'compact': str(compact).lower(),
            'verbose': str(verbose).lower()
        }
        
        # Add optional filters
        if level_id is not None:
            params['levelid'] = level_id
        if connection_type_id is not None:
            params['connectiontypeid'] = connection_type_id
        if status_type_id is not None:
            params['statustypeid'] = status_type_id
        if latitude is not None:
            params['latitude'] = latitude
        if longitude is not None:
            params['longitude'] = longitude
        if distance is not None:
            params['distance'] = distance
            params['distanceunit'] = 'km'
        
        try:
            logger.info(f"Retrieving charging stations for India with params: {params}")
            response = self.session.get(f"{self.base_url}/poi", params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully retrieved {len(data)} charging stations")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving data: {e}")
            return []
    
    def get_all_india_stations(self, batch_size: int = 1000, max_total: int = 10000) -> List[Dict]:
        """
        Retrieve all charging stations for India using pagination
        
        Args:
            batch_size: Number of results per request
            max_total: Maximum total results to retrieve
            
        Returns:
            List of all charging station dictionaries
        """
        all_stations = []
        total_retrieved = 0
        
        while total_retrieved < max_total:
            remaining = max_total - total_retrieved
            current_batch_size = min(batch_size, remaining)
            
            # Get stations with offset using greaterthanid parameter
            last_id = all_stations[-1]['ID'] if all_stations else 0
            
            stations = self.get_india_charging_stations(
                max_results=current_batch_size,
                compact=True,
                verbose=False
            )
            
            if not stations:
                break
            
            # Filter stations with ID greater than last retrieved
            new_stations = [s for s in stations if s['ID'] > last_id]
            
            if not new_stations:
                break
            
            all_stations.extend(new_stations)
            total_retrieved += len(new_stations)
            
            logger.info(f"Retrieved {len(new_stations)} stations (total: {total_retrieved})")
            
            # Rate limiting - wait between requests
            time.sleep(1)
        
        logger.info(f"Total stations retrieved: {len(all_stations)}")
        return all_stations
    
    def save_to_csv(self, stations: List[Dict], filename: str = "india_ev_charging_stations.csv"):
        """Save charging stations data to CSV file"""
        if not stations:
            logger.warning("No stations data to save")
            return
        
        # Convert to DataFrame
        df = pd.json_normalize(stations)
        
        # Save to CSV
        output_path = f"data/raw/{filename}"
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        logger.info(f"Shape: {df.shape}")
        
        return output_path
    
    def get_station_summary(self, stations: List[Dict]) -> Dict:
        """Generate summary statistics for charging stations"""
        if not stations:
            return {}
        
        summary = {
            'total_stations': len(stations),
            'states': set(),
            'cities': set(),
            'operators': set(),
            'connection_types': set(),
            'status_types': set(),
            'charging_levels': set()
        }
        
        for station in stations:
            # Address info
            if 'AddressInfo' in station and station['AddressInfo']:
                addr = station['AddressInfo']
                if addr.get('StateOrProvince'):
                    summary['states'].add(addr['StateOrProvince'])
                if addr.get('Town'):
                    summary['cities'].add(addr['Town'])
            
            # Operator info
            if station.get('OperatorInfo') and station['OperatorInfo'].get('description'):
                summary['operators'].add(station['OperatorInfo']['description'])
            
            # Connections
            if 'Connections' in station:
                for conn in station['Connections']:
                    if conn.get('ConnectionType') and conn['ConnectionType'].get('description'):
                        summary['connection_types'].add(conn['ConnectionType']['description'])
                    if conn.get('Level') and conn['Level'].get('description'):
                        summary['charging_levels'].add(conn['Level']['description'])
                    if conn.get('StatusType') and conn['StatusType'].get('description'):
                        summary['status_types'].add(conn['StatusType']['description'])
        
        # Convert sets to counts
        summary['states_count'] = len(summary['states'])
        summary['cities_count'] = len(summary['cities'])
        summary['operators_count'] = len(summary['operators'])
        summary['connection_types_count'] = len(summary['connection_types'])
        summary['status_types_count'] = len(summary['status_types'])
        summary['charging_levels_count'] = len(summary['charging_levels'])
        
        return summary

def main():
    """Main function to retrieve India charging station data"""
    retriever = OpenChargeMapRetriever()
    
    logger.info("Starting retrieval of India EV charging stations data")
    
    # Get all stations (up to 10,000)
    stations = retriever.get_all_india_stations(batch_size=1000, max_total=10000)
    
    if stations:
        # Save to CSV
        csv_path = retriever.save_to_csv(stations)
        
        # Generate summary
        summary = retriever.get_station_summary(stations)
        
        logger.info("Data retrieval completed successfully!")
        logger.info(f"Summary: {summary}")
        
        # Display first few stations as sample
        logger.info("Sample stations:")
        for i, station in enumerate(stations[:3]):
            logger.info(f"  {i+1}. ID: {station.get('ID')}, "
                       f"Location: {station.get('AddressInfo', {}).get('Town', 'Unknown')}, "
                       f"State: {station.get('AddressInfo', {}).get('StateOrProvince', 'Unknown')}")
    else:
        logger.error("No stations data retrieved")

if __name__ == "__main__":
    main()
