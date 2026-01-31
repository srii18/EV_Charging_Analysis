"""
Test OpenChargeMap API to understand available data
"""

import requests
import json

def test_api():
    """Test various API endpoints to understand data structure"""
    
    base_url = "https://api.openchargemap.org/v3"
    
    # Test 1: Get reference data to see available countries
    print("=== Getting Reference Data ===")
    try:
        response = requests.get(f"{base_url}/referencedata")
        if response.status_code == 200:
            data = response.json()
            
            # Find India in countries
            india = None
            for country in data.get('Countries', []):
                if country.get('ISOCode') == 'IN' or 'India' in country.get('Title', ''):
                    india = country
                    break
            
            if india:
                print(f"Found India: ID={india.get('ID')}, Title={india.get('Title')}, ISOCode={india.get('ISOCode')}")
            else:
                print("India not found in reference data")
                print("Available countries with 'India' in name:")
                for country in data.get('Countries', []):
                    if 'india' in country.get('Title', '').lower():
                        print(f"  ID={country.get('ID')}, Title={country.get('Title')}")
                
                print("First 10 countries:")
                for i, country in enumerate(data.get('Countries', [])[:10]):
                    print(f"  {i+1}. ID={country.get('ID')}, Title={country.get('Title')}, ISOCode={country.get('ISOCode')}")
        else:
            print(f"Error getting reference data: {response.status_code}")
    except Exception as e:
        print(f"Exception getting reference data: {e}")
    
    print("\n=== Testing POI endpoint without country filter ===")
    try:
        params = {
            'output': 'json',
            'maxresults': 10,
            'compact': 'true',
            'verbose': 'false'
        }
        response = requests.get(f"{base_url}/poi", params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Retrieved {len(data)} stations without country filter")
            
            # Check countries in results
            countries = set()
            for station in data:
                if station.get('AddressInfo') and station['AddressInfo'].get('Country'):
                    countries.add(station['AddressInfo']['Country'].get('Title', 'Unknown'))
            
            print(f"Countries in sample: {list(countries)}")
            
            # Show sample station
            if data:
                sample = data[0]
                print(f"\nSample station:")
                print(f"  ID: {sample.get('ID')}")
                print(f"  Title: {sample.get('AddressInfo', {}).get('Title', 'N/A')}")
                print(f"  Country: {sample.get('AddressInfo', {}).get('Country', {}).get('Title', 'N/A')}")
                print(f"  ISO Code: {sample.get('AddressInfo', {}).get('Country', {}).get('ISOCode', 'N/A')}")
        else:
            print(f"Error getting POI data: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Exception getting POI data: {e}")
    
    print("\n=== Testing with different India country IDs ===")
    # Try different possible India IDs
    possible_india_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    
    for country_id in possible_india_ids:
        try:
            params = {
                'output': 'json',
                'countryid': str(country_id),
                'maxresults': 5,
                'compact': 'true',
                'verbose': 'false'
            }
            response = requests.get(f"{base_url}/poi", params=params)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    print(f"Country ID {country_id}: Found {len(data)} stations")
                    # Check first station's country
                    if data[0].get('AddressInfo') and data[0]['AddressInfo'].get('Country'):
                        country = data[0]['AddressInfo']['Country']
                        print(f"  Sample country: {country.get('Title')} (ISO: {country.get('ISOCode')})")
                        if 'India' in country.get('Title', '') or country.get('ISOCode') == 'IN':
                            print(f"  *** This looks like India! ***")
                            break
            else:
                print(f"Country ID {country_id}: Error {response.status_code}")
        except Exception as e:
            print(f"Country ID {country_id}: Exception {e}")

if __name__ == "__main__":
    test_api()
