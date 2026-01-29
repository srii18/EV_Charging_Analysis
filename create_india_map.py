"""
Create accurate India state boundaries for choropleth map
"""

import json
from pathlib import Path

# Simplified India state boundaries (approximate coordinates)
# This creates a more accurate representation of Indian states
india_state_boundaries = {
    "type": "FeatureCollection",
    "features": [
        # North India
        {
            "type": "Feature",
            "properties": {"name": "Jammu & Kashmir"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.4, 37.1], [80.3, 37.1], [80.3, 32.5], [74.4, 32.5], [74.4, 37.1]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Punjab"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[73.9, 32.5], [76.8, 32.5], [76.8, 29.9], [73.9, 29.9], [73.9, 32.5]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Haryana"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.4, 30.4], [77.6, 30.4], [77.6, 27.7], [74.4, 27.7], [74.4, 30.4]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Uttar Pradesh"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[77.0, 30.0], [84.4, 30.0], [84.4, 23.9], [77.0, 23.9], [77.0, 30.0]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Rajasthan"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[69.5, 30.0], [78.2, 30.0], [78.2, 23.0], [69.5, 23.0], [69.5, 30.0]]]
            }
        },
        # West India
        {
            "type": "Feature",
            "properties": {"name": "Gujarat"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[68.4, 24.7], [74.5, 24.7], [74.5, 20.1], [68.4, 20.1], [68.4, 24.7]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Maharashtra"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.6, 21.5], [80.9, 21.5], [80.9, 15.6], [72.6, 15.6], [72.6, 21.5]]]
            }
        },
        # Central India
        {
            "type": "Feature",
            "properties": {"name": "Madhya Pradesh"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.0, 26.5], [84.4, 26.5], [84.4, 21.0], [74.0, 21.0], [74.0, 26.5]]]
            }
        },
        # East India
        {
            "type": "Feature",
            "properties": {"name": "Bihar"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[83.3, 27.5], [88.3, 27.5], [88.3, 24.3], [83.3, 24.3], [83.3, 27.5]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "West Bengal"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[85.8, 27.2], [89.9, 27.2], [89.9, 21.5], [85.8, 21.5], [85.8, 27.2]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Odisha"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[81.4, 22.5], [87.5, 22.5], [87.5, 17.8], [81.4, 17.8], [81.4, 22.5]]]
            }
        },
        # South India
        {
            "type": "Feature",
            "properties": {"name": "Karnataka"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.0, 18.5], [78.5, 18.5], [78.5, 11.6], [74.0, 11.6], [74.0, 18.5]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Tamil Nadu"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.2, 13.5], [80.3, 13.5], [80.3, 8.1], [76.2, 8.1], [76.2, 13.5]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Kerala"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.9, 12.7], [77.4, 12.7], [77.4, 8.2], [74.9, 8.2], [74.9, 12.7]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Andhra Pradesh"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.8, 19.2], [84.7, 19.2], [84.7, 12.6], [76.8, 12.6], [76.8, 19.2]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Telangana"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[77.3, 19.5], [81.2, 19.5], [81.2, 15.8], [77.3, 15.8], [77.3, 19.5]]]
            }
        },
        # Northeast India
        {
            "type": "Feature",
            "properties": {"name": "Assam"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[89.7, 27.5], [96.0, 27.5], [96.0, 24.1], [89.7, 24.1], [89.7, 27.5]]]
            }
        },
        # Union Territories
        {
            "type": "Feature",
            "properties": {"name": "Delhi"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.8, 28.8], [77.3, 28.8], [77.3, 28.4], [76.8, 28.4], [76.8, 28.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Goa"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[73.7, 15.7], [74.3, 15.7], [74.3, 15.1], [73.7, 15.1], [73.7, 15.7]]]
            }
        }
    ]
}

# Save the GeoJSON data
output_path = Path("data/india_states.geojson")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(india_state_boundaries, f, indent=2)

print(f"India state boundaries saved to: {output_path}")
print(f"Created {len(india_state_boundaries['features'])} state boundaries")
