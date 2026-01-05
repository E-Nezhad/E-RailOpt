import requests
import json
import pandas as pd

def get_railways_between_coords(start_lat, start_lon, end_lat, end_lon, buffer_deg=0.2):
    south = min(start_lat, end_lat) - buffer_deg
    north = max(start_lat, end_lat) + buffer_deg
    west = min(start_lon, end_lon) - buffer_deg
    east = max(start_lon, end_lon) + buffer_deg

    query = f"""
    [out:json][timeout:90];
    way["railway"="rail"]["usage"~"freight|main"]({south},{west},{north},{east});
    out geom;
    """

    url = "http://overpass-api.de/api/interpreter"
    response = requests.post(url, data={"data": query})
    response.raise_for_status()
    data = response.json()
    
    segments = [
        [(pt["lat"], pt["lon"]) for pt in element["geometry"]]
        for element in data["elements"]
        if "geometry" in element
    ]
    return segments

def save_as_json(segments, filename="railway_segments.json"):
    with open(filename, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"Saved {len(segments)} segments to {filename}")

def save_as_excel(segments, filename="railway_segments.xlsx"):
    # Flatten for Excel: each row is one coordinate point with segment index
    rows = []
    for i, segment in enumerate(segments):
        for lat, lon in segment:
            rows.append({"segment": i+1, "latitude": lat, "longitude": lon})
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"Saved {len(segments)} segments ({len(df)} points) to {filename}")

# # Example usage
# start_lat, start_lon = -33.98, 151.22   # Port Botany
# end_lat, end_lon = -33.13, 148.18       # Parkes

# segments = get_railways_between_coords(start_lat, start_lon, end_lat, end_lon)

# # Save as JSON (recommended if you want to keep segments grouped)
# save_as_json(segments)

# # Save as Excel (recommended for manual inspection)
# save_as_excel(segments)
