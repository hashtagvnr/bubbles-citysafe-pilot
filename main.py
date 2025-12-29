import requests
import pandas as pd
import time

# --- 1. DEFINE TARGET LOCATIONS (The Probes) ---
# We pick the center points of every major area in Gloucestershire
locations = {
    "Gloucester": {"lat": "51.8642", "lng": "-2.2386"},
    "Cheltenham": {"lat": "51.9001", "lng": "-2.0874"},
    "Stroud": {"lat": "51.7457", "lng": "-2.2178"},
    "Tewkesbury": {"lat": "51.9934", "lng": "-2.1607"},
    "Cirencester": {"lat": "51.7171", "lng": "-1.9638"},
    "Forest of Dean (Coleford)": {"lat": "51.7942", "lng": "-2.6167"},
    "Dursley": {"lat": "51.6811", "lng": "-2.3533"}
}

# --- 2. DEFINE DATE RANGE (The Time Machine) ---
# Jan 2023 to Aug 2025
months = [
    f"{year}-{month:02d}" 
    for year in range(2023, 2026) 
    for month in range(1, 13)
    if not (year == 2025 and month > 9) # Stop at Sep 2025
]

all_crimes = []

print(f"🚀 Starting County-Wide Extraction for {len(locations)} towns...")

# --- 3. THE DOUBLE LOOP ---
for town_name, coords in locations.items():
    print(f"\nScanning Area: {town_name}...")
    
    for month in months:
        # Construct URL for this specific town and month
        url = f"https://data.police.uk/api/crimes-street/all-crime?lat={coords['lat']}&lng={coords['lng']}&date={month}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Tag the data with the town name so we know where it came from
                for crime in data:
                    crime['fetched_town'] = town_name
                
                all_crimes.extend(data)
                print(f"   -> {month}: Found {len(data)} crimes.")
            else:
                print(f"   -> {month}: No data.")
                
        except Exception as e:
            print(f"   -> Error: {e}")
            
        # Be polite to the server
        time.sleep(0.1)

print("\nProcessing Data...")

# --- 4. DATA CLEANING & DEDUPLICATION ---
if len(all_crimes) > 0:
    df = pd.DataFrame(all_crimes)
    
    # CRITICAL: Remove Duplicates
    # Because towns are close, radius might overlap. 
    # The 'id' column is unique for every crime. We drop duplicates based on ID.
    initial_count = len(df)
    df.drop_duplicates(subset=['id'], inplace=True)
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate crimes (Overlapping zones).")
    
    # Extract Coordinates
    df['latitude'] = df['location'].apply(lambda x: x['latitude'])
    df['longitude'] = df['location'].apply(lambda x: x['longitude'])
    
    # Convert types
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    
    # Save the massive dataset
    # We keep 'fetched_town' so you can filter by town in the dashboard if you want!
    df = df[['category', 'month', 'latitude', 'longitude', 'fetched_town']]
    
    df.to_csv("gloucester_crime_data.csv", index=False)
    
    print("------------------------------------------------")
    print(f"SUCCESS! County-Wide Database Built.")
    print(f"Total Unique Crimes: {len(df)}")
    print("File saved as 'gloucester_crime_data.csv'")
    print("------------------------------------------------")
else:
    print("Failed to get data.")