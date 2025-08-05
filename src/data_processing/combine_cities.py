"""
Combine individual city CSV files into a single housing dataset
"""

import pandas as pd
import os
from pathlib import Path

def combine_city_data():
    """Combine all city CSV files into a single dataset"""
    raw_data_path = Path("data/raw")
    city_files = [
        "Bangalore.csv",
        "Chennai.csv", 
        "Delhi.csv",
        "Hyderabad.csv",
        "Kolkata.csv",
        "Mumbai.csv"
    ]
    
    combined_data = []
    
    for city_file in city_files:
        file_path = raw_data_path / city_file
        if file_path.exists():
            print(f"Reading {city_file}...")
            city_data = pd.read_csv(file_path)
            
            # Add city column
            city_name = city_file.replace('.csv', '')
            city_data['City'] = city_name
            
            combined_data.append(city_data)
            print(f"  - {city_name}: {city_data.shape[0]} rows")
        else:
            print(f"Warning: {city_file} not found")
    
    if combined_data:
        # Combine all dataframes
        final_data = pd.concat(combined_data, ignore_index=True)
        
        # Save combined dataset
        output_path = raw_data_path / "housing_prices_combined.csv"
        final_data.to_csv(output_path, index=False)
        
        print(f"\nCombined dataset saved to: {output_path}")
        print(f"Total rows: {final_data.shape[0]}")
        print(f"Total columns: {final_data.shape[1]}")
        print(f"Columns: {list(final_data.columns)}")
        
        return final_data
    else:
        print("No data files found to combine")
        return None

if __name__ == "__main__":
    combine_city_data()
