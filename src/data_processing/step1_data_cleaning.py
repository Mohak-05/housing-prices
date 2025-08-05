"""
Step 1: Data Ingestion & Cleaning
Housing Prices Dataset Processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class HousingDataProcessor:
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.processed_path = Path("data/processed")
        self.raw_data = None
        self.cleaned_data = None
        
    def load_dataset(self, filename):
        """Load the housing dataset"""
        try:
            file_path = self.data_path / filename
            print(f"Loading dataset from: {file_path}")
            
            # Try different file extensions
            if file_path.suffix == '.csv':
                self.raw_data = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                self.raw_data = pd.read_excel(file_path)
            else:
                # Try csv first, then excel
                try:
                    self.raw_data = pd.read_csv(file_path.with_suffix('.csv'))
                except:
                    self.raw_data = pd.read_excel(file_path.with_suffix('.xlsx'))
            
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.raw_data.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nAvailable files in data/raw:")
            for file in self.data_path.glob("*"):
                print(f"  - {file.name}")
            return False
    
    def explore_dataset(self):
        """Initial exploration of the dataset"""
        if self.raw_data is None:
            print("No data loaded. Please load dataset first.")
            return
        
        print("=" * 60)
        print("DATASET EXPLORATION")
        print("=" * 60)
        
        # Basic info
        print(f"\nDataset Shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.raw_data.head())
        
        # Data types
        print("\nData Types:")
        print(self.raw_data.dtypes)
        
        # Missing values
        print("\nMissing Values:")
        missing_vals = self.raw_data.isnull().sum()
        missing_percent = (missing_vals / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_vals,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.raw_data.describe())
        
        return missing_df
    
    def clean_data(self):
        """Clean the dataset and handle missing values"""
        if self.raw_data is None:
            print("No data loaded. Please load dataset first.")
            return
        
        print("\n" + "=" * 60)
        print("DATA CLEANING")
        print("=" * 60)
        
        # Make a copy for cleaning
        self.cleaned_data = self.raw_data.copy()
        
        # Standardize column names (lowercase, replace spaces with underscores)
        self.cleaned_data.columns = [col.lower().replace(' ', '_').replace('-', '_') 
                                   for col in self.cleaned_data.columns]
        
        print(f"Standardized column names: {list(self.cleaned_data.columns)}")
        
        # Identify key columns (this will vary based on actual dataset structure)
        # We'll make this flexible to handle different column naming conventions
        
        # Common column mappings - more specific matching
        column_mappings = {
            'price': ['price'],
            'area': ['area'],
            'bedrooms': ['bedroom', 'bhk', 'bed'],
            'bathrooms': ['bathroom', 'bath'],
            'location': ['location'],
            'city': ['city'],
            'property_type': ['property_type', 'type']
        }
        
        # Find actual column names with exact matching
        actual_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in self.cleaned_data.columns:
                for name in possible_names:
                    if name in col.lower() and standard_name not in actual_columns:
                        actual_columns[standard_name] = col
                        break
        
        print(f"\nIdentified columns: {actual_columns}")
        
        # Handle missing values based on column type
        initial_rows = len(self.cleaned_data)
        
        # Remove rows with missing price (target variable)
        if 'price' in actual_columns:
            price_col = actual_columns['price']
            self.cleaned_data = self.cleaned_data.dropna(subset=[price_col])
            print(f"Removed {initial_rows - len(self.cleaned_data)} rows with missing prices")
        
        # Handle missing values in other columns
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].dtype in ['object', 'string']:
                # For categorical columns, fill with 'Unknown'
                missing_count = self.cleaned_data[col].isnull().sum()
                if missing_count > 0:
                    self.cleaned_data[col].fillna('Unknown', inplace=True)
                    print(f"Filled {missing_count} missing values in {col} with 'Unknown'")
            else:
                # For numerical columns, fill with median
                missing_count = self.cleaned_data[col].isnull().sum()
                if missing_count > 0:
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"Filled {missing_count} missing values in {col} with median: {median_val}")
        
        # Standardize location names
        if 'location' in actual_columns:
            loc_col = actual_columns['location']
            # Remove extra spaces and standardize case
            self.cleaned_data[loc_col] = self.cleaned_data[loc_col].str.strip().str.title()
            print(f"Standardized location names in {loc_col}")
        
        if 'city' in actual_columns:
            city_col = actual_columns['city']
            self.cleaned_data[city_col] = self.cleaned_data[city_col].str.strip().str.title()
            print(f"Standardized city names in {city_col}")
        
        print(f"\nCleaned dataset shape: {self.cleaned_data.shape}")
        return actual_columns
    
    def engineer_features(self, actual_columns):
        """Engineer new features"""
        if self.cleaned_data is None:
            print("No cleaned data available. Please clean data first.")
            return
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # 1. Price per sqft
        if 'price' in actual_columns and 'area' in actual_columns:
            price_col = actual_columns['price']
            area_col = actual_columns['area']
            
            # Ensure area is not zero
            self.cleaned_data = self.cleaned_data[self.cleaned_data[area_col] > 0]
            self.cleaned_data['price_per_sqft'] = self.cleaned_data[price_col] / self.cleaned_data[area_col]
            print(f"Created price_per_sqft feature")
        
        # 2. Total rooms
        if 'bedrooms' in actual_columns and 'bathrooms' in actual_columns:
            bed_col = actual_columns['bedrooms']
            bath_col = actual_columns['bathrooms']
            self.cleaned_data['total_rooms'] = self.cleaned_data[bed_col] + self.cleaned_data[bath_col]
            print(f"Created total_rooms feature")
        
        # 3. Location frequency (popularity of neighborhood)
        if 'location' in actual_columns:
            loc_col = actual_columns['location']
            location_counts = self.cleaned_data[loc_col].value_counts()
            self.cleaned_data['location_frequency'] = self.cleaned_data[loc_col].map(location_counts)
            print(f"Created location_frequency feature")
            print(f"Most popular locations:")
            print(location_counts.head(10))
        
        print(f"\nFinal dataset shape: {self.cleaned_data.shape}")
        print(f"New columns added: {[col for col in self.cleaned_data.columns if col not in self.raw_data.columns]}")
    
    def save_cleaned_data(self):
        """Save the cleaned dataset"""
        if self.cleaned_data is None:
            print("No cleaned data to save.")
            return
        
        output_path = self.processed_path / "cleaned_data.csv"
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        print(f"Final dataset shape: {self.cleaned_data.shape}")
    
    def generate_summary_insights(self, actual_columns):
        """Generate summary insights and visualizations"""
        if self.cleaned_data is None:
            print("No cleaned data available.")
            return
        
        print("\n" + "=" * 60)
        print("SUMMARY INSIGHTS")
        print("=" * 60)
        
        # City-wise statistics
        if 'city' in actual_columns and 'price' in actual_columns:
            city_col = actual_columns['city']
            price_col = actual_columns['price']
            
            city_stats = self.cleaned_data.groupby(city_col)[price_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            city_stats.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price', 'Min_Price', 'Max_Price']
            print("\nCity-wise Price Statistics:")
            print(city_stats.sort_values('Mean_Price', ascending=False))
        
        # Top neighborhoods per city
        if 'city' in actual_columns and 'location' in actual_columns:
            city_col = actual_columns['city']
            loc_col = actual_columns['location']
            price_col = actual_columns['price']
            
            print("\nTop 10 Neighborhoods per City (by average price):")
            for city in self.cleaned_data[city_col].unique()[:5]:  # Top 5 cities
                city_data = self.cleaned_data[self.cleaned_data[city_col] == city]
                top_neighborhoods = city_data.groupby(loc_col)[price_col].agg([
                    'count', 'mean'
                ]).round(2)
                top_neighborhoods = top_neighborhoods[top_neighborhoods['count'] >= 3]  # At least 3 listings
                top_neighborhoods = top_neighborhoods.sort_values('mean', ascending=False).head(10)
                
                print(f"\n{city}:")
                print(top_neighborhoods)
    
    def create_visualizations(self, actual_columns):
        """Create visualizations"""
        if self.cleaned_data is None:
            print("No cleaned data available.")
            return
        
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Housing Price Analysis - Step 1 Results', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        if 'price' in actual_columns:
            price_col = actual_columns['price']
            axes[0, 0].hist(self.cleaned_data[price_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Price Distribution')
            axes[0, 0].set_xlabel('Price')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. City-wise price comparison
        if 'city' in actual_columns and 'price' in actual_columns:
            city_col = actual_columns['city']
            price_col = actual_columns['price']
            
            # Box plot for top cities
            top_cities = self.cleaned_data[city_col].value_counts().head(6).index
            city_data = self.cleaned_data[self.cleaned_data[city_col].isin(top_cities)]
            
            sns.boxplot(data=city_data, x=city_col, y=price_col, ax=axes[0, 1])
            axes[0, 1].set_title('Price Distribution by City')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Price per sqft distribution
        if 'price_per_sqft' in self.cleaned_data.columns:
            axes[1, 0].hist(self.cleaned_data['price_per_sqft'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].set_title('Price per Sqft Distribution')
            axes[1, 0].set_xlabel('Price per Sqft')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Location frequency distribution
        if 'location_frequency' in self.cleaned_data.columns:
            axes[1, 1].hist(self.cleaned_data['location_frequency'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('Location Frequency Distribution')
            axes[1, 1].set_xlabel('Number of Listings per Location')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save the plot
        plots_dir = Path("data/processed")
        plot_path = plots_dir / "step1_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {plot_path}")
        
        plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("AI-POWERED REAL ESTATE PRICE ADVISOR - STEP 1")
    print("Data Ingestion & Cleaning")
    print("=" * 80)
    
    # Initialize processor
    processor = HousingDataProcessor()
    
    # Instructions for user
    print("\nINSTRUCTIONS:")
    print("1. Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india")
    print("2. Place the CSV file in the 'data/raw/' folder")
    print("3. Update the filename below if different from 'housing_prices.csv'")
    print("\nAlternatively, if you have Kaggle API set up:")
    print("   Run: kaggle datasets download -d ruchi798/housing-prices-in-metropolitan-areas-of-india")
    
    # Try to load dataset (prioritize combined dataset)
    dataset_files = []
    combined_file = Path("data/raw/housing_prices_combined.csv")
    if combined_file.exists():
        dataset_files.append(combined_file)
    else:
        dataset_files = list(Path("data/raw").glob("*.csv")) + list(Path("data/raw").glob("*.xlsx"))
    
    if dataset_files:
        print(f"\nFound dataset files: {[f.name for f in dataset_files]}")
        # Use the combined file if available, otherwise use the first found file
        filename = dataset_files[0].name
        
        if processor.load_dataset(filename):
            # Step 1: Explore dataset
            missing_info = processor.explore_dataset()
            
            # Step 2: Clean data
            actual_columns = processor.clean_data()
            
            # Step 3: Engineer features
            processor.engineer_features(actual_columns)
            
            # Step 4: Save cleaned data
            processor.save_cleaned_data()
            
            # Step 5: Generate insights
            processor.generate_summary_insights(actual_columns)
            
            # Step 6: Create visualizations
            processor.create_visualizations(actual_columns)
            
            print("\n" + "=" * 80)
            print("STEP 1 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nNext Steps:")
            print("1. Review the cleaned_data.csv file")
            print("2. Check the visualizations (step1_analysis.png)")
            print("3. Verify the data quality and feature engineering")
            print("4. Ready to proceed to Step 2: Location Enrichment")
            
        else:
            print("\nPlease download the dataset and place it in data/raw/ folder")
    else:
        print("\nNo dataset found in data/raw/ folder.")
        print("Please download the dataset from Kaggle and place it there.")


if __name__ == "__main__":
    main()
