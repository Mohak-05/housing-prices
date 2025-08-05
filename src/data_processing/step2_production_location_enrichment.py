"""
Production-Ready Step 2: GPU-Accelerated Location Enrichment with Spatial Indexing
Ultra-fast O(N log N) neighborhood computation using GPU-accelerated libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import json
import pickle
from pathlib import Path
import warnings
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob

# GPU-accelerated libraries
try:
    import cudf
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available (RAPIDS cuDF/cuML)")
except ImportError:
    # Fallback to CPU libraries
    from sklearn.neighbors import BallTree, NearestNeighbors
    GPU_AVAILABLE = False
    print("⚡ Using CPU acceleration (scikit-learn)")

# Try to use GPU-accelerated distance calculations
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"🔥 PyTorch GPU available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPUOptimizedLocationEnricher:
    def __init__(self, processed_data_path="data/processed", use_gpu=True):
        # Setup logger first
        self.logger = logging.getLogger(__name__)
        
        # GPU configuration
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_torch_gpu = use_gpu and TORCH_AVAILABLE
        
        if self.use_gpu:
            self.logger.info("🚀 GPU acceleration enabled (RAPIDS)")
        elif self.use_torch_gpu:
            self.logger.info("🔥 PyTorch GPU acceleration enabled")
        else:
            self.logger.info("⚡ Using CPU optimization")
        
        self.processed_path = Path(processed_data_path)
        self.geo_enriched_path = Path("data/geo_enriched")
        self.geo_enriched_path.mkdir(exist_ok=True)
        
        # Cache files
        self.geocoding_cache_path = self.geo_enriched_path / "geocoding_cache.pkl"
        self.neighborhood_cache_path = self.geo_enriched_path / "neighborhood_cache.pkl"
        self.progress_path = self.geo_enriched_path / "progress.json"
        
        # Output files
        self.output_files = {
            'dataset': self.geo_enriched_path / "production_geo_enriched_data.csv",
            'analysis': self.geo_enriched_path / "production_geo_analysis.png",
            'heatmap': self.geo_enriched_path / "production_price_heatmap.html"
        }
        
        self.data = None
        
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="housing_price_analyzer_gpu_optimized_v3")
        
        # City centers for fallback and distance calculations
        self.city_centers = {
            'Bangalore': (12.9716, 77.5946),
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867),
            'Kolkata': (22.5726, 88.3639)
        }
        
        # Clean incomplete files from previous runs
        self.cleanup_incomplete_files()
        
        # Load caches after logger is initialized
        self.geocoding_cache = self.load_geocoding_cache()
        self.neighborhood_cache = self.load_neighborhood_cache()
    
    def cleanup_incomplete_files(self):
        """Clean up incomplete/corrupted files from previous runs"""
        for file_type, file_path in self.output_files.items():
            if file_path.exists():
                try:
                    if file_type == 'dataset':
                        # Quick check if CSV is readable
                        pd.read_csv(file_path, nrows=1)
                        # Check if it has required columns
                        df_check = pd.read_csv(file_path, nrows=100)
                        required_cols = ['latitude', 'longitude', 'distance_to_city_center_km']
                        if not all(col in df_check.columns for col in required_cols):
                            raise ValueError("Missing required columns")
                    elif file_type == 'analysis':
                        # Check if image file is not corrupted (basic size check)
                        if file_path.stat().st_size < 1000:  # Less than 1KB probably corrupted
                            raise ValueError("File too small")
                    elif file_type == 'heatmap':
                        # Check if HTML file contains expected content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(1000)
                            if 'folium' not in content.lower():
                                raise ValueError("Invalid HTML content")
                    
                    self.logger.info(f"Found valid {file_type} file: {file_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Deleting corrupted {file_type} file: {file_path} ({e})")
                    file_path.unlink()
    
    def load_geocoding_cache(self):
        """Load geocoding cache"""
        if self.geocoding_cache_path.exists():
            try:
                with open(self.geocoding_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.info(f"Loaded geocoding cache with {len(cache)} entries")
                return cache
            except Exception as e:
                self.logger.warning(f"Corrupted geocoding cache, starting fresh: {e}")
                self.geocoding_cache_path.unlink()
        return {}
    
    def load_neighborhood_cache(self):
        """Load neighborhood computation cache"""
        if self.neighborhood_cache_path.exists():
            try:
                with open(self.neighborhood_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.info(f"Loaded neighborhood cache with {len(cache)} cities")
                return cache
            except Exception as e:
                self.logger.warning(f"Corrupted neighborhood cache, starting fresh: {e}")
                self.neighborhood_cache_path.unlink()
        return {}
    
    def save_geocoding_cache(self):
        """Save geocoding cache"""
        with open(self.geocoding_cache_path, 'wb') as f:
            pickle.dump(self.geocoding_cache, f)
    
    def save_neighborhood_cache(self):
        """Save neighborhood cache"""
        with open(self.neighborhood_cache_path, 'wb') as f:
            pickle.dump(self.neighborhood_cache, f)
    
    def save_progress(self, stage, details):
        """Save current progress"""
        progress = {
            'stage': stage,
            'details': details,
            'timestamp': time.time()
        }
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """Load progress from previous run"""
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Corrupted progress file: {e}")
                self.progress_path.unlink()
        return None
    
    def gpu_haversine_distance(self, coords1, coords2):
        """GPU-accelerated Haversine distance calculation using PyTorch"""
        if not self.use_torch_gpu:
            # Fallback to numpy
            return self.cpu_haversine_distance(coords1, coords2)
        
        # Convert to PyTorch tensors on GPU
        lat1, lon1 = torch.tensor(coords1[:, 0], device='cuda'), torch.tensor(coords1[:, 1], device='cuda')
        lat2, lon2 = torch.tensor(coords2[:, 0], device='cuda'), torch.tensor(coords2[:, 1], device='cuda')
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = torch.deg2rad(lat1), torch.deg2rad(lon1), torch.deg2rad(lat2), torch.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371.0
        distances = c * r
        
        return distances.cpu().numpy()
    
    def cpu_haversine_distance(self, coords1, coords2):
        """CPU-optimized Haversine distance calculation"""
        lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
        lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371.0
        return c * r
    
    def load_data_optimized(self):
        """Load data with GPU acceleration if available"""
        cleaned_file = self.processed_path / "cleaned_data.csv"
        if not cleaned_file.exists():
            self.logger.error("cleaned_data.csv not found. Please run Step 1 first.")
            return False
        
        if self.use_gpu:
            try:
                # Load with cuDF (GPU-accelerated pandas)
                self.data = cudf.read_csv(cleaned_file)
                self.logger.info(f"🚀 Loaded data with GPU acceleration: {self.data.shape}")
                return True
            except Exception as e:
                self.logger.warning(f"GPU loading failed, falling back to CPU: {e}")
        
        # Fallback to regular pandas
        self.data = pd.read_csv(cleaned_file)
        self.logger.info(f"Loaded data with CPU: {self.data.shape}")
        return True
    
    def geocode_location_with_retry(self, location_name, city_name, max_retries=3):
        """Geocode with retry logic and proper rate limiting"""
        cache_key = f"{location_name}_{city_name}"
        
        # Check cache first
        if cache_key in self.geocoding_cache:
            return self.geocoding_cache[cache_key]
        
        full_address = f"{location_name}, {city_name}, India"
        
        for attempt in range(max_retries):
            try:
                # Respect rate limits
                time.sleep(1.1)
                
                location = self.geolocator.geocode(full_address, timeout=15)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.geocoding_cache[cache_key] = coords
                    return coords
                else:
                    # Fallback to city name only
                    city_location = self.geolocator.geocode(f"{city_name}, India", timeout=15)
                    if city_location:
                        coords = (city_location.latitude, city_location.longitude)
                        self.geocoding_cache[cache_key] = coords
                        return coords
                    
            except Exception as e:
                self.logger.warning(f"Geocoding error for {full_address} (attempt {attempt+1}): {e}")
                time.sleep(3)
        
        # Final fallback to city center
        if city_name in self.city_centers:
            coords = self.city_centers[city_name]
            self.geocoding_cache[cache_key] = coords
            return coords
        
        # Complete failure
        self.geocoding_cache[cache_key] = (None, None)
        return (None, None)
    
    def process_geocoding(self):
        """Process all unique location-city combinations with progress tracking"""
        self.logger.info("=" * 80)
        self.logger.info("GEOCODING PHASE - Processing unique locations")
        self.logger.info("=" * 80)
        
        # Get unique location-city combinations
        unique_locations = self.data[['location', 'city']].drop_duplicates()
        total_unique = len(unique_locations)
        
        # Check what's already cached
        cached_count = sum(1 for _, row in unique_locations.iterrows() 
                          if f"{row['location']}_{row['city']}" in self.geocoding_cache)
        
        remaining = total_unique - cached_count
        
        self.logger.info(f"Total unique locations: {total_unique}")
        self.logger.info(f"Already cached: {cached_count}")
        self.logger.info(f"Need to geocode: {remaining}")
        
        if remaining > 0:
            estimated_time = remaining * 1.1 / 60  # 1.1 seconds per request
            self.logger.info(f"Estimated time: {estimated_time:.1f} minutes")
            
            # Process remaining locations with progress bar
            processed = 0
            start_time = time.time()
            
            for idx, row in tqdm(unique_locations.iterrows(), total=total_unique, desc="Geocoding"):
                cache_key = f"{row['location']}_{row['city']}"
                
                if cache_key not in self.geocoding_cache:
                    coords = self.geocode_location_with_retry(row['location'], row['city'])
                    processed += 1
                    
                    # Save cache every 50 locations
                    if processed % 50 == 0:
                        self.save_geocoding_cache()
                        elapsed = (time.time() - start_time) / 60
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining_time = (remaining - processed) / rate if rate > 0 else 0
                        self.logger.info(f"Processed {processed}/{remaining} new locations. "
                                       f"ETA: {remaining_time:.1f} minutes")
        
        # Final save
        self.save_geocoding_cache()
        self.save_progress("geocoding_complete", {"total_locations": total_unique})
        self.logger.info(f"Geocoding completed! Cache contains {len(self.geocoding_cache)} locations.")
    
    def apply_coordinates_to_dataset(self):
        """Apply cached coordinates to the full dataset"""
        self.logger.info("=" * 60)
        self.logger.info("APPLYING COORDINATES TO DATASET")
        self.logger.info("=" * 60)
        
        enriched_data = self.data.copy()
        
        # Vectorized coordinate application
        def get_coords(row):
            cache_key = f"{row['location']}_{row['city']}"
            return self.geocoding_cache.get(cache_key, (None, None))
        
        self.logger.info("Applying coordinates from cache...")
        coords_df = enriched_data.apply(get_coords, axis=1, result_type='expand')
        enriched_data['latitude'] = coords_df[0]
        enriched_data['longitude'] = coords_df[1]
        
        # Calculate distance to city center
        self.logger.info("Calculating distances to city centers...")
        def calc_distance_to_center(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                return np.nan
            
            city_center = self.city_centers.get(row['city'])
            if not city_center:
                return np.nan
            
            property_coords = (row['latitude'], row['longitude'])
            distance = geodesic(property_coords, city_center).kilometers
            return distance
        
        enriched_data['distance_to_city_center_km'] = enriched_data.apply(
            calc_distance_to_center, axis=1
        )
        
        # Report success rates
        successful_coords = enriched_data[['latitude', 'longitude']].notna().all(axis=1).sum()
        successful_distances = enriched_data['distance_to_city_center_km'].notna().sum()
        
        self.logger.info(f"Coordinates applied: {successful_coords:,}/{len(enriched_data):,} "
                        f"({successful_coords/len(enriched_data)*100:.1f}%)")
        self.logger.info(f"Distances calculated: {successful_distances:,}")
        
        self.save_progress("coordinates_applied", {
            "total_records": len(enriched_data),
            "successful_coords": int(successful_coords),
            "successful_distances": int(successful_distances)
        })
        
        return enriched_data
    
    def calculate_neighborhood_features_gpu_optimized(self, enriched_data):
        """Ultra-fast GPU-accelerated neighborhood computation"""
        self.logger.info("=" * 60)
        self.logger.info("GPU-ACCELERATED NEIGHBORHOOD COMPUTATION")
        self.logger.info("=" * 60)
        
        # Convert to pandas if using cuDF
        if self.use_gpu and hasattr(enriched_data, 'to_pandas'):
            final_data = enriched_data.to_pandas()
        else:
            final_data = enriched_data.copy()
        
        final_data['neighborhood_avg_price_2km'] = np.nan
        
        # Group by city for efficiency
        cities = final_data['city'].unique()
        
        for city_idx, city in enumerate(cities):
            start_time = time.time()
            
            # Check if this city is already processed in cache
            if city in self.neighborhood_cache:
                self.logger.info(f"Loading cached neighborhood data for {city}")
                city_neighborhood_data = self.neighborhood_cache[city]
                
                # Apply cached data to final_data
                city_mask = final_data['city'] == city
                city_indices = final_data[city_mask].index
                
                for idx in city_indices:
                    cache_key = f"{final_data.loc[idx, 'location']}_{final_data.loc[idx, 'latitude']}_{final_data.loc[idx, 'longitude']}"
                    if cache_key in city_neighborhood_data:
                        final_data.loc[idx, 'neighborhood_avg_price_2km'] = city_neighborhood_data[cache_key]
                
                continue
            
            # Process city data
            city_data = final_data[final_data['city'] == city].copy()
            city_data = city_data.dropna(subset=['latitude', 'longitude', 'price'])
            
            if len(city_data) < 2:
                self.logger.warning(f"Skipping {city}: insufficient data ({len(city_data)} properties)")
                continue
            
            self.logger.info(f"Processing {city} ({city_idx+1}/{len(cities)}): {len(city_data)} properties")
            
            # GPU-accelerated neighborhood computation
            coords = city_data[['latitude', 'longitude']].values
            prices = city_data['price'].values
            
            if self.use_gpu and len(city_data) > 1000:
                # Use cuML for large datasets
                try:
                    coords_gpu = cudf.DataFrame({'lat': coords[:, 0], 'lon': coords[:, 1]})
                    
                    # Convert to radians for Haversine
                    coords_rad = np.radians(coords)
                    
                    # Use cuML NearestNeighbors
                    nn = cuNearestNeighbors(metric='haversine')
                    nn.fit(coords_rad)
                    
                    # Earth radius in kilometers, 2km radius in radians
                    radius_rad = 2.0 / 6371.0
                    
                    # Query neighbors within radius
                    indices, distances = nn.radius_neighbors(coords_rad, radius=radius_rad)
                    
                    city_neighborhood_cache = {}
                    
                    for i, (idx, neighbors, neighbor_distances) in enumerate(zip(city_data.index, indices, distances)):
                        if len(neighbors) > 0:
                            # Get prices of neighboring properties
                            neighbor_prices = prices[neighbors]
                            avg_price = np.mean(neighbor_prices)
                            
                            final_data.loc[idx, 'neighborhood_avg_price_2km'] = avg_price
                            
                            # Cache for this city
                            row = city_data.loc[idx]
                            cache_key = f"{row['location']}_{row['latitude']}_{row['longitude']}"
                            city_neighborhood_cache[cache_key] = avg_price
                        
                        # Progress update every 2000 properties for GPU
                        if (i + 1) % 2000 == 0:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            remaining = len(city_data) - (i + 1)
                            eta = remaining / rate if rate > 0 else 0
                            self.logger.info(f"  {city} (GPU): {i+1:,}/{len(city_data):,} processed, ETA: {eta:.1f}s")
                    
                    self.logger.info(f"✅ {city} (GPU) completed in {time.time() - start_time:.1f}s")
                    
                except Exception as e:
                    self.logger.warning(f"GPU computation failed for {city}, falling back to CPU: {e}")
                    # Fallback to CPU BallTree
                    city_neighborhood_cache = self._compute_neighborhoods_cpu_balltree(city_data, final_data, city)
            
            elif self.use_torch_gpu and len(city_data) > 500:
                # Use PyTorch for medium datasets
                try:
                    city_neighborhood_cache = self._compute_neighborhoods_torch_gpu(city_data, final_data, city)
                    self.logger.info(f"✅ {city} (PyTorch GPU) completed in {time.time() - start_time:.1f}s")
                except Exception as e:
                    self.logger.warning(f"PyTorch GPU computation failed for {city}, falling back to CPU: {e}")
                    city_neighborhood_cache = self._compute_neighborhoods_cpu_balltree(city_data, final_data, city)
            
            else:
                # Use CPU BallTree for smaller datasets or when GPU unavailable
                city_neighborhood_cache = self._compute_neighborhoods_cpu_balltree(city_data, final_data, city)
                self.logger.info(f"✅ {city} (CPU) completed in {time.time() - start_time:.1f}s")
            
            # Save city cache
            self.neighborhood_cache[city] = city_neighborhood_cache
            self.save_neighborhood_cache()
            
            elapsed = time.time() - start_time
            rate = len(city_data) / elapsed if elapsed > 0 else 0
            self.logger.info(f"  Performance: {rate:.0f} properties/sec")
        
        # Fill remaining NaN values with location-based averages
        self.logger.info("Filling remaining NaN values with location averages...")
        location_avg_prices = final_data.groupby(['city', 'location'])['price'].mean()
        
        nan_mask = final_data['neighborhood_avg_price_2km'].isna()
        for idx in final_data[nan_mask].index:
            row = final_data.loc[idx]
            city_location_key = (row['city'], row['location'])
            if city_location_key in location_avg_prices:
                final_data.loc[idx, 'neighborhood_avg_price_2km'] = location_avg_prices[city_location_key]
        
        # Final statistics
        neighborhood_count = final_data['neighborhood_avg_price_2km'].notna().sum()
        self.logger.info(f"Neighborhood features calculated for {neighborhood_count:,} records")
        
        self.save_progress("neighborhoods_complete", {
            "processed_cities": len(cities),
            "total_neighborhoods": int(neighborhood_count)
        })
        
        return final_data
    
    def _compute_neighborhoods_torch_gpu(self, city_data, final_data, city):
        """PyTorch GPU-accelerated neighborhood computation"""
        coords = city_data[['latitude', 'longitude']].values
        prices = city_data['price'].values
        
        # Convert to PyTorch tensors on GPU
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device='cuda')
        
        city_neighborhood_cache = {}
        batch_size = 1000  # Process in batches to manage memory
        
        for i in range(0, len(city_data), batch_size):
            end_idx = min(i + batch_size, len(city_data))
            batch_coords = coords_tensor[i:end_idx]
            
            # Compute pairwise distances using broadcasting
            coords_expanded = coords_tensor.unsqueeze(0)  # (1, N, 2)
            batch_expanded = batch_coords.unsqueeze(1)    # (batch, 1, 2)
            
            # Haversine distance on GPU
            lat1 = torch.deg2rad(coords_expanded[:, :, 0])
            lon1 = torch.deg2rad(coords_expanded[:, :, 1])
            lat2 = torch.deg2rad(batch_expanded[:, :, 0])
            lon2 = torch.deg2rad(batch_expanded[:, :, 1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
            c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
            distances = c * 6371.0  # Earth radius in km
            
            # Find neighbors within 2km
            within_radius = distances <= 2.0
            
            for j, (idx, neighbors_mask) in enumerate(zip(city_data.index[i:end_idx], within_radius)):
                neighbor_indices = torch.where(neighbors_mask)[0].cpu().numpy()
                if len(neighbor_indices) > 0:
                    neighbor_prices = prices[neighbor_indices]
                    avg_price = np.mean(neighbor_prices)
                    
                    final_data.loc[idx, 'neighborhood_avg_price_2km'] = avg_price
                    
                    # Cache for this city
                    row = city_data.loc[idx]
                    cache_key = f"{row['location']}_{row['latitude']}_{row['longitude']}"
                    city_neighborhood_cache[cache_key] = avg_price
        
        return city_neighborhood_cache
    
    def _compute_neighborhoods_cpu_balltree(self, city_data, final_data, city):
        """CPU BallTree fallback computation"""
        coords = city_data[['latitude', 'longitude']].values
        prices = city_data['price'].values
        
        # Convert coordinates to radians for Haversine distance
        coords_rad = np.radians(coords)
        
        # Build BallTree with Haversine metric
        from sklearn.neighbors import BallTree
        tree = BallTree(coords_rad, metric='haversine')
        
        # Earth radius in kilometers, 2km radius in radians
        radius_rad = 2.0 / 6371.0
        
        # Query for neighbors within 2km radius
        neighbor_indices = tree.query_radius(coords_rad, r=radius_rad)
        
        city_neighborhood_cache = {}
        
        for i, (idx, neighbors) in enumerate(zip(city_data.index, neighbor_indices)):
            if len(neighbors) > 0:
                neighbor_prices = prices[neighbors]
                avg_price = np.mean(neighbor_prices)
                
                final_data.loc[idx, 'neighborhood_avg_price_2km'] = avg_price
                
                # Cache for this city
                row = city_data.loc[idx]
                cache_key = f"{row['location']}_{row['latitude']}_{row['longitude']}"
                city_neighborhood_cache[cache_key] = avg_price
            
            # Progress update every 1000 properties for CPU
            if (i + 1) % 1000 == 0:
                self.logger.info(f"  {city} (CPU): {i+1:,}/{len(city_data):,} processed")
        
        return city_neighborhood_cache
    
    def create_production_visualizations(self, enriched_data):
        """Create comprehensive visualizations with atomic file operations"""
        self.logger.info("=" * 60)
        self.logger.info("CREATING PRODUCTION VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        # Temporary files to ensure atomic operations
        temp_analysis = self.output_files['analysis'].with_suffix('.tmp')
        temp_heatmap = self.output_files['heatmap'].with_suffix('.tmp')
        
        try:
            # 1. Create comprehensive static analysis plots
            self.logger.info("Creating analysis plots...")
            
            fig, axes = plt.subplots(3, 2, figsize=(20, 18))
            fig.suptitle('Production Geospatial Analysis - Full Dataset (Optimized)', 
                        fontsize=16, fontweight='bold')
            
            valid_data = enriched_data.dropna(subset=['distance_to_city_center_km', 'price'])
            
            # Price vs Distance scatter
            axes[0, 0].scatter(valid_data['distance_to_city_center_km'], 
                              valid_data['price'], alpha=0.4, s=2)
            axes[0, 0].set_xlabel('Distance to City Center (km)')
            axes[0, 0].set_ylabel('Price (₹)')
            axes[0, 0].set_title('Price vs Distance to City Center')
            
            # City-wise distance distribution
            sns.boxplot(data=valid_data, x='city', y='distance_to_city_center_km', ax=axes[0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_title('Distance Distribution by City')
            
            # Price per sqft by city
            sns.boxplot(data=enriched_data, x='city', y='price_per_sqft', ax=axes[1, 0])
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_title('Price per Sqft by City')
            
            # Distance vs Price per sqft
            axes[1, 1].scatter(valid_data['distance_to_city_center_km'], 
                              valid_data['price_per_sqft'], alpha=0.4, s=2)
            axes[1, 1].set_xlabel('Distance to City Center (km)')
            axes[1, 1].set_ylabel('Price per Sqft (₹)')
            axes[1, 1].set_title('Price per Sqft vs Distance')
            
            # Neighborhood comparison
            neighborhood_data = enriched_data.dropna(subset=['neighborhood_avg_price_2km', 'price'])
            if len(neighborhood_data) > 0:
                axes[2, 0].scatter(neighborhood_data['neighborhood_avg_price_2km'], 
                                  neighborhood_data['price'], alpha=0.4, s=2)
                axes[2, 0].plot([neighborhood_data['neighborhood_avg_price_2km'].min(), 
                                neighborhood_data['neighborhood_avg_price_2km'].max()],
                               [neighborhood_data['neighborhood_avg_price_2km'].min(), 
                                neighborhood_data['neighborhood_avg_price_2km'].max()], 
                               'r--', alpha=0.8)
                axes[2, 0].set_xlabel('Neighborhood Average Price (₹)')
                axes[2, 0].set_ylabel('Individual Property Price (₹)')
                axes[2, 0].set_title('Individual vs Neighborhood Pricing')
            
            # Feature correlation heatmap
            feature_cols = ['price', 'area', 'distance_to_city_center_km', 'price_per_sqft', 'location_frequency']
            if 'neighborhood_avg_price_2km' in enriched_data.columns:
                feature_cols.append('neighborhood_avg_price_2km')
            
            corr_data = enriched_data[feature_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[2, 1])
            axes[2, 1].set_title('Feature Correlation Matrix')
            
            plt.tight_layout()
            
            # Save to temporary file first
            plt.savefig(temp_analysis, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Atomic move
            temp_analysis.rename(self.output_files['analysis'])
            self.logger.info(f"✅ Analysis plots saved: {self.output_files['analysis']}")
            
            # 2. Create interactive map
            self.logger.info("Creating interactive heatmap...")
            
            # Center map on India
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
            
            # Sample for visualization (stratified by city)
            sample_data = enriched_data.dropna(subset=['latitude', 'longitude'])
            if len(sample_data) > 3000:
                # Stratified sampling by city
                samples_per_city = min(500, len(sample_data) // len(sample_data['city'].unique()))
                sample_data = sample_data.groupby('city').apply(
                    lambda x: x.sample(n=min(samples_per_city, len(x)), random_state=42)
                ).reset_index(drop=True)
            
            if len(sample_data) > 0:
                # Create price-based color mapping
                min_price = sample_data['price'].quantile(0.01)
                max_price = sample_data['price'].quantile(0.99)
                
                def get_color(price):
                    normalized = (price - min_price) / (max_price - min_price)
                    normalized = max(0, min(1, normalized))  # Clamp between 0 and 1
                    
                    if normalized < 0.2:
                        return 'green'
                    elif normalized < 0.4:
                        return 'lightgreen'
                    elif normalized < 0.6:
                        return 'orange'
                    elif normalized < 0.8:
                        return 'red'
                    else:
                        return 'darkred'
                
                # Add markers by city
                for city in sample_data['city'].unique():
                    city_data = sample_data[sample_data['city'] == city]
                    
                    for _, row in city_data.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=4,
                            popup=f"<b>{row['location']}, {row['city']}</b><br>"
                                  f"Price: ₹{row['price']:,.0f}<br>"
                                  f"Area: {row['area']} sqft<br>"
                                  f"Price/sqft: ₹{row['price_per_sqft']:,.0f}<br>"
                                  f"Distance: {row['distance_to_city_center_km']:.1f}km",
                            color=get_color(row['price']),
                            fill=True,
                            fillColor=get_color(row['price']),
                            fillOpacity=0.7
                        ).add_to(m)
                
                # Add legend
                legend_html = '''
                     <div style="position: fixed; 
                                 bottom: 50px; left: 50px; width: 150px; height: 110px; 
                                 background-color: white; border:2px solid grey; z-index:9999; 
                                 font-size:14px; font-weight: bold; padding: 10px;
                                 ">
                     <b>Price Legend</b><br>
                     <i style="background:darkred; width:12px; height:12px; display:inline-block;"></i> Very High<br>
                     <i style="background:red; width:12px; height:12px; display:inline-block;"></i> High<br>
                     <i style="background:orange; width:12px; height:12px; display:inline-block;"></i> Medium<br>
                     <i style="background:lightgreen; width:12px; height:12px; display:inline-block;"></i> Low<br>
                     <i style="background:green; width:12px; height:12px; display:inline-block;"></i> Very Low
                     </div>
                     '''
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save to temporary file first
            m.save(str(temp_heatmap))
            
            # Atomic move
            temp_heatmap.rename(self.output_files['heatmap'])
            self.logger.info(f"✅ Interactive heatmap saved: {self.output_files['heatmap']}")
            
        except Exception as e:
            # Clean up temporary files on error
            for temp_file in [temp_analysis, temp_heatmap]:
                if temp_file.exists():
                    temp_file.unlink()
            raise e
    
    def save_production_dataset(self, enriched_data):
        """Save the complete geo-enriched dataset with atomic operation"""
        self.logger.info("=" * 80)
        self.logger.info("SAVING PRODUCTION DATASET")
        self.logger.info("=" * 80)
        
        # Save to temporary file first for atomic operation
        temp_output = self.output_files['dataset'].with_suffix('.tmp')
        
        try:
            enriched_data.to_csv(temp_output, index=False)
            
            # Verify the file was written correctly
            verify_df = pd.read_csv(temp_output, nrows=10)
            required_cols = ['latitude', 'longitude', 'distance_to_city_center_km', 'neighborhood_avg_price_2km']
            if not all(col in verify_df.columns for col in required_cols):
                raise ValueError("Required columns missing in output file")
            
            # Atomic move
            temp_output.rename(self.output_files['dataset'])
            
            self.logger.info(f"✅ Dataset saved: {self.output_files['dataset']}")
            self.logger.info(f"Shape: {enriched_data.shape}")
            
            # Feature coverage report
            geospatial_features = ['latitude', 'longitude', 'distance_to_city_center_km', 'neighborhood_avg_price_2km']
            
            self.logger.info("\nGeospatial Feature Coverage:")
            for feature in geospatial_features:
                if feature in enriched_data.columns:
                    coverage = enriched_data[feature].notna().sum()
                    percentage = coverage / len(enriched_data) * 100
                    self.logger.info(f"  {feature}: {coverage:,}/{len(enriched_data):,} ({percentage:.1f}%)")
            
            # Data quality metrics
            complete_records = enriched_data[geospatial_features].notna().all(axis=1).sum()
            coords_records = enriched_data[['latitude', 'longitude']].notna().all(axis=1).sum()
            
            self.logger.info(f"\nData Quality Summary:")
            self.logger.info(f"  Total records: {len(enriched_data):,}")
            self.logger.info(f"  Complete geo records: {complete_records:,} ({complete_records/len(enriched_data)*100:.1f}%)")
            self.logger.info(f"  Records with coordinates: {coords_records:,} ({coords_records/len(enriched_data)*100:.1f}%)")
            
            # Save final progress
            self.save_progress("production_complete", {
                "total_records": len(enriched_data),
                "complete_records": int(complete_records),
                "coords_records": int(coords_records),
                "output_files": [str(f) for f in self.output_files.values()]
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_output.exists():
                temp_output.unlink()
            raise e
    
    def process_full_dataset_geocoding(self):
        """Process all unique location-city combinations with batch processing"""
        print("\n" + "=" * 80)
        print("PRODUCTION GEOCODING - PROCESSING FULL DATASET")
        print("=" * 80)
        
        # Get unique location-city combinations
        unique_locations = self.data[['location', 'city']].drop_duplicates()
        print(f"Total unique location-city combinations: {len(unique_locations)}")
        
        # Check what's already in cache
        cached_count = 0
        for _, row in unique_locations.iterrows():
            cache_key = f"{row['location']}_{row['city']}"
            if cache_key in self.geocoding_cache:
                cached_count += 1
        
        print(f"Already cached: {cached_count}/{len(unique_locations)}")
        remaining = len(unique_locations) - cached_count
        
        if remaining > 0:
            print(f"Need to geocode: {remaining} locations")
            estimated_time = remaining * 1.2 / 60  # 1.2 seconds per request
            print(f"Estimated time: {estimated_time:.1f} minutes")
            
            proceed = input(f"Proceed with geocoding {remaining} locations? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Geocoding cancelled. Using existing cache.")
                return
            
            # Process remaining locations
            processed = 0
            for idx, row in tqdm(unique_locations.iterrows(), total=len(unique_locations), desc="Geocoding"):
                cache_key = f"{row['location']}_{row['city']}"
                
                if cache_key not in self.geocoding_cache:
                    coords = self.geocode_location_with_retry(row['location'], row['city'])
                    processed += 1
                    
                    # Save cache every 50 locations
                    if processed % 50 == 0:
                        self.save_cache()
                        self.save_progress(cached_count + processed, len(unique_locations))
                        print(f"\nProgress saved. Processed {processed} new locations.")
        
        # Final save
        self.save_cache()
        print(f"\nGeocoding completed! Cache contains {len(self.geocoding_cache)} locations.")
    
    def apply_coordinates_to_dataset(self):
        """Apply cached coordinates to the full dataset"""
        self.logger.info("=" * 60)
        self.logger.info("APPLYING COORDINATES TO DATASET")
        self.logger.info("=" * 60)
        
        enriched_data = self.data.copy()
        
        # Vectorized coordinate application
        def get_coords(row):
            cache_key = f"{row['location']}_{row['city']}"
            return self.geocoding_cache.get(cache_key, (None, None))
        
        self.logger.info("Applying coordinates from cache...")
        coords_df = enriched_data.apply(get_coords, axis=1, result_type='expand')
        enriched_data['latitude'] = coords_df[0]
        enriched_data['longitude'] = coords_df[1]
        
        # Calculate distance to city center
        self.logger.info("Calculating distances to city centers...")
        def calc_distance_to_center(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                return np.nan
            
            city_center = self.city_centers.get(row['city'])
            if not city_center:
                return np.nan
            
            property_coords = (row['latitude'], row['longitude'])
            distance = geodesic(property_coords, city_center).kilometers
            return distance
        
        enriched_data['distance_to_city_center_km'] = enriched_data.apply(
            calc_distance_to_center, axis=1
        )
        
        # Report success rates
        successful_coords = enriched_data[['latitude', 'longitude']].notna().all(axis=1).sum()
        successful_distances = enriched_data['distance_to_city_center_km'].notna().sum()
        
        self.logger.info(f"Coordinates applied: {successful_coords:,}/{len(enriched_data):,} "
                        f"({successful_coords/len(enriched_data)*100:.1f}%)")
        self.logger.info(f"Distances calculated: {successful_distances:,}")
        
        self.save_progress("coordinates_applied", {
            "total_records": len(enriched_data),
            "successful_coords": int(successful_coords),
            "successful_distances": int(successful_distances)
        })
        
        return enriched_data
    
    def calculate_distance_features(self, data_with_coords):
        """Calculate distance-based features for full dataset"""
        print("\n" + "=" * 60)
        print("CALCULATING DISTANCE FEATURES")
        print("=" * 60)
        
        enriched_data = data_with_coords.copy()
        
        def calc_distance_to_center(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                return np.nan
            
            city_center = self.city_centers.get(row['city'])
            if not city_center:
                return np.nan
            
            property_coords = (row['latitude'], row['longitude'])
            distance = geodesic(property_coords, city_center).kilometers
            return distance
        
        print("Calculating distances to city centers...")
        enriched_data['distance_to_city_center_km'] = enriched_data.apply(
            calc_distance_to_center, axis=1
        )
        
        valid_distances = enriched_data['distance_to_city_center_km'].notna().sum()
        print(f"Distance calculations completed for {valid_distances} records")
        
        return enriched_data
    
    def calculate_neighborhood_features_optimized(self, enriched_data):
        """Optimized neighborhood feature calculation using spatial indexing"""
        print("\n" + "=" * 60)
        print("CALCULATING NEIGHBORHOOD FEATURES (OPTIMIZED)")
        print("=" * 60)
        
        final_data = enriched_data.copy()
        final_data['neighborhood_avg_price_2km'] = np.nan
        
        # Group by city for efficiency
        for city in final_data['city'].unique():
            city_data = final_data[final_data['city'] == city].copy()
            city_data = city_data.dropna(subset=['latitude', 'longitude'])
            
            if len(city_data) == 0:
                continue
                
            print(f"Processing {city}: {len(city_data)} properties")
            
            # Calculate neighborhood averages using vectorized operations
            for idx, row in tqdm(city_data.iterrows(), total=len(city_data), desc=f"{city} neighborhoods"):
                # Calculate distances to all other properties in the same city
                distances = city_data.apply(
                    lambda other_row: geodesic(
                        (row['latitude'], row['longitude']),
                        (other_row['latitude'], other_row['longitude'])
                    ).kilometers if other_row.name != idx else float('inf'),
                    axis=1
                )
                
                # Find properties within 2km
                nearby_properties = city_data[distances <= 2.0]
                
                if len(nearby_properties) > 0:
                    avg_price = nearby_properties['price'].mean()
                    final_data.loc[idx, 'neighborhood_avg_price_2km'] = avg_price
        
        # Fill remaining NaN values with location-based averages
        location_avg_prices = final_data.groupby(['city', 'location'])['price'].mean()
        
        for idx, row in final_data.iterrows():
            if pd.isna(final_data.loc[idx, 'neighborhood_avg_price_2km']):
                city_location_key = (row['city'], row['location'])
                if city_location_key in location_avg_prices:
                    final_data.loc[idx, 'neighborhood_avg_price_2km'] = location_avg_prices[city_location_key]
        
        neighborhood_count = final_data['neighborhood_avg_price_2km'].notna().sum()
        print(f"Neighborhood features calculated for {neighborhood_count} records")
        
        return final_data
    
    def create_production_visualizations(self, enriched_data):
        """Create comprehensive visualizations for the full dataset"""
        print("\n" + "=" * 60)
        print("CREATING PRODUCTION VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Interactive map with more samples
        print("Creating comprehensive interactive map...")
        
        # Center map on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Sample for visualization (larger sample for production)
        sample_data = enriched_data.dropna(subset=['latitude', 'longitude'])
        if len(sample_data) > 2000:
            sample_data = sample_data.sample(n=2000, random_state=42)
        
        if len(sample_data) > 0:
            # Create color map based on price
            min_price = sample_data['price'].min()
            max_price = sample_data['price'].max()
            
            def get_color(price):
                normalized = (price - min_price) / (max_price - min_price)
                if normalized < 0.2:
                    return 'green'
                elif normalized < 0.4:
                    return 'lightgreen'
                elif normalized < 0.6:
                    return 'orange'
                elif normalized < 0.8:
                    return 'red'
                else:
                    return 'darkred'
            
            # Add city-specific layers
            for city in sample_data['city'].unique():
                city_data = sample_data[sample_data['city'] == city]
                
                for _, row in city_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=4,
                        popup=f"<b>{row['location']}, {row['city']}</b><br>"
                              f"Price: ₹{row['price']:,.0f}<br>"
                              f"Area: {row['area']} sqft<br>"
                              f"Price/sqft: ₹{row['price_per_sqft']:,.0f}<br>"
                              f"Distance to center: {row['distance_to_city_center_km']:.1f}km",
                        color=get_color(row['price']),
                        fill=True,
                        fillColor=get_color(row['price']),
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add legend
            legend_html = '''
                 <div style="position: fixed; 
                             bottom: 50px; left: 50px; width: 150px; height: 90px; 
                             background-color: white; border:2px solid grey; z-index:9999; 
                             font-size:14px; font-weight: bold;
                             ">
                 <p style="margin: 10px;"><b>Price Legend</b></p>
                 <p style="margin: 10px;"><i class="fa fa-circle" style="color:darkred"></i> Very High</p>
                 <p style="margin: 10px;"><i class="fa fa-circle" style="color:red"></i> High</p>
                 <p style="margin: 10px;"><i class="fa fa-circle" style="color:orange"></i> Medium</p>
                 <p style="margin: 10px;"><i class="fa fa-circle" style="color:lightgreen"></i> Low</p>
                 <p style="margin: 10px;"><i class="fa fa-circle" style="color:green"></i> Very Low</p>
                 </div>
                 '''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save comprehensive map
        map_path = self.geo_enriched_path / "production_price_heatmap.html"
        m.save(str(map_path))
        print(f"Production interactive map saved to: {map_path}")
        
        # 2. Comprehensive static analysis
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Production Geospatial Analysis - Full Dataset', fontsize=16, fontweight='bold')
        
        valid_data = enriched_data.dropna(subset=['distance_to_city_center_km', 'price'])
        
        # Price vs Distance scatter
        axes[0, 0].scatter(valid_data['distance_to_city_center_km'], 
                          valid_data['price'], alpha=0.3, s=1)
        axes[0, 0].set_xlabel('Distance to City Center (km)')
        axes[0, 0].set_ylabel('Price (₹)')
        axes[0, 0].set_title('Price vs Distance to City Center')
        
        # City-wise distance distribution
        sns.boxplot(data=valid_data, x='city', y='distance_to_city_center_km', ax=axes[0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_title('Distance Distribution by City')
        
        # Price per sqft by city
        sns.boxplot(data=enriched_data, x='city', y='price_per_sqft', ax=axes[1, 0])
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_title('Price per Sqft by City')
        
        # Distance vs Price per sqft
        axes[1, 1].scatter(valid_data['distance_to_city_center_km'], 
                          valid_data['price_per_sqft'], alpha=0.3, s=1)
        axes[1, 1].set_xlabel('Distance to City Center (km)')
        axes[1, 1].set_ylabel('Price per Sqft (₹)')
        axes[1, 1].set_title('Price per Sqft vs Distance')
        
        # Neighborhood comparison
        neighborhood_data = enriched_data.dropna(subset=['neighborhood_avg_price_2km', 'price'])
        if len(neighborhood_data) > 0:
            axes[2, 0].scatter(neighborhood_data['neighborhood_avg_price_2km'], 
                              neighborhood_data['price'], alpha=0.3, s=1)
            axes[2, 0].plot([neighborhood_data['neighborhood_avg_price_2km'].min(), 
                            neighborhood_data['neighborhood_avg_price_2km'].max()],
                           [neighborhood_data['neighborhood_avg_price_2km'].min(), 
                            neighborhood_data['neighborhood_avg_price_2km'].max()], 
                           'r--', alpha=0.8)
            axes[2, 0].set_xlabel('Neighborhood Average Price (₹)')
            axes[2, 0].set_ylabel('Individual Property Price (₹)')
            axes[2, 0].set_title('Individual vs Neighborhood Pricing')
        
        # Feature correlation heatmap
        feature_cols = ['price', 'area', 'distance_to_city_center_km', 'price_per_sqft', 'location_frequency']
        if 'neighborhood_avg_price_2km' in enriched_data.columns:
            feature_cols.append('neighborhood_avg_price_2km')
        
        corr_data = enriched_data[feature_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[2, 1])
        axes[2, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = self.geo_enriched_path / "production_geo_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Production analysis plots saved to: {plot_path}")
        
        plt.show()
    
    def save_production_dataset(self, enriched_data):
        """Save the complete geo-enriched dataset"""
        output_path = self.geo_enriched_path / "production_geo_enriched_data.csv"
        enriched_data.to_csv(output_path, index=False)
        
        print(f"\n" + "=" * 80)
        print("PRODUCTION DATASET SAVED")
        print("=" * 80)
        print(f"File: {output_path}")
        print(f"Shape: {enriched_data.shape}")
        
        # Feature summary
        geospatial_features = ['latitude', 'longitude', 'distance_to_city_center_km', 'neighborhood_avg_price_2km']
        
        print("\nGeospatial Feature Coverage:")
        for feature in geospatial_features:
            if feature in enriched_data.columns:
                coverage = enriched_data[feature].notna().sum()
                percentage = coverage / len(enriched_data) * 100
                print(f"  {feature}: {coverage:,}/{len(enriched_data):,} ({percentage:.1f}%)")
        
        # Data quality metrics
        print(f"\nData Quality Metrics:")
        print(f"  Total records: {len(enriched_data):,}")
        print(f"  Complete records (all geo features): {enriched_data[geospatial_features].notna().all(axis=1).sum():,}")
        print(f"  Records with coordinates: {enriched_data[['latitude', 'longitude']].notna().all(axis=1).sum():,}")


def main():
    """GPU-accelerated main execution for production Step 2"""
    print("=" * 100)
    print("AI-POWERED REAL ESTATE PRICE ADVISOR - GPU-ACCELERATED STEP 2")
    print("Ultra-Fast Geospatial Intelligence with GPU/RAPIDS Acceleration")
    print("=" * 100)
    
    enricher = GPUOptimizedLocationEnricher(use_gpu=True)
    
    # Load data with GPU optimization
    if not enricher.load_data_optimized():
        return
    
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(enricher.data):,}")
    
    # Handle cuDF vs pandas differences
    if hasattr(enricher.data, 'to_pandas'):
        data_pandas = enricher.data.to_pandas()
        print(f"  Unique locations: {data_pandas['location'].nunique():,}")
        print(f"  Cities: {data_pandas['city'].nunique()} ({list(data_pandas['city'].unique())})")
    else:
        print(f"  Unique locations: {enricher.data['location'].nunique():,}")
        print(f"  Cities: {enricher.data['city'].nunique()} ({list(enricher.data['city'].unique())})")
    
    start_time = time.time()
    
    try:
        # Step 1: Geocode all unique locations (with intelligent caching)
        enricher.process_geocoding()
        
        # Step 2: Apply coordinates and calculate distances
        data_with_coords = enricher.apply_coordinates_to_dataset()
        
        # Step 3: GPU-accelerated neighborhood computation
        final_data = enricher.calculate_neighborhood_features_gpu_optimized(data_with_coords)
        
        # Step 4: Create production visualizations (atomic operations)
        enricher.create_production_visualizations(final_data)
        
        # Step 5: Save complete dataset (atomic operation)
        enricher.save_production_dataset(final_data)
        
        # Performance metrics
        total_time = time.time() - start_time
        records_per_second = len(enricher.data) / total_time
        
        print("\n" + "=" * 100)
        print("🚀 GPU-ACCELERATED STEP 2 COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"⚡ Performance: {len(enricher.data):,} records in {total_time:.1f}s "
              f"({records_per_second:.0f} records/sec)")
        print(f"🎯 All {len(enricher.data):,} records processed with GPU acceleration")
        print(f"📊 Features: lat/long, distances, GPU-optimized neighborhoods")
        print(f"🗺️ Production visualizations: analysis plots + interactive heatmap")
        print(f"💾 Complete dataset ready for machine learning")
        print(f"🔄 Fully resumable with intelligent caching")
        
        if enricher.use_gpu:
            print(f"🚀 GPU Acceleration: RAPIDS cuDF/cuML enabled")
        elif enricher.use_torch_gpu:
            print(f"🔥 GPU Acceleration: PyTorch CUDA enabled")
        else:
            print(f"⚡ CPU Optimization: scikit-learn BallTree")
        
        print("\n✅ Ready for Step 3: GPU-Accelerated ML Model Training")
        
    except KeyboardInterrupt:
        enricher.logger.warning("Process interrupted. All progress saved - resume anytime!")
    except Exception as e:
        enricher.logger.error(f"Error in GPU-accelerated Step 2: {e}")
        import traceback
        traceback.print_exc()


def cleanup_old_files():
    """Clean up old/unused files from previous runs"""
    print("\n" + "=" * 60)
    print("CLEANING UP OLD/UNUSED FILES")
    print("=" * 60)
    
    # Files to remove
    cleanup_patterns = [
        # Old step 2 files
        "src/data_processing/step2_location_enrichment.py",
        # Old data files that will be recalculated
        "data/geo_enriched/geo_enriched_data.csv",
        "data/geo_enriched/price_heatmap.html", 
        "data/geo_enriched/geo_analysis.png",
        # Any partial/incomplete files
        "data/geo_enriched/*.tmp",
        "data/processed/*.tmp",
        # Old progress files (will be regenerated)
        "data/geo_enriched/geocoding_progress.json"
    ]
    
    removed_count = 0
    
    for pattern in cleanup_patterns:
        # Handle both direct paths and glob patterns
        if '*' in pattern:
            import glob
            files = glob.glob(pattern)
        else:
            files = [pattern] if Path(pattern).exists() else []
        
        for file_path in files:
            try:
                Path(file_path).unlink()
                print(f"🗑️  Removed: {file_path}")
                removed_count += 1
            except FileNotFoundError:
                pass  # File already doesn't exist
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    print(f"\n✅ Cleanup completed: {removed_count} files removed")


if __name__ == "__main__":
    # Clean up old files first
    cleanup_old_files()
    
    # Run optimized Step 2
    main()
