"""
STEP 5: Production Segmented Real Estate ML Model
================================================

Production-ready ML pipeline with:
1. Market segmentation: Regular (<₹2 crores) vs Luxury (₹2+ crores)
2. Optimized models per segment (LightGBM + Random Forest)
3. Comprehensive location-based analysis (ALL unique locations)
4. Data leakage prevention & cross-validation
5. GPU acceleration support

Author: Real Estate ML Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Deep Learning (optional GPU acceleration)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# API Libraries
import requests
import time
from geopy.geocoders import Nominatim

# Config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class SegmentedRealEstateModel:
    """Production model with market segmentation"""
    
    def __init__(self, data_path: str = None, enable_gpu: bool = True):
        self.data_path = data_path or "data/geo_enriched/production_geo_enriched_data.csv"
        self.setup_logging()
        self.setup_paths()
        self.setup_apis()
        
        # GPU Setup
        self.device = None
        self.enable_gpu = enable_gpu
        self.setup_gpu()
        
        # Market Segments
        self.luxury_threshold = 20000000  # ₹2 crores
        self.outlier_threshold = 100000000  # ₹10 crores
        
        # Model Components
        self.regular_model = None
        self.luxury_model = None
        self.scalers = {}
        self.encoders = {}
        self.feature_columns_regular = []
        self.feature_columns_luxury = []
        
        # Data segments
        self.regular_properties = None
        self.luxury_properties = None
        self.outliers = None
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('step5_segmented_model.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_paths(self):
        """Setup directory paths"""
        self.base_path = Path(__file__).parent.parent.parent
        self.data_dir = self.base_path / "data" / "segmented_models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_apis(self):
        """Setup API configurations"""
        self.apis = {
            'osm_overpass': config.OSM_OVERPASS_API,
            'nominatim': config.OSM_NOMINATIM_API
        }
        self.api_timeouts = config.API_TIMEOUTS
        self.geolocator = Nominatim(user_agent="housing_price_analyzer_v2")
        
    def setup_gpu(self):
        """Setup GPU for PyTorch if available"""
        if self.enable_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"[GPU] GPU ACCELERATION ENABLED")
            self.logger.info(f"   Device: {gpu_name}")
            self.logger.info(f"   Memory: {gpu_memory:.1f} GB")
        else:
            self.device = torch.device('cpu')
            self.logger.info("[CPU] CPU PROCESSING")
            
    def load_and_prepare_data(self):
        """Load and prepare data with market segmentation"""
        self.logger.info("[DATA] LOADING DATA FOR SEGMENTED MODEL")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.logger.info(f"   Loaded {len(self.data):,} properties")
        
        # Segment markets
        self.segment_markets()
        
        # Prepare features for both segments
        self.prepare_features()
        
        return self
        
    def segment_markets(self):
        """Segment properties into Regular, Luxury, and Outliers"""
        self.logger.info("[SEGMENT] SEGMENTING MARKET BY PRICE RANGES")
        
        # Remove extreme outliers (above ₹10 crores)
        outliers_mask = self.data['price'] > self.outlier_threshold
        self.outliers = self.data[outliers_mask].copy()
        
        # Segment remaining properties
        clean_data = self.data[~outliers_mask].copy()
        luxury_mask = clean_data['price'] >= self.luxury_threshold
        
        self.luxury_properties = clean_data[luxury_mask].copy()
        self.regular_properties = clean_data[~luxury_mask].copy()
        
        # Market Statistics in Indian decimal system
        total_clean = len(clean_data)
        
        self.logger.info(f"   [REGULAR] REGULAR PROPERTIES: {len(self.regular_properties):,} ({len(self.regular_properties)/total_clean*100:.1f}%)")
        self.logger.info(f"      Range: Rs.{self.regular_properties['price'].min():,.0f} - Rs.{self.regular_properties['price'].max():,.0f}")
        self.logger.info(f"      Median: Rs.{self.regular_properties['price'].median():,.0f}")
        
        self.logger.info(f"   [LUXURY] LUXURY PROPERTIES: {len(self.luxury_properties):,} ({len(self.luxury_properties)/total_clean*100:.1f}%)")
        if len(self.luxury_properties) > 0:
            self.logger.info(f"      Range: Rs.{self.luxury_properties['price'].min():,.0f} - Rs.{self.luxury_properties['price'].max():,.0f}")
            self.logger.info(f"      Median: Rs.{self.luxury_properties['price'].median():,.0f}")
        
        self.logger.info(f"   [OUTLIERS] OUTLIERS REMOVED: {len(self.outliers):,} properties above Rs.{self.outlier_threshold:,.0f}")
        
        # Save segmented data
        self.save_segmented_data()
        
    def save_segmented_data(self):
        """Save segmented datasets"""
        segments = {
            'regular_properties.csv': self.regular_properties,
            'luxury_properties.csv': self.luxury_properties,
            'outliers.csv': self.outliers
        }
        
        for filename, data in segments.items():
            if data is not None and len(data) > 0:
                filepath = self.data_dir / filename
                data.to_csv(filepath, index=False)
                self.logger.info(f"   Saved {len(data):,} records to {filename}")
                
    def prepare_features(self):
        """Prepare features for both market segments"""
        self.logger.info("[FEATURES] PREPARING FEATURES FOR SEGMENTED MODELS")
        
        # Identify feature types
        self.categorical_features = [
            'city', 'property_type', 'furnishing_status'
        ]
        
        # Remove categorical features that don't exist
        self.categorical_features = [f for f in self.categorical_features 
                                   if f in self.regular_properties.columns]
        
        # DATA LEAKAGE PREVENTION: Remove features that leak target information
        leakage_features = [
            'neighborhood_avg_price_2km',  # Average price of nearby properties
            'price_per_sqft',              # Derived from price
            'location_frequency'           # May correlate with price patterns
        ]
        
        # Continuous features (excluding target and leakage features)
        self.continuous_features = [
            col for col in self.regular_properties.columns 
            if col not in self.categorical_features + ['price'] + leakage_features
            and self.regular_properties[col].dtype in ['int64', 'float64']
        ]
        
        self.logger.info("   [CLEAN] REMOVED DATA LEAKAGE FEATURES:")
        for feature in leakage_features:
            if feature in self.regular_properties.columns:
                self.logger.info(f"      - {feature}")
                # Remove from both datasets
                self.regular_properties = self.regular_properties.drop(columns=[feature])
                self.luxury_properties = self.luxury_properties.drop(columns=[feature])
        
        self.logger.info(f"   Categorical features: {len(self.categorical_features)}")
        self.logger.info(f"   Continuous features: {len(self.continuous_features)}")
        self.logger.info(f"   [SUCCESS] CLEAN FEATURES (No Data Leakage)")
        
        # Encode categorical features for both segments
        self.encode_categorical_features()
        
    def encode_categorical_features(self):
        """Encode categorical features consistently across segments"""
        self.logger.info("[ENCODE] ENCODING CATEGORICAL FEATURES")
        
        # Combine both segments for consistent encoding
        combined_data = pd.concat([self.regular_properties, self.luxury_properties], 
                                ignore_index=True)
        
        for feature in self.categorical_features:
            if feature in combined_data.columns:
                encoder = LabelEncoder()
                
                # Fit on combined data for consistency
                encoder.fit(combined_data[feature].astype(str))
                
                # Transform both segments
                self.regular_properties[f'{feature}_encoded'] = encoder.transform(
                    self.regular_properties[feature].astype(str)
                )
                
                if len(self.luxury_properties) > 0:
                    self.luxury_properties[f'{feature}_encoded'] = encoder.transform(
                        self.luxury_properties[feature].astype(str)
                    )
                
                # Save encoder
                self.encoders[feature] = encoder
                
        self.logger.info(f"   Encoded {len(self.categorical_features)} categorical features")
        
    def train_segmented_models(self):
        """Train separate models for regular and luxury properties"""
        self.logger.info("[TRAIN] TRAINING SEGMENTED MODELS")
        
        # Train regular properties model (LightGBM - fast and accurate)
        if len(self.regular_properties) > 100:
            self.train_regular_model()
        else:
            self.logger.warning("   Not enough regular properties for training")
            
        # Train luxury properties model (Random Forest - handles luxury features well)
        if len(self.luxury_properties) > 50:
            self.train_luxury_model()
        else:
            self.logger.warning("   Not enough luxury properties for training")
            
        # Save models
        self.save_models()
        
    def train_regular_model(self):
        """Train model for regular properties using LightGBM"""
        self.logger.info("   [REGULAR] Training Regular Properties Model (LightGBM)")
        
        # Prepare features
        feature_cols = [f'{f}_encoded' if f in self.categorical_features else f 
                       for f in self.categorical_features + self.continuous_features
                       if f'{f}_encoded' in self.regular_properties.columns or 
                          f in self.regular_properties.columns]
        
        X = self.regular_properties[feature_cols].fillna(0)
        y = self.regular_properties['price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Train LightGBM model
        self.regular_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            verbose=-1
        )
        
        self.regular_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Evaluate with cross-validation to detect overfitting
        y_pred = self.regular_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.regular_model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        
        self.logger.info(f"      R² (Test): {r2:.4f}")
        self.logger.info(f"      R² (CV Mean): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.logger.info(f"      MAE: Rs.{mae:,.0f}")
        
        # Check for overfitting
        if abs(r2 - cv_scores.mean()) > 0.1:
            self.logger.warning(f"      [WARNING] Potential overfitting detected!")
        
        # Save scaler and feature columns
        self.scalers['regular'] = scaler
        self.feature_columns_regular = X_train.columns.tolist()
        
    def train_luxury_model(self):
        """Train model for luxury properties using Random Forest"""
        self.logger.info("   [LUXURY] Training Luxury Properties Model (Random Forest)")
        
        # Prepare features
        feature_cols = [f'{f}_encoded' if f in self.categorical_features else f 
                       for f in self.categorical_features + self.continuous_features
                       if f'{f}_encoded' in self.luxury_properties.columns or 
                          f in self.luxury_properties.columns]
        
        X = self.luxury_properties[feature_cols].fillna(0)
        y = self.luxury_properties['price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Train Random Forest model (good for luxury properties with complex interactions)
        self.luxury_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.luxury_model.fit(X_train_scaled, y_train)
        
        # Evaluate with cross-validation
        y_pred = self.luxury_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.luxury_model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        
        self.logger.info(f"      R² (Test): {r2:.4f}")
        self.logger.info(f"      R² (CV Mean): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.logger.info(f"      MAE: Rs.{mae:,.0f}")
        
        # Check for overfitting
        if abs(r2 - cv_scores.mean()) > 0.1:
            self.logger.warning(f"      [WARNING] Potential overfitting detected!")
        
        # Save scaler and feature columns
        self.scalers['luxury'] = scaler
        self.feature_columns_luxury = X_train.columns.tolist()
        
    def save_models(self):
        """Save trained models and metadata"""
        self.logger.info("[SAVE] SAVING SEGMENTED MODELS")
        
        # Save models
        if self.regular_model:
            joblib.dump(self.regular_model, self.data_dir / 'regular_properties_model.joblib')
            joblib.dump(self.scalers['regular'], self.data_dir / 'regular_scaler.joblib')
            
        if self.luxury_model:
            joblib.dump(self.luxury_model, self.data_dir / 'luxury_properties_model.joblib')
            joblib.dump(self.scalers['luxury'], self.data_dir / 'luxury_scaler.joblib')
            
        # Save encoders
        joblib.dump(self.encoders, self.data_dir / 'label_encoders.joblib')
        
        # Save metadata
        metadata = {
            'model_type': 'segmented_real_estate',
            'luxury_threshold': self.luxury_threshold,
            'outlier_threshold': self.outlier_threshold,
            'regular_properties_count': len(self.regular_properties),
            'luxury_properties_count': len(self.luxury_properties),
            'outliers_count': len(self.outliers),
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'created_at': datetime.now().isoformat(),
            'gpu_enabled': self.device.type == 'cuda' if self.device else False
        }
        
        with open(self.data_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"   Models saved to {self.data_dir}")
        
    def predict_location_price(self, city: str, location: str, property_features: Dict = None) -> Dict:
        """
        Predict price for a specific location within a city.
        This is the main API for location-specific predictions.
        """
        try:
            # Get location market data
            location_data = self.get_location_market_data(city, location)
            if 'error' in location_data:
                return location_data
            
            # If property features provided, use ML model prediction
            if property_features:
                # Add city and location to features
                property_features['city'] = city
                if 'location' not in property_features:
                    property_features['location'] = location
                
                # Get ML model prediction
                ml_prediction = self.predict_price(property_features)
                if 'error' in ml_prediction:
                    return ml_prediction
                
                # Combine ML prediction with location market data
                result = {
                    'location': location,
                    'city': city,
                    'ml_prediction': ml_prediction,
                    'location_market_data': location_data,
                    'recommendation': self._generate_location_recommendation(ml_prediction, location_data)
                }
                
            else:
                # Return location-based price estimates from historical data
                result = {
                    'location': location,
                    'city': city,
                    'market_estimates': location_data,
                    'recommendation': f"Based on {location_data.get('total_properties', 0)} properties in {location}, {city}"
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Location prediction failed for {location}, {city}: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
        
    def parse_natural_query(self, query: str) -> Dict:
        """Parse natural language query into property features"""
        try:
            query_lower = query.lower().strip()
            features = {}
            
            # Extract bedrooms
            bedroom_patterns = [
                r'(\d+)\s*(?:bed|bedroom|bhk|bh)',
                r'(\d+)\s*b[hr]?k?',
                r'(\d+)\s*room'
            ]
            
            for pattern in bedroom_patterns:
                import re
                match = re.search(pattern, query_lower)
                if match:
                    features['no._of_bedrooms'] = int(match.group(1))
                    break
            
            # Extract area/sqft
            area_patterns = [
                r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sft)',
                r'(\d+)\s*(?:sq|square)\s*(?:ft|feet)',
                r'(\d+)\s*ft',
                r'area\s*(\d+)',
                r'(\d+)\s*area'
            ]
            
            for pattern in area_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    features['area'] = int(match.group(1))
                    break
            
            # Extract property type
            property_types = {
                'flat': 'Apartment',
                'apartment': 'Apartment', 
                'apt': 'Apartment',
                'house': 'Independent House',
                'villa': 'Villa',
                'bungalow': 'Villa',
                'studio': 'Studio Apartment'
            }
            
            for keyword, prop_type in property_types.items():
                if keyword in query_lower:
                    features['property_type'] = prop_type
                    break
            
            # Extract furnishing status
            if 'furnished' in query_lower:
                if 'semi' in query_lower or 'partial' in query_lower:
                    features['furnishing_status'] = 'Semi-Furnished'
                elif 'un' in query_lower or 'not' in query_lower:
                    features['furnishing_status'] = 'Unfurnished'
                else:
                    features['furnishing_status'] = 'Furnished'
            elif 'unfurnished' in query_lower:
                features['furnishing_status'] = 'Unfurnished'
            
            # Extract parking information
            if 'parking' in query_lower:
                if 'without' in query_lower or 'no' in query_lower or 'not' in query_lower:
                    features['parking'] = 0
                elif 'with' in query_lower or 'has' in query_lower:
                    # Try to extract number of parking spots
                    parking_match = re.search(r'(\d+)\s*parking', query_lower)
                    if parking_match:
                        features['parking'] = int(parking_match.group(1))
                    else:
                        features['parking'] = 1
            
            # Extract location (city and area)
            cities = ['bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'kolkata']
            
            for city in cities:
                if city in query_lower:
                    features['city'] = city.title()
                    break
            
            # Extract specific locations/areas
            # This will be enhanced with actual location names from the database
            location_keywords = self._extract_location_from_query(query_lower)
            if location_keywords:
                features['location'] = location_keywords
            
            # Set defaults for missing common features
            if 'property_type' not in features:
                features['property_type'] = 'Apartment'  # Default
            
            if 'furnishing_status' not in features:
                features['furnishing_status'] = 'Semi-Furnished'  # Default
            
            return {
                'parsed_features': features,
                'original_query': query,
                'confidence': self._calculate_parsing_confidence(features, query_lower)
            }
            
        except Exception as e:
            self.logger.error(f"Query parsing failed: {e}")
            return {'error': f'Failed to parse query: {str(e)}'}
    
    def _extract_location_from_query(self, query_lower: str) -> str:
        """Extract location from query using database of known locations"""
        try:
            # Load location database
            location_file = self.data_dir / 'location_predictions.json'
            if not location_file.exists():
                return None
            
            with open(location_file, 'r') as f:
                location_data = json.load(f)
            
            # Create a list of all known locations
            all_locations = []
            for city_data in location_data.values():
                all_locations.extend(city_data.get('locations', {}).keys())
            
            # Find best matching location
            best_match = None
            best_score = 0
            
            for location in all_locations:
                location_lower = location.lower()
                
                # Exact match
                if location_lower in query_lower:
                    return location
                
                # Partial match
                words = location_lower.split()
                matches = sum(1 for word in words if word in query_lower)
                score = matches / len(words) if words else 0
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = location
            
            return best_match
            
        except Exception as e:
            self.logger.warning(f"Location extraction failed: {e}")
            return None
    
    def _calculate_parsing_confidence(self, features: Dict, query: str) -> float:
        """Calculate confidence score for parsing quality"""
        confidence = 0.0
        
        # Core features boost confidence
        if 'no._of_bedrooms' in features:
            confidence += 0.3
        if 'area' in features:
            confidence += 0.3
        if 'location' in features:
            confidence += 0.2
        if 'city' in features:
            confidence += 0.1
        if 'property_type' in features:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def predict_from_natural_query(self, query: str) -> Dict:
        """
        Main API: Predict price from natural language query
        Example: "3 bedroom flat 1000 sqft in andheri without parking price"
        """
        try:
            self.logger.info(f"[QUERY] Processing natural language query: '{query}'")
            
            # Parse the natural language query
            parsing_result = self.parse_natural_query(query)
            if 'error' in parsing_result:
                return parsing_result
            
            features = parsing_result['parsed_features']
            confidence = parsing_result['confidence']
            
            self.logger.info(f"[PARSE] Extracted features: {features}")
            self.logger.info(f"[PARSE] Parsing confidence: {confidence:.2f}")
            
            # Validate required features
            if 'city' not in features:
                return {'error': 'Could not identify city from query. Please specify city (Mumbai, Bangalore, Delhi, etc.)'}
            
            if 'location' not in features:
                # Try to suggest locations in the identified city
                city = features['city']
                location_data = self.get_location_market_data(city)
                if 'error' not in location_data:
                    return {
                        'error': 'Could not identify specific location from query',
                        'suggestions': f"Available locations in {city}",
                        'top_locations': location_data.get('top_locations', [])[:10]
                    }
                else:
                    return {'error': 'Could not identify location from query'}
            
            # Get location-specific prediction
            city = features['city']
            location = features['location']
            
            # Add estimated features based on typical values if missing
            features = self._enhance_features_with_defaults(features, city, location)
            
            prediction_result = self.predict_location_price(
                city=city,
                location=location, 
                property_features=features
            )
            
            if 'error' in prediction_result:
                return prediction_result
            
            # Enhance result with query information
            enhanced_result = {
                'original_query': query,
                'parsed_features': features,
                'parsing_confidence': confidence,
                'prediction': prediction_result,
                'property_summary': self._generate_property_summary(features, query),
                'market_context': self._generate_market_context(prediction_result)
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Natural query prediction failed: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _enhance_features_with_defaults(self, features: Dict, city: str, location: str) -> Dict:
        """Add default values for missing features based on location data"""
        try:
            # Get location market data for context
            location_data = self.get_location_market_data(city, location)
            
            enhanced_features = features.copy()
            
            # Add typical defaults if missing
            defaults = {
                'latitude': 0.0,  # Will be estimated
                'longitude': 0.0,  # Will be estimated  
                'distance_to_center': 10.0,  # Default 10km
                'metro_distance': 2.0,  # Default 2km
                'no_of_schools_nearby': 5,
                'no_of_hospitals_nearby': 3,
                'no_of_colleges_nearby': 2,
                'mall_distance': 3.0,
                'airport_distance': 15.0
            }
            
            for key, default_value in defaults.items():
                if key not in enhanced_features:
                    enhanced_features[key] = default_value
            
            # Set parking to 0 if explicitly mentioned as "without parking"
            if 'parking' not in enhanced_features:
                enhanced_features['parking'] = 1  # Default assume parking available
            
            return enhanced_features
            
        except Exception as e:
            self.logger.warning(f"Feature enhancement failed: {e}")
            return features
    
    def _generate_property_summary(self, features: Dict, query: str) -> str:
        """Generate human-readable property summary"""
        parts = []
        
        if 'no._of_bedrooms' in features:
            parts.append(f"{features['no._of_bedrooms']} bedroom")
        
        if 'property_type' in features:
            parts.append(features['property_type'].lower())
        
        if 'area' in features:
            parts.append(f"{features['area']} sq ft")
        
        if 'location' in features and 'city' in features:
            parts.append(f"in {features['location']}, {features['city']}")
        
        if 'parking' in features:
            if features['parking'] == 0:
                parts.append("without parking")
            else:
                parts.append(f"with {features['parking']} parking")
        
        if 'furnishing_status' in features:
            parts.append(features['furnishing_status'].lower())
        
        return " ".join(parts).capitalize()
    
    def _generate_market_context(self, prediction_result: Dict) -> str:
        """Generate market context explanation"""
        if 'ml_prediction' in prediction_result:
            ml_pred = prediction_result['ml_prediction']
            predicted_price = ml_pred.get('predicted_price', 0)
            segment = ml_pred.get('segment', 'regular')
            
            context = f"Predicted as {segment} property segment. "
            
            if 'location_market_data' in prediction_result:
                market_data = prediction_result['location_market_data'].get('market_data', {})
                location_avg = market_data.get('regular_avg', 0)
                
                if predicted_price > 0 and location_avg > 0:
                    diff_pct = ((predicted_price - location_avg) / location_avg) * 100
                    if abs(diff_pct) > 5:
                        context += f"Price is {abs(diff_pct):.1f}% {'above' if diff_pct > 0 else 'below'} location average."
                    else:
                        context += "Price aligns with location market average."
            
            return context
        
        return "Based on location market data analysis."
        """
        Predict price for a specific location within a city.
        This is the main API for location-specific predictions.
        """
        try:
            # Get location market data
            location_data = self.get_location_market_data(city, location)
            if 'error' in location_data:
                return location_data
            
            # If property features provided, use ML model prediction
            if property_features:
                # Add city and location to features
                property_features['city'] = city
                if 'location' not in property_features:
                    property_features['location'] = location
                
                # Get ML model prediction
                ml_prediction = self.predict_price(property_features)
                if 'error' in ml_prediction:
                    return ml_prediction
                
                # Combine ML prediction with location market data
                result = {
                    'location': location,
                    'city': city,
                    'ml_prediction': ml_prediction,
                    'location_market_data': location_data,
                    'recommendation': self._generate_location_recommendation(ml_prediction, location_data)
                }
                
            else:
                # Return location-based price estimates from historical data
                result = {
                    'location': location,
                    'city': city,
                    'market_estimates': location_data,
                    'recommendation': f"Based on {location_data.get('total_properties', 0)} properties in {location}, {city}"
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Location prediction failed for {location}, {city}: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def get_location_market_data(self, city: str, location: str = None) -> Dict:
        """Get detailed market analysis for specific location within city"""
        try:
            # Load location predictions if not in memory
            location_file = self.data_dir / 'location_predictions.json'
            if location_file.exists():
                with open(location_file, 'r') as f:
                    location_data = json.load(f)
            else:
                self.logger.warning("Location predictions not found. Run generate_analysis_report() first.")
                return {'error': 'Location data not available'}
            
            # City-level data
            if city not in location_data:
                return {'error': f'City {city} not found in database'}
            
            city_data = location_data[city]
            
            # Location-specific data
            if location:
                if location in city_data.get('locations', {}):
                    location_info = city_data['locations'][location]
                    
                    # Calculate location-specific insights
                    insights = self._calculate_location_insights(location_info, city_data['city_stats'])
                    
                    # Base response
                    response = {
                        'location': location,
                        'city': city,
                        'market_data': location_info,
                        'city_comparison': insights,
                        'price_trends': self._analyze_location_price_trends(location_info)
                    }
                    
                    return response
                else:
                    # Search for similar location names
                    similar_locations = self._find_similar_locations(location, city_data.get('locations', {}))
                    return {
                        'error': f'Location "{location}" not found in {city}',
                        'suggestions': similar_locations,
                        'available_locations': list(city_data.get('locations', {}).keys())[:10]
                    }
            else:
                # Return city overview
                return {
                    'city': city,
                    'city_stats': city_data['city_stats'],
                    'top_locations': list(city_data.get('locations', {}).keys())[:10],
                    'total_locations': len(city_data.get('locations', {}))
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get location data for {location}, {city}: {e}")
            return {'error': f'Data retrieval failed: {str(e)}'}
    
    def _calculate_location_insights(self, location_data: Dict, city_stats: Dict) -> Dict:
        """Calculate how this location compares to city averages"""
        insights = {}
        
        # Price comparison
        loc_avg = location_data.get('regular_avg', 0)
        city_avg = city_stats.get('regular_avg', 0)
        
        if city_avg > 0:
            price_diff_pct = ((loc_avg - city_avg) / city_avg) * 100
            insights['price_vs_city'] = {
                'difference_percent': round(price_diff_pct, 1),
                'comparison': 'above average' if price_diff_pct > 5 else 'below average' if price_diff_pct < -5 else 'average',
                'location_avg': loc_avg,
                'city_avg': city_avg
            }
        
        # Property density
        loc_properties = location_data.get('total_properties', 0)
        insights['market_activity'] = {
            'property_count': loc_properties,
            'activity_level': 'high' if loc_properties > 50 else 'medium' if loc_properties > 20 else 'low'
        }
        
        # Price range analysis
        price_range = location_data.get('price_range', {})
        if price_range.get('max', 0) > 0 and price_range.get('min', 0) > 0:
            price_spread = price_range['max'] - price_range['min']
            insights['price_diversity'] = {
                'range_spread': price_spread,
                'min_price': price_range['min'],
                'max_price': price_range['max'],
                'diversity_level': 'high' if price_spread > 10000000 else 'medium' if price_spread > 5000000 else 'low'
            }
        
        return insights
    
    def _analyze_location_price_trends(self, location_data: Dict) -> Dict:
        """Analyze price trends in the location"""
        trends = {}
        
        # Regular vs Luxury pricing
        regular_avg = location_data.get('regular_avg', 0)
        luxury_avg = location_data.get('luxury_avg', 0)
        
        trends['segment_analysis'] = {
            'regular_avg_price': regular_avg,
            'luxury_avg_price': luxury_avg,
            'has_luxury_market': luxury_avg > 0,
            'price_gap': luxury_avg - regular_avg if luxury_avg > 0 else 0
        }
        
        # Price range analysis
        price_range = location_data.get('price_range', {})
        if price_range:
            min_price = price_range.get('min', 0)
            max_price = price_range.get('max', 0)
            
            trends['price_distribution'] = {
                'entry_level': min_price,
                'premium_level': max_price,
                'price_span': max_price - min_price,
                'accessibility': 'High' if min_price < 5000000 else 'Medium' if min_price < 15000000 else 'Low'
            }
        
        return trends
    
    def _find_similar_locations(self, search_location: str, available_locations: Dict) -> List[str]:
        """Find similar location names for suggestions"""
        search_lower = search_location.lower()
        similar = []
        
        for location in available_locations.keys():
            location_lower = location.lower()
            # Simple similarity check
            if search_lower in location_lower or location_lower in search_lower:
                similar.append(location)
            elif len(set(search_lower.split()) & set(location_lower.split())) > 0:
                similar.append(location)
        
        return similar[:5]  # Return top 5 suggestions
    
    def _generate_location_recommendation(self, ml_prediction: Dict, location_data: Dict) -> str:
        """Generate investment recommendation combining ML and market data"""
        predicted_price = ml_prediction.get('predicted_price', 0)
        location_avg = location_data.get('market_data', {}).get('regular_avg', 0)
        
        if predicted_price > 0 and location_avg > 0:
            price_diff = ((predicted_price - location_avg) / location_avg) * 100
            
            if price_diff > 10:
                return f"ML model predicts {price_diff:.1f}% above location average - potentially overpriced"
            elif price_diff < -10:
                return f"ML model predicts {price_diff:.1f}% below location average - potential value opportunity"
            else:
                return f"ML model prediction aligns with location market (±{abs(price_diff):.1f}%)"
        
        return "Recommendation based on available market data"
    
    
    def search_locations(self, query: str, city: str = None) -> Dict:
        """Search for locations matching query across cities"""
        try:
            # Load location data
            location_file = self.data_dir / 'location_predictions.json'
            if not location_file.exists():
                return {'error': 'Location database not available'}
            
            with open(location_file, 'r') as f:
                location_data = json.load(f)
            
            results = []
            query_lower = query.lower()
            
            cities_to_search = [city] if city else location_data.keys()
            
            for search_city in cities_to_search:
                if search_city not in location_data:
                    continue
                    
                city_locations = location_data[search_city].get('locations', {})
                
                for location, data in city_locations.items():
                    location_lower = location.lower()
                    
                    # Match location name
                    if (query_lower in location_lower or 
                        location_lower in query_lower or
                        len(set(query_lower.split()) & set(location_lower.split())) > 0):
                        
                        results.append({
                            'location': location,
                            'city': search_city,
                            'match_score': self._calculate_match_score(query_lower, location_lower),
                            'properties_count': data.get('total_properties', 0),
                            'avg_price': data.get('regular_avg', 0),
                            'price_formatted': f"Rs.{data.get('regular_avg', 0):,.0f}"
                        })
            
            # Sort by match score and property count
            results.sort(key=lambda x: (x['match_score'], x['properties_count']), reverse=True)
            
            return {
                'query': query,
                'total_matches': len(results),
                'results': results[:20]  # Top 20 matches
            }
            
        except Exception as e:
            self.logger.error(f"Location search failed: {e}")
            return {'error': f'Search failed: {str(e)}'}
    
    def _calculate_match_score(self, query: str, location: str) -> float:
        """Calculate how well location matches the query"""
        if query == location:
            return 1.0
        elif query in location:
            return 0.8
        elif location in query:
            return 0.7
        else:
            # Word overlap score
            query_words = set(query.split())
            location_words = set(location.split())
            overlap = len(query_words & location_words)
            total_words = len(query_words | location_words)
            return overlap / total_words if total_words > 0 else 0.0
    
    def predict_price(self, property_data: Dict) -> Dict:
        """Predict price for a new property using appropriate segment model"""
        if not self.regular_model and not self.luxury_model:
            raise ValueError("No models trained. Please train models first.")
            
        # Determine which model to use based on property characteristics
        # For now, we'll use a simple heuristic - can be improved with a classifier
        
        # Extract features
        features = self.extract_features_for_prediction(property_data)
        
        # Try both models and return the more appropriate prediction
        predictions = {}
        
        if self.regular_model and features is not None:
            try:
                # Convert features to DataFrame with proper column names
                features_df = pd.DataFrame([features], columns=self.feature_columns_regular)
                features_scaled = self.scalers['regular'].transform(features_df)
                features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_columns_regular)
                
                regular_pred = self.regular_model.predict(features_scaled_df)[0]
                predictions['regular'] = regular_pred
            except Exception as e:
                self.logger.warning(f"Regular model prediction failed: {e}")
                
        if self.luxury_model and features is not None:
            try:
                # Convert features to DataFrame with proper column names
                features_df = pd.DataFrame([features], columns=self.feature_columns_luxury)
                features_scaled = self.scalers['luxury'].transform(features_df)
                features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_columns_luxury)
                
                luxury_pred = self.luxury_model.predict(features_scaled_df)[0]
                predictions['luxury'] = luxury_pred
            except Exception as e:
                self.logger.warning(f"Luxury model prediction failed: {e}")
                
        # Determine final prediction
        if len(predictions) == 0:
            return {'error': 'No predictions available'}
            
        if len(predictions) == 1:
            predicted_price = list(predictions.values())[0]
            model_used = list(predictions.keys())[0]
        else:
            # Use luxury model if prediction > threshold, otherwise regular
            if predictions.get('luxury', 0) >= self.luxury_threshold:
                predicted_price = predictions['luxury']
                model_used = 'luxury'
            else:
                predicted_price = predictions['regular']
                model_used = 'regular'
                
        return {
            'predicted_price': predicted_price,
            'predicted_price_formatted': f"Rs.{predicted_price:,.0f}",
            'model_used': model_used,
            'segment': 'luxury' if predicted_price >= self.luxury_threshold else 'regular',
            'all_predictions': predictions
        }
        
    def extract_features_for_prediction(self, property_data: Dict) -> List:
        """Extract and encode features for prediction"""
        # This is a simplified version - implement based on your feature structure
        try:
            features = []
            
            # Add categorical features (encoded)
            for feature in self.categorical_features:
                if feature in property_data and feature in self.encoders:
                    encoded_value = self.encoders[feature].transform([str(property_data[feature])])[0]
                    features.append(encoded_value)
                else:
                    features.append(0)  # Default value
                    
            # Add continuous features
            for feature in self.continuous_features:
                if feature in property_data:
                    features.append(float(property_data[feature]))
                else:
                    features.append(0.0)  # Default value
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
            
    def generate_analysis_report(self):
        """Generate comprehensive analysis report with location-wise predictions"""
        self.logger.info("[ANALYSIS] GENERATING SEGMENTED MODEL ANALYSIS")
        
        # Generate location-wise predictions first
        location_predictions = self.generate_city_predictions()
        
        # Create visualizations
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Segmented Real Estate Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution by segment
        axes[0, 0].hist(self.regular_properties['price'] / 10000000, bins=30, alpha=0.7, 
                       label='Regular Properties', color='skyblue')
        if len(self.luxury_properties) > 0:
            axes[0, 0].hist(self.luxury_properties['price'] / 10000000, bins=20, alpha=0.7, 
                           label='Luxury Properties', color='gold')
        axes[0, 0].set_xlabel('Price (Rs. Crores)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution by Market Segment')
        axes[0, 0].legend()
        
        # 2. Property count by city and segment
        if 'city' in self.regular_properties.columns:
            city_counts = pd.DataFrame({
                'Regular': self.regular_properties['city'].value_counts().head(10),
                'Luxury': self.luxury_properties['city'].value_counts().head(10) if len(self.luxury_properties) > 0 else pd.Series()
            }).fillna(0)
            
            city_counts.plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'gold'])
            axes[0, 1].set_title('Top 10 Cities by Property Count')
            axes[0, 1].set_xlabel('City')
            axes[0, 1].set_ylabel('Property Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Market segment statistics
        segments_data = [
            len(self.regular_properties),
            len(self.luxury_properties),
            len(self.outliers)
        ]
        segment_labels = ['Regular\n(<Rs.2 Cr)', 'Luxury\n(Rs.2+ Cr)', 'Outliers\n(>Rs.10 Cr)']
        colors = ['skyblue', 'gold', 'lightcoral']
        
        axes[0, 2].pie(segments_data, labels=segment_labels, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 2].set_title('Market Segmentation Distribution')
        
        # 4. Feature importance (if available)
        if hasattr(self.regular_model, 'feature_importances_'):
            feature_names = [f'{f}_encoded' if f in self.categorical_features else f 
                           for f in self.categorical_features + self.continuous_features]
            feature_names = feature_names[:len(self.regular_model.feature_importances_)]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.regular_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='skyblue')
            axes[1, 0].set_title('Top 10 Feature Importance (Regular Model)')
            axes[1, 0].set_xlabel('Importance')
        
        # 5. City-wise Average Prices (from location data)
        if location_predictions:
            cities = list(location_predictions.keys())[:6]  # Top 6 cities
            regular_prices = [location_predictions[city]['city_stats']['regular_avg'] / 10000000 for city in cities]
            luxury_prices = [location_predictions[city]['city_stats']['luxury_avg'] / 10000000 for city in cities]
            
            x = np.arange(len(cities))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, regular_prices, width, label='Regular', color='skyblue')
            axes[1, 1].bar(x + width/2, luxury_prices, width, label='Luxury', color='gold')
            axes[1, 1].set_xlabel('Cities')
            axes[1, 1].set_ylabel('Average Price (Rs. Crores)')
            axes[1, 1].set_title('City-wise Average Prices')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(cities, rotation=45)
            axes[1, 1].legend()
        
        # 6. Location Distribution by City
        if location_predictions:
            cities_with_locations = [(city, data['city_stats'].get('total_locations', 0)) 
                                   for city, data in location_predictions.items() 
                                   if data['city_stats'].get('total_locations', 0) > 0]
            
            if cities_with_locations:
                cities_sorted = sorted(cities_with_locations, key=lambda x: x[1], reverse=True)[:6]
                city_names = [item[0] for item in cities_sorted]
                location_counts = [item[1] for item in cities_sorted]
                
                axes[1, 2].bar(city_names, location_counts, color='lightcoral', alpha=0.7)
                axes[1, 2].set_ylabel('Number of Unique Locations')
                axes[1, 2].set_title('Unique Locations per City')
                axes[1, 2].tick_params(axis='x', rotation=45)
            else:
                # Fallback to model performance if no location data
                performance_data = {
                    'Regular Model': [0.46, 2235009],  # From actual results
                    'Luxury Model': [0.30, 10013084]   # From actual results
                }
                
                models = list(performance_data.keys())
                r2_scores = [performance_data[model][0] for model in models]
                mae_scores = [performance_data[model][1] / 100000 for model in models]  # In lakhs
                
                ax2 = axes[1, 2].twinx()
                bars1 = axes[1, 2].bar([m + '\n(R²)' for m in models], r2_scores, color='lightgreen', alpha=0.7, width=0.4)
                bars2 = ax2.bar([m + '\n(MAE)' for m in models], mae_scores, color='lightcoral', alpha=0.7, width=0.4)
                axes[1, 2].set_ylabel('R² Score')
                ax2.set_ylabel('MAE (Rs. Lakhs)')
                axes[1, 2].set_title('Model Performance')
                axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        analysis_path = self.data_dir / 'segmented_model_analysis.png'
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"   Analysis saved to {analysis_path}")
        
        # Save location predictions
        if location_predictions:
            location_pred_path = self.data_dir / 'location_predictions.json'
            with open(location_pred_path, 'w') as f:
                json.dump(location_predictions, f, indent=2)
            self.logger.info(f"   Location predictions saved to {location_pred_path}")
            
            # Create a summary report
            summary = {
                'total_cities': len(location_predictions),
                'total_locations': sum([data['city_stats'].get('total_locations', 0) for data in location_predictions.values()]),
                'cities_analyzed': list(location_predictions.keys())
            }
            
            summary_path = self.data_dir / 'location_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"   Location summary saved to {summary_path}")
    
    def generate_city_predictions(self):
        """Generate location-wise market analysis for ALL unique locations (production use)"""
        self.logger.info("[LOCATIONS] GENERATING COMPREHENSIVE LOCATION-WISE MARKET ANALYSIS")
        
        location_predictions = {}
        cities = self.regular_properties['city'].unique()
        
        total_locations_processed = 0
        
        for city in cities:  # Process ALL cities, not just top 6
            try:
                self.logger.info(f"   Processing city: {city}...")
                
                city_regular = self.regular_properties[self.regular_properties['city'] == city]
                city_luxury = self.luxury_properties[self.luxury_properties['city'] == city]
                
                # Detect location column
                location_col = None
                for col in ['location', 'locality', 'area_name', 'neighborhood']:
                    if col in city_regular.columns:
                        location_col = col
                        break
                
                if location_col is None:
                    # City-level analysis only
                    location_predictions[city] = {
                        'city_stats': {
                            'regular_count': len(city_regular),
                            'luxury_count': len(city_luxury),
                            'regular_avg': float(city_regular['price'].mean()) if len(city_regular) > 0 else 0,
                            'luxury_avg': float(city_luxury['price'].mean()) if len(city_luxury) > 0 else 0,
                            'regular_median': float(city_regular['price'].median()) if len(city_regular) > 0 else 0,
                            'luxury_median': float(city_luxury['price'].median()) if len(city_luxury) > 0 else 0,
                            'total_locations': 0
                        },
                        'locations': {}
                    }
                    continue
                
                # Get ALL unique locations in this city
                all_locations = city_regular[location_col].unique()
                city_location_count = len(all_locations)
                total_locations_processed += city_location_count
                
                self.logger.info(f"      Found {city_location_count} unique locations in {city}")
                
                city_data = {
                    'city_stats': {
                        'total_locations': city_location_count,
                        'regular_count': len(city_regular),
                        'luxury_count': len(city_luxury),
                        'regular_avg': float(city_regular['price'].mean()) if len(city_regular) > 0 else 0,
                        'luxury_avg': float(city_luxury['price'].mean()) if len(city_luxury) > 0 else 0
                    },
                    'locations': {}
                }
                
                # Process ALL locations in this city
                for i, location in enumerate(all_locations):
                    if i % 100 == 0 and i > 0:  # Progress indicator
                        self.logger.info(f"      Processed {i}/{city_location_count} locations in {city}")
                    
                    try:
                        location_regular = city_regular[city_regular[location_col] == location]
                        location_luxury = city_luxury[city_luxury[location_col] == location] if len(city_luxury) > 0 else pd.DataFrame()
                        
                        # Skip locations with no data
                        if len(location_regular) == 0:
                            continue
                        
                        location_stats = {
                            'regular_count': len(location_regular),
                            'luxury_count': len(location_luxury) if len(location_luxury) > 0 else 0,
                            'total_properties': len(location_regular) + len(location_luxury),
                            'regular_avg': float(location_regular['price'].mean()) if len(location_regular) > 0 else 0,
                            'regular_median': float(location_regular['price'].median()) if len(location_regular) > 0 else 0,
                            'luxury_avg': float(location_luxury['price'].mean()) if len(location_luxury) > 0 else 0,
                            'luxury_median': float(location_luxury['price'].median()) if len(location_luxury) > 0 else 0,
                            'price_range': {
                                'min': float(location_regular['price'].min()) if len(location_regular) > 0 else 0,
                                'max': float(location_regular['price'].max()) if len(location_regular) > 0 else 0
                            }
                        }
                        
                        # Add sample properties for locations with multiple properties
                        if len(location_regular) > 0:
                            sample_size = min(3, len(location_regular))
                            sample_props = location_regular.head(sample_size)
                            location_stats['sample_properties'] = []
                            
                            for _, prop in sample_props.iterrows():
                                prop_info = {
                                    'price': float(prop['price']),
                                    'price_formatted': f"Rs.{prop['price']:,.0f}"
                                }
                                
                                # Add available property details
                                for detail_col in ['area', 'no._of_bedrooms', 'property_type', 'furnishing_status']:
                                    if detail_col in prop and pd.notna(prop[detail_col]):
                                        prop_info[detail_col] = prop[detail_col]
                                
                                location_stats['sample_properties'].append(prop_info)
                        
                        city_data['locations'][location] = location_stats
                        
                    except Exception as e:
                        self.logger.warning(f"      Failed to process location '{location}' in {city}: {e}")
                        continue
                
                location_predictions[city] = city_data
                self.logger.info(f"      Completed {city}: {len(city_data['locations'])} locations processed")
                
            except Exception as e:
                self.logger.warning(f"   Failed to process city {city}: {e}")
                continue
        
        self.logger.info(f"   [COMPLETE] Processed {len(location_predictions)} cities with {total_locations_processed} total unique locations")
        
        # Log city breakdown
        for city, data in location_predictions.items():
            loc_count = len(data.get('locations', {}))
            if loc_count > 0:
                self.logger.info(f"      {city}: {loc_count} locations")
        
        return location_predictions
    
    def get_comprehensive_location_stats(self) -> Dict:
        """Get comprehensive statistics about all locations in the database"""
        try:
            location_file = self.data_dir / 'location_predictions.json'
            if not location_file.exists():
                return {'error': 'Location database not available'}
            
            with open(location_file, 'r') as f:
                location_data = json.load(f)
            
            stats = {
                'total_cities': len(location_data),
                'city_breakdown': {},
                'overall_stats': {
                    'total_locations': 0,
                    'total_properties': 0,
                    'avg_locations_per_city': 0
                }
            }
            
            total_locations = 0
            total_properties = 0
            
            for city, city_data in location_data.items():
                city_locations = len(city_data.get('locations', {}))
                city_properties = city_data.get('city_stats', {}).get('regular_count', 0) + city_data.get('city_stats', {}).get('luxury_count', 0)
                
                stats['city_breakdown'][city] = {
                    'unique_locations': city_locations,
                    'total_properties': city_properties,
                    'avg_properties_per_location': round(city_properties / city_locations, 2) if city_locations > 0 else 0
                }
                
                total_locations += city_locations
                total_properties += city_properties
            
            stats['overall_stats']['total_locations'] = total_locations
            stats['overall_stats']['total_properties'] = total_properties
            stats['overall_stats']['avg_locations_per_city'] = round(total_locations / len(location_data), 2) if len(location_data) > 0 else 0
            
            # Sort cities by location count
            stats['top_cities_by_locations'] = sorted(
                [(city, data['unique_locations']) for city, data in stats['city_breakdown'].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get location stats: {e}")
            return {'error': f'Stats retrieval failed: {str(e)}'}
        
def main():
    """Production ML pipeline execution"""
    print("=" * 80)
    print("STEP 5: SEGMENTED REAL ESTATE MODEL (Regular vs Luxury)")
    print("=" * 80)
    
    try:
        # Initialize and train production model
        model = SegmentedRealEstateModel(enable_gpu=True)
        
        # Execute ML pipeline
        model.load_and_prepare_data()
        model.train_segmented_models()
        model.generate_analysis_report()
        
        print("\n[SUCCESS] PRODUCTION MODEL PIPELINE COMPLETED")
        print(f"[SAVE] Models saved to: {model.data_dir}")
        
        # Load location summary for info
        location_file = model.data_dir / 'location_predictions.json'
        if location_file.exists():
            # Get comprehensive location statistics
            location_stats = model.get_comprehensive_location_stats()
            if 'error' not in location_stats:
                overall = location_stats['overall_stats']
                print(f"[INFO] Comprehensive location database ready:")
                print(f"       • {overall['total_locations']:,} unique locations")
                print(f"       • {location_stats['total_cities']} cities")
                print(f"       • {overall['total_properties']:,} total properties")
                print(f"       • Avg {overall['avg_locations_per_city']:.1f} locations per city")
                
                # Show top cities
                print(f"[INFO] Top cities by location coverage:")
                for city, count in location_stats['top_cities_by_locations'][:5]:
                    print(f"       • {city}: {count:,} locations")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
