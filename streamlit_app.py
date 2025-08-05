"""
ENHANCED STREAMLIT REAL ESTATE PRICE PREDICTION APP WITH GEMINI AI
=================================================================

Production-ready Streamlit application with:
1. Gemini AI conversational interface for natural language queries
2. Interactive price prediction with location intelligence
3. City-wise market analysis and insights
4. Visual analytics and interactive maps
5. Model performance insights and validation
6. Enhanced UI/UX with proper chat management
7. Robust error handling and fallbacks

Author: Real Estate ML Pipeline
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import json
from pathlib import Path
import requests
from datetime import datetime
import sys
import os
from dotenv import load_dotenv
import google.genai as genai
import time
import traceback

# Load environment variables
load_dotenv()

# Setup path for imports
sys.path.append(str(Path(__file__).parent))
try:
    from src.ml_models.step5_segmented_model import SegmentedRealEstateModel
except ImportError as e:
    st.error(f"Failed to import ML model: {e}")
    SegmentedRealEstateModel = None

# Page configuration
st.set_page_config(
    page_title="🏡 AI Real Estate Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern UI
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    
    /* Chat interface styling */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        max-width: 85%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        color: #333;
        margin-left: 0;
        margin-right: auto;
        border-left: 4px solid #667eea;
    }
    
    /* Cards and metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .price-prediction {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .quick-action-btn {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        background-color: white;
        color: #667eea;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .quick-action-btn:hover {
        background-color: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .chat-message {
            max-width: 95%;
        }
        .quick-actions {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Gemini AI Configuration
class GeminiAI:
    """Enhanced Gemini AI integration for real estate queries using new API format"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        self.client = None
        self.model_name = None
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini AI with new Client API format"""
        if not self.api_key or self.api_key == 'your_gemini_api_key_here':
            st.session_state.gemini_status = "❌ Gemini API key not configured"
            return
        
        try:
            # Initialize the new GenAI client
            self.client = genai.Client(api_key=self.api_key)
            
            # Try different Gemini model names in order of preference
            model_names = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
            
            for model_name in model_names:
                try:
                    # Test the model with a simple query
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents="Hello"
                    )
                    
                    if response and response.text:
                        self.available = True
                        self.model_name = model_name
                        st.session_state.gemini_status = f"✅ Gemini AI connected successfully ({model_name})"
                        st.session_state.gemini_available = True
                        break
                        
                except Exception as model_error:
                    print(f"Failed to initialize {model_name}: {model_error}")
                    continue
            
            if not self.available:
                st.session_state.gemini_status = "❌ No compatible Gemini model found"
                
        except Exception as e:
            st.session_state.gemini_status = f"❌ Gemini AI setup failed: {str(e)}"
            self.available = False
    
    def generate_chat_response(self, user_query: str, prediction_data: dict = None, market_data: dict = None) -> str:
        """Generate comprehensive AI response for chat queries using new API"""
        if not self.available or not self.client:
            return self._get_fallback_response(user_query, prediction_data)
        
        try:
            # Create rich context for the AI
            context = self._build_context(user_query, prediction_data, market_data)
            
            # Use the new API format
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context
            )
            
            return response.text if response and response.text else self._get_fallback_response(user_query, prediction_data)
            
        except Exception as e:
            st.error(f"Gemini AI error: {str(e)}")
            return self._get_fallback_response(user_query, prediction_data)
    
    def generate_response(self, user_query: str) -> str:
        """Simple response generation method for compatibility"""
        return self.generate_chat_response(user_query)
    
    def _build_context(self, user_query: str, prediction_data: dict = None, market_data: dict = None) -> str:
        """Build comprehensive context for AI responses"""
        context = f"""
You are an expert real estate advisor for Indian metropolitan cities with deep market knowledge.

User Query: {user_query}

Available Cities: Mumbai, Bangalore, Delhi, Chennai, Hyderabad, Kolkata

Core Capabilities:
1. Property price predictions using ML models
2. Location-based market analysis  
3. Investment recommendations
4. Market trend insights
5. Natural language property queries

Guidelines:
- Be conversational, helpful, and professional
- Use Indian currency (₹ Lakhs/Crores) 
- Consider location, amenities, and market dynamics
- Provide actionable insights
- If asking for predictions, guide users to provide: city, location, property type, bedrooms, area
"""

        if prediction_data:
            context += f"\n\nRecent Prediction Results:\n{json.dumps(prediction_data, indent=2)}"
        
        if market_data:
            context += f"\n\nMarket Data:\n{json.dumps(market_data, indent=2)}"
        
        return context
    
    def _get_fallback_response(self, query: str, prediction_data: dict = None) -> str:
        """Provide fallback responses when Gemini is unavailable"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'predict', 'cost', 'bhk', 'apartment']):
            if prediction_data:
                pred_price = prediction_data.get('predicted_price_formatted', 'N/A')
                model_used = prediction_data.get('model_used', 'Unknown')
                return f"""
**Price Prediction Complete!**

🏠 **Predicted Price**: {pred_price}
🤖 **Model Used**: {model_used.title()}
📊 **Confidence**: {prediction_data.get('confidence', 75):.0f}%

*Note: Gemini AI explanation unavailable. Please configure GEMINI_API_KEY for enhanced insights.*
"""
            else:
                return """
I can help you with property price predictions! Please provide:
- City (Mumbai, Delhi, Bangalore, etc.)
- Location/Area
- Property type (Apartment, Villa, etc.)
- Number of bedrooms
- Area in sq ft

Example: "Predict price for 3BHK apartment in Andheri West Mumbai, 1200 sqft"
"""
        
        elif any(word in query_lower for word in ['market', 'trend', 'analysis']):
            return """
**Market Insights** (Metro Cities):

📈 **Current Trends**:
- Mumbai: Premium locations showing steady growth
- Bangalore: Tech corridors driving demand
- Delhi NCR: Infrastructure development boosting value

💡 **Investment Tips**:
- Consider location connectivity
- Check upcoming infrastructure projects
- Evaluate amenities and facilities

*Configure Gemini AI for detailed market analysis and personalized recommendations.*
"""
        
        else:
            return """
Hello! I'm your AI Real Estate Assistant. I can help you with:

🏠 **Property Predictions**: Get accurate price estimates
📊 **Market Analysis**: Understand trends and insights  
💰 **Investment Advice**: Find the best opportunities
🗺️ **Location Intelligence**: Explore neighborhoods

Try asking:
- "What's the price of a 2BHK in Koramangala Bangalore?"
- "Show me market trends for Mumbai"
- "Best investment locations in Delhi"

*Enhanced AI responses available with Gemini API configuration.*
"""

# Initialize Gemini AI
@st.cache_resource
def init_gemini():
    return GeminiAI()

class EnhancedRealEstateApp:
    """Enhanced Streamlit Real Estate Application with improved ML integration"""
    
    def __init__(self):
        self.ml_model = None
        self.models_loaded = False
        self.data_loaded = False
        self.gemini_ai = None
        self.initialize_app()
        
    def initialize_app(self):
        """Initialize the application with proper error handling"""
        try:
            self.setup_ml_model()
            self.setup_session_state()
            self.gemini_ai = init_gemini()
            self.ai_assistant = self.gemini_ai  # Alias for compatibility
        except Exception as e:
            st.error(f"App initialization failed: {str(e)}")
            st.session_state.app_initialized = False
            
    def setup_ml_model(self):
        """Setup the ML model with enhanced integration"""
        try:
            if SegmentedRealEstateModel is None:
                st.error("❌ ML Model class not available. Please check imports.")
                return
                
            # Initialize the model
            self.ml_model = SegmentedRealEstateModel()
            
            # Check if models exist
            model_dir = Path("data/segmented_models")
            if not model_dir.exists():
                st.error("❌ Model directory not found. Please run training first.")
                return
                
            # Load models if available
            required_files = [
                "regular_properties_model.joblib",
                "luxury_properties_model.joblib",
                "model_metadata.json"
            ]
            
            missing_files = [f for f in required_files if not (model_dir / f).exists()]
            if missing_files:
                st.warning(f"⚠️ Missing model files: {missing_files}")
                return
                
            # Models are loaded automatically when needed by the ML class
            self.models_loaded = True
            st.session_state.models_loaded = True
            
            # Load additional data
            self.load_market_data()
            
        except Exception as e:
            st.error(f"ML Model setup failed: {str(e)}")
            st.session_state.models_loaded = False
            
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hello! I'm your AI Real Estate Assistant. I can help you with property price predictions, market analysis, and investment advice for Indian metro cities. What would you like to know?",
                    "timestamp": datetime.now()
                }
            ]
            
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
            
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            
    def load_market_data(self):
        """Load market data for analysis"""
        try:
            data_dir = Path("data/segmented_models")
            
            # Load predictions and location data if available
            if (data_dir / "location_predictions.json").exists():
                with open(data_dir / "location_predictions.json", 'r') as f:
                    self.location_data = json.load(f)
                    
            if (data_dir / "city_predictions.json").exists():
                with open(data_dir / "city_predictions.json", 'r') as f:
                    self.city_data = json.load(f)
                    
            self.data_loaded = True
            st.session_state.data_loaded = True
            
        except Exception as e:
            st.warning(f"Market data loading failed: {str(e)}")
            
    def predict_property_price(self, property_features: dict) -> dict:
        """Enhanced property price prediction with better error handling"""
        if not self.models_loaded or not self.ml_model:
            return {
                'error': 'Models not loaded. Please run training first.',
                'suggestion': 'Run: python src/ml_models/step5_segmented_model.py'
            }
            
        try:
            # Use the ML model's prediction method
            result = self.ml_model.predict_location_price(
                city=property_features.get('city', 'Mumbai'),
                location=property_features.get('location', 'Central'),
                property_features=property_features
            )
            
            if 'error' not in result:
                # Enhance the result with additional information
                enhanced_result = {
                    'success': True,
                    'predicted_price': result['ml_prediction']['predicted_price'],
                    'predicted_price_formatted': result['ml_prediction']['predicted_price_formatted'],
                    'predicted_price_crores': f"₹{result['ml_prediction']['predicted_price']/10000000:.2f} Cr",
                    'model_used': result['ml_prediction']['model_used'],
                    'segment': result['ml_prediction']['segment'],
                    'confidence': result['ml_prediction'].get('confidence', 75),
                    'location_info': result.get('location_info', {}),
                    'market_context': result.get('market_context', 'No additional context available'),
                    'recommendation': result.get('recommendation', 'Property analysis completed'),
                    'timestamp': datetime.now()
                }
                
                # Add to prediction history
                st.session_state.prediction_history.append(enhanced_result)
                
                return enhanced_result
            else:
                return result
                
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'suggestion': 'Please check your input parameters and try again.'
            }
    
    def process_natural_language_query(self, query: str) -> dict:
        """Process natural language queries using the ML model"""
        if not self.models_loaded or not self.ml_model:
            return {
                'error': 'ML model not available',
                'response': 'I need the ML models to be loaded first. Please run the training process.'
            }
            
        try:
            # Use the ML model's natural language processing
            result = self.ml_model.predict_from_natural_query(query)
            
            if result and 'error' not in result:
                # Convert to our format
                return {
                    'success': True,
                    'prediction_data': {
                        'predicted_price': result.get('prediction', {}).get('ml_prediction', {}).get('predicted_price', 0),
                        'predicted_price_formatted': result.get('prediction', {}).get('ml_prediction', {}).get('predicted_price_formatted', 'N/A'),
                        'model_used': result.get('prediction', {}).get('ml_prediction', {}).get('model_used', 'unknown'),
                        'confidence': result.get('parsing_confidence', 0.75) * 100,
                        'property_details': result.get('parsed_features', {}),
                        'market_context': result.get('market_context', '')
                    },
                    'raw_result': result
                }
            else:
                return {
                    'error': result.get('error', 'Query processing failed'),
                    'suggestions': result.get('suggestions', [])
                }
                
        except Exception as e:
            return {
                'error': f'Natural language processing failed: {str(e)}',
                'response': 'I had trouble understanding your query. Please try rephrasing or use the structured form.'
            }
    
    def get_market_insights(self, city: str = None) -> dict:
        """Get market insights for a city or overall market"""
        if not self.data_loaded:
            return {'error': 'Market data not available'}
            
        try:
            if city and hasattr(self, 'city_data') and city in self.city_data:
                city_stats = self.city_data[city]
                return {
                    'city': city,
                    'regular_avg': city_stats.get('regular_avg', 0),
                    'luxury_avg': city_stats.get('luxury_avg', 0),
                    'total_properties': city_stats.get('regular_count', 0) + city_stats.get('luxury_count', 0),
                    'luxury_percentage': (city_stats.get('luxury_count', 0) / max(1, city_stats.get('regular_count', 0) + city_stats.get('luxury_count', 0))) * 100
                }
            else:
                # Return overall market data
                return {
                    'available_cities': list(self.city_data.keys()) if hasattr(self, 'city_data') else [],
                    'note': 'Select a specific city for detailed insights'
                }
                
        except Exception as e:
            return {'error': f'Market insights failed: {str(e)}'}

def main():
    """Enhanced main Streamlit application"""
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = EnhancedRealEstateApp()
    
    app = st.session_state.app
    
    # Header with status indicators
    st.markdown('<h1 class="main-header">🏡 AI Real Estate Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Property Price Prediction with AI-Powered Market Insights</div>', unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✅ Ready" if app.models_loaded else "❌ Not Ready"
        st.markdown(f"**ML Models**: {status}")
    with col2:
        status = "✅ Loaded" if app.data_loaded else "❌ Missing"
        st.markdown(f"**Market Data**: {status}")
    with col3:
        gemini_status = st.session_state.get('gemini_status', '❓ Unknown')
        st.markdown(f"**Gemini AI**: {gemini_status}")
    with col4:
        pred_count = len(st.session_state.get('prediction_history', []))
        st.markdown(f"**Predictions**: {pred_count}")
    
    # Check critical requirements
    if not app.models_loaded:
        st.error("⚠️ **ML Models not found!** Please run the training process first.")
        with st.expander("📋 Setup Instructions"):
            st.code("""
# Run the training script
python src/ml_models/step5_segmented_model.py

# Or run from the location_prediction_example
python location_prediction_example.py
            """)
        st.info("💡 The models need to be trained before using the prediction features.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation with enhanced options
    page = st.sidebar.selectbox(
        "Choose your experience:",
        [
            "💬 AI Chat Assistant", 
            "🏠 Price Prediction", 
            "📊 Market Analysis", 
            "🗺️ City Explorer", 
            "📈 Model Performance",
            "⚙️ Settings"
        ]
    )
    
    # Quick stats in sidebar
    st.sidebar.markdown("### 📊 Quick Stats")
    if app.data_loaded and hasattr(app, 'city_data'):
        st.sidebar.metric("Cities Covered", len(app.city_data))
        total_props = sum(city.get('regular_count', 0) + city.get('luxury_count', 0) 
                         for city in app.city_data.values())
        st.sidebar.metric("Total Properties", f"{total_props:,}")
    
    st.sidebar.markdown("---")
    
    # Import enhanced pages
    try:
        from streamlit_app_enhanced_pages import (
            enhanced_chat_assistant_page,
            enhanced_prediction_page, 
            enhanced_market_analysis_page,
            enhanced_city_explorer_page,
            enhanced_model_performance_page,
            settings_page
        )
        
        # Main content routing
        if page == "💬 AI Chat Assistant":
            enhanced_chat_assistant_page(app)
        elif page == "🏠 Price Prediction":
            enhanced_prediction_page(app)
        elif page == "📊 Market Analysis":
            enhanced_market_analysis_page(app)
        elif page == "🗺️ City Explorer":
            enhanced_city_explorer_page(app)
        elif page == "📈 Model Performance":
            enhanced_model_performance_page(app)
        elif page == "⚙️ Settings":
            settings_page(app)
            
    except ImportError as e:
        st.error(f"Enhanced pages not available: {str(e)}")
        st.info("Using basic interface...")
        
        # Basic fallback interface
        if page == "💬 AI Chat Assistant":
            basic_chat_interface(app)
        else:
            st.info(f"Page '{page}' is not available in basic mode.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">🏡 AI Real Estate Advisor | Powered by Gemini AI & Advanced ML Models</div>', 
        unsafe_allow_html=True
    )

def basic_chat_interface(app):
    """Basic chat interface when enhanced pages are not available"""
    st.header("💬 Basic Chat Interface")
    
    # Simple chat input
    if prompt := st.chat_input("Ask me about real estate..."):
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Generate response
        if app.gemini_ai and app.gemini_ai.available:
            response = app.gemini_ai.generate_chat_response(prompt)
        else:
            response = "I'm here to help with real estate queries! Please configure Gemini AI for enhanced responses."
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M")
        })
    
    # Display messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()
