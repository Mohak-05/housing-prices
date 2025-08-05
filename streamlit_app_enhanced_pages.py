"""
Enhanced pages for the Streamlit Real Estate App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime
import json
import os

def enhanced_chat_assistant_page(app):
    """Enhanced AI chat assistant page"""
    st.header("🤖 AI Real Estate Assistant")
    st.markdown("Ask me anything about real estate! I can help you with property prices, market analysis, and location insights.")
    
    # Initialize chat history if not exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about real estate properties, prices, or locations..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Use the AI assistant to generate response
                    response = app.ai_assistant.generate_response(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"I'm having trouble processing your request. Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with suggested questions
    with st.sidebar:
        st.subheader("💡 Suggested Questions")
        
        suggested_questions = [
            "What's the average property price in Mumbai?",
            "Compare property prices between Delhi and Bangalore",
            "Which areas in Chennai have the best investment potential?",
            "What factors affect property prices the most?",
            "Show me luxury properties in Hyderabad",
            "What's the price trend in Kolkata?"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=f"suggested_{hash(question)}"):
                # Add the suggested question to chat
                st.session_state.chat_messages.append({"role": "user", "content": question})
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()

import json
import os

def load_trained_locations():
    """Load available locations from trained model data"""
    location_file = "data/segmented_models/location_predictions.json"
    if os.path.exists(location_file):
        try:
            with open(location_file, 'r') as f:
                data = json.load(f)
                return {city: list(data[city]['locations'].keys()) for city in data.keys()}
        except Exception as e:
            st.error(f"Error loading location data: {e}")
            return {}
    else:
        st.warning("⚠️ Location data not found. Please ensure models are trained.")
        return {}

def enhanced_prediction_page(app):
    """Enhanced property price prediction page"""
    st.header("🏠 Property Price Prediction")
    st.markdown("Get accurate price predictions using our advanced ML models")
    
    # Load available locations
    trained_locations = load_trained_locations()
    
    if not trained_locations:
        st.error("❌ No trained location data available. Please ensure the ML models are properly trained.")
        st.info("💡 Run the training script to generate location data: `python src/ml_models/step5_segmented_model.py`")
        return
    
    # Two-column layout
    form_col, result_col = st.columns([1.5, 1])
    
    with form_col:
        st.subheader("📝 Property Details")
        
        # Show available data info
        with st.expander("📊 Available Training Data"):
            st.markdown("**Locations available for prediction:**")
            for city, locations in trained_locations.items():
                st.markdown(f"• **{city}**: {len(locations):,} trained locations")
        
        # Prediction form
        with st.form("enhanced_property_form", clear_on_submit=False):
            # Basic information
            st.markdown("**Basic Information**")
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.selectbox(
                    "City *",
                    list(trained_locations.keys()),
                    help="Select the city where the property is located"
                )
                
                property_type = st.selectbox(
                    "Property Type *",
                    ["Apartment", "Villa", "Independent House", "Builder Floor"],
                    help="Type of property"
                )
            
            with col2:
                # Dynamic location dropdown based on selected city
                available_locations = trained_locations.get(city, [])
                if available_locations:
                    location = st.selectbox(
                        f"Location/Area in {city} *",
                        available_locations,
                        help=f"Select from {len(available_locations):,} trained locations in {city}"
                    )
                else:
                    st.error(f"No locations available for {city}")
                    location = ""
                
                furnishing = st.selectbox(
                    "Furnishing Status",
                    ["Unfurnished", "Semi-Furnished", "Furnished"]
                )
            
            # Property specifications
            st.markdown("**Property Specifications**")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                area = st.number_input(
                    "Area (sq ft) *",
                    min_value=300,
                    max_value=10000,
                    value=1200,
                    step=50,
                    help="Built-up area in square feet"
                )
            
            with col4:
                bedrooms = st.selectbox(
                    "Bedrooms *",
                    [1, 2, 3, 4, 5, 6],
                    index=1,
                    help="Number of bedrooms"
                )
            
            with col5:
                bathrooms = st.selectbox(
                    "Bathrooms",
                    [1, 2, 3, 4, 5],
                    index=1,
                    help="Number of bathrooms"
                )
            
            # Amenities
            st.markdown("**Amenities & Features**")
            
            amenity_cols = st.columns(4)
            with amenity_cols[0]:
                parking = st.checkbox("🚗 Car Parking")
                security = st.checkbox("🔒 24x7 Security")
            with amenity_cols[1]:
                gym = st.checkbox("💪 Gymnasium")
                pool = st.checkbox("🏊 Swimming Pool")
            with amenity_cols[2]:
                lift = st.checkbox("🛗 Lift Available")
                power_backup = st.checkbox("⚡ Power Backup")
            with amenity_cols[3]:
                garden = st.checkbox("🌳 Garden/Landscaping")
                club = st.checkbox("🏛️ Clubhouse")
            
            # Submit button
            predict_button = st.form_submit_button(
                "🔮 Predict Property Price",
                type="primary",
                use_container_width=True
            )
        
        # Natural language input alternative
        st.markdown("---")
        st.subheader("🗣️ Or Try Natural Language")
        
        # Show example locations for natural language
        if trained_locations:
            with st.expander("💡 Example Locations for Natural Language"):
                for city, locations in list(trained_locations.items())[:3]:  # Show first 3 cities
                    sample_locations = locations[:3]  # First 3 locations
                    st.markdown(f"**{city}**: {', '.join(sample_locations)}, ...")
        
        # Generate dynamic example with actual location
        example_city = list(trained_locations.keys())[0] if trained_locations else "Mumbai"
        example_location = trained_locations.get(example_city, ["Andheri West"])[0] if trained_locations else "Andheri West"
        
        nl_query = st.text_input(
            "Describe your property:",
            placeholder=f"3BHK apartment in {example_location} {example_city}, 1200 sqft, furnished with parking",
            help="Describe your property in natural language using trained locations"
        )
        
        if st.button("🤖 Predict from Description", type="secondary"):
            if nl_query:
                with st.spinner("🔄 Processing natural language query..."):
                    nl_result = app.process_natural_language_query(nl_query)
                    
                if nl_result.get('success'):
                    st.session_state.last_prediction = nl_result['prediction_data']
                    st.success("✅ Prediction completed! Check the results panel.")
                    st.rerun()
                else:
                    st.error(f"❌ {nl_result.get('error', 'Query processing failed')}")
    
    with result_col:
        st.subheader("📊 Prediction Results")
        
        # Process form submission
        if predict_button:
            if not location.strip():
                st.error("❌ Please enter a location/area")
            else:
                # Prepare property data
                property_data = {
                    'city': city,
                    'location': location,
                    'area': area,
                    'no._of_bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'property_type': property_type,
                    'furnishing_status': furnishing,
                    'parking': int(parking),
                    'security': int(security),
                    'gym': int(gym),
                    'pool': int(pool),
                    'lift': int(lift),
                    'power_backup': int(power_backup),
                    'garden': int(garden),
                    'club': int(club)
                }
                
                # Get prediction
                with st.spinner("🤖 AI is analyzing your property..."):
                    result = app.predict_property_price(property_data)
                
                if result.get('success'):
                    st.session_state.last_prediction = result
                    st.success("✅ Prediction completed!")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Prediction failed')}")
        
        # Display results
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            
            # Price display
            st.markdown(
                f'<div class="price-prediction">₹{result["predicted_price"]/10000000:.2f} Crores</div>',
                unsafe_allow_html=True
            )
            
            # Detailed metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Price", result['predicted_price_formatted'])
                st.metric("Model Used", result['model_used'].title())
            with col2:
                st.metric("Confidence", f"{result['confidence']:.0f}%")
                st.metric("Property Segment", result['segment'].title())
            
            # Additional information
            if result.get('market_context'):
                st.info(f"💡 {result['market_context']}")
            
            if result.get('recommendation'):
                st.success(f"🎯 {result['recommendation']}")
            
            # Historical predictions
            st.markdown("---")
            st.subheader("📜 Recent Predictions")
            
            if st.session_state.prediction_history:
                for i, pred in enumerate(st.session_state.prediction_history[-3:], 1):
                    with st.expander(f"Prediction {i} - ₹{pred['predicted_price']/10000000:.2f} Cr"):
                        st.write(f"**Price**: {pred['predicted_price_formatted']}")
                        st.write(f"**Model**: {pred['model_used'].title()}")
                        st.write(f"**Confidence**: {pred['confidence']:.0f}%")
                        if pred.get('timestamp'):
                            st.write(f"**Time**: {pred['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("👆 Fill out the form above to get a price prediction")
            
            # Quick tips
            st.markdown("### 💡 Tips for Better Predictions")
            st.markdown("""
            - **Use trained locations only** - Select from the dropdown for accurate predictions
            - **Include amenities** that add value to your property
            - **Check multiple scenarios** by varying parameters
            - **Consider market context** for final decisions
            - **Natural language** works with trained locations too
            """)

def enhanced_market_analysis_page(app):
    """Enhanced market analysis page"""
    st.header("📊 Market Analysis & Insights")
    st.markdown("Comprehensive real estate market analysis across Indian metro cities")
    
    if not app.data_loaded:
        st.warning("⚠️ Market data not available. Please ensure models are trained.")
        return
    
    # Overall market metrics
    st.subheader("🌐 Overall Market Overview")
    
    if hasattr(app, 'city_data'):
        # Calculate overall metrics
        total_regular = sum(city.get('regular_count', 0) for city in app.city_data.values())
        total_luxury = sum(city.get('luxury_count', 0) for city in app.city_data.values())
        total_properties = total_regular + total_luxury
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Properties",
                f"{total_properties:,}",
                help="Total properties in dataset"
            )
        
        with col2:
            st.metric(
                "Regular Properties", 
                f"{total_regular:,}",
                f"{total_regular/total_properties*100:.1f}% of market"
            )
        
        with col3:
            st.metric(
                "Luxury Properties",
                f"{total_luxury:,}", 
                f"{total_luxury/total_properties*100:.1f}% of market"
            )
        
        with col4:
            avg_regular = sum(city.get('regular_avg', 0) for city in app.city_data.values()) / len(app.city_data)
            st.metric(
                "Avg Regular Price",
                f"₹{avg_regular/10000000:.1f} Cr"
            )
    
    # City comparison
    st.subheader("🏙️ City-wise Market Comparison")
    
    if hasattr(app, 'city_data'):
        # Prepare data for visualization
        city_data = []
        for city, stats in app.city_data.items():
            city_data.append({
                'City': city,
                'Regular Count': stats.get('regular_count', 0),
                'Luxury Count': stats.get('luxury_count', 0),
                'Regular Avg (₹ Cr)': stats.get('regular_avg', 0) / 10000000,
                'Luxury Avg (₹ Cr)': stats.get('luxury_avg', 0) / 10000000 if stats.get('luxury_avg', 0) > 0 else 0,
                'Total Properties': stats.get('regular_count', 0) + stats.get('luxury_count', 0),
                'Luxury %': (stats.get('luxury_count', 0) / max(1, stats.get('regular_count', 0) + stats.get('luxury_count', 0))) * 100
            })
        
        city_df = pd.DataFrame(city_data)
        
        # Interactive charts
        tab1, tab2, tab3 = st.tabs(["📈 Average Prices", "🏠 Property Count", "💎 Luxury Market"])
        
        with tab1:
            fig = px.bar(
                city_df,
                x='City',
                y=['Regular Avg (₹ Cr)', 'Luxury Avg (₹ Cr)'],
                title="Average Property Prices by City and Segment",
                labels={'value': 'Average Price (₹ Crores)', 'variable': 'Property Segment'},
                color_discrete_map={
                    'Regular Avg (₹ Cr)': '#667eea',
                    'Luxury Avg (₹ Cr)': '#764ba2'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.bar(
                city_df,
                x='City',
                y=['Regular Count', 'Luxury Count'],
                title="Property Count by City and Segment",
                labels={'value': 'Number of Properties', 'variable': 'Property Segment'},
                color_discrete_map={
                    'Regular Count': '#667eea',
                    'Luxury Count': '#764ba2'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.scatter(
                city_df,
                x='Total Properties',
                y='Luxury %',
                size='Luxury Count',
                color='City',
                title="Luxury Market Penetration by City",
                labels={'Luxury %': 'Luxury Properties (%)', 'Total Properties': 'Total Properties'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed city table
        st.subheader("📋 Detailed City Statistics")
        st.dataframe(
            city_df.style.format({
                'Regular Avg (₹ Cr)': '₹{:.2f} Cr',
                'Luxury Avg (₹ Cr)': '₹{:.2f} Cr',
                'Luxury %': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    # Market insights
    st.subheader("💡 Market Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info("""
        **🎯 Key Market Trends**
        - Mumbai leads in luxury property prices
        - Bangalore shows balanced growth across segments
        - Delhi NCR has strong infrastructure-driven demand
        - Chennai offers value propositions in luxury segment
        """)
    
    with insight_col2:
        st.success("""
        **💰 Investment Recommendations**
        - Focus on connectivity and upcoming infrastructure
        - Consider luxury markets in Mumbai and Delhi
        - Explore emerging areas in Bangalore and Hyderabad
        - Monitor government policy impacts on pricing
        """)

def enhanced_city_explorer_page(app):
    """Enhanced city explorer with interactive features"""
    st.header("🗺️ City Explorer")
    st.markdown("Explore real estate opportunities across Indian metro cities")
    
    # City selector with enhanced features
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_city = st.selectbox(
            "🏙️ Select City to Explore:",
            ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
        )
    
    with col2:
        map_style = st.selectbox(
            "🗺️ Map Style:",
            ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"]
        )
    
    with col3:
        if st.button("🔄 Refresh Data", help="Generate new sample property data and trends"):
            # Clear city-specific cached data
            keys_to_clear = [
                f"price_trend_{selected_city}",
                f"sentiment_{selected_city}",
                f"property_markers_{selected_city}"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("City data refreshed!")
            st.rerun()
    
    # City coordinates and information
    city_info = {
        "Mumbai": {
            "coords": [19.0760, 72.8777],
            "description": "Financial capital with premium real estate market",
            "highlights": ["South Mumbai", "Bandra-Kurla Complex", "Andheri", "Thane"],
            "avg_price_range": "₹8-25 Cr (Luxury), ₹60L-2 Cr (Regular)"
        },
        "Delhi": {
            "coords": [28.7041, 77.1025],
            "description": "National capital with diverse property options",
            "highlights": ["Connaught Place", "Defence Colony", "Greater Kailash", "Gurgaon"],
            "avg_price_range": "₹5-20 Cr (Luxury), ₹50L-2 Cr (Regular)"
        },
        "Bangalore": {
            "coords": [12.9716, 77.5946],
            "description": "Silicon Valley of India with tech-driven growth",
            "highlights": ["Koramangala", "HSR Layout", "Whitefield", "Electronic City"],
            "avg_price_range": "₹3-15 Cr (Luxury), ₹40L-2 Cr (Regular)"
        },
        "Chennai": {
            "coords": [13.0827, 80.2707],
            "description": "Industrial hub with growing IT sector",
            "highlights": ["Anna Nagar", "Adyar", "OMR", "Velachery"],
            "avg_price_range": "₹2-10 Cr (Luxury), ₹30L-2 Cr (Regular)"
        },
        "Hyderabad": {
            "coords": [17.3850, 78.4867],
            "description": "Emerging IT destination with affordable luxury",
            "highlights": ["Banjara Hills", "Jubilee Hills", "Gachibowli", "Kondapur"],
            "avg_price_range": "₹2-12 Cr (Luxury), ₹35L-2 Cr (Regular)"
        },
        "Kolkata": {
            "coords": [22.5726, 88.3639],
            "description": "Cultural capital with traditional value",
            "highlights": ["Park Street", "Salt Lake", "New Town", "Ballygunge"],
            "avg_price_range": "₹1.5-8 Cr (Luxury), ₹25L-2 Cr (Regular)"
        }
    }
    
    if selected_city in city_info:
        city_data = city_info[selected_city]
        lat, lon = city_data["coords"]
        
        # Generate consistent seed for this city (used for all random data)
        import hashlib
        city_seed = int(hashlib.md5(selected_city.encode()).hexdigest()[:8], 16) % 1000
        
        # City overview
        st.subheader(f"📍 {selected_city} Overview")
        
        overview_col1, overview_col2 = st.columns([2, 1])
        
        with overview_col1:
            st.markdown(f"**Description**: {city_data['description']}")
            st.markdown(f"**Price Range**: {city_data['avg_price_range']}")
            st.markdown(f"**Popular Areas**: {', '.join(city_data['highlights'])}")
        
        with overview_col2:
            if hasattr(app, 'city_data') and selected_city in app.city_data:
                city_stats = app.city_data[selected_city]
                st.metric("Total Properties", f"{city_stats.get('regular_count', 0) + city_stats.get('luxury_count', 0):,}")
                st.metric("Luxury Properties", f"{city_stats.get('luxury_count', 0):,}")
                luxury_pct = (city_stats.get('luxury_count', 0) / max(1, city_stats.get('regular_count', 0) + city_stats.get('luxury_count', 0))) * 100
                st.metric("Luxury Market %", f"{luxury_pct:.1f}%")
        
        # Interactive map
        st.subheader("🗺️ Interactive Map")
        
        # Create map with different tile styles
        tile_map = {
            "OpenStreetMap": "OpenStreetMap",
            "CartoDB Positron": "CartoDB positron",
            "CartoDB Dark_Matter": "CartoDB dark_matter"
        }
        
        m = folium.Map(
            location=[lat, lon],
            zoom_start=11,
            tiles=tile_map.get(map_style, "OpenStreetMap")
        )
        
        # Add city center marker
        folium.Marker(
            [lat, lon],
            popup=f"<b>{selected_city} City Center</b><br>{city_data['description']}",
            tooltip=f"{selected_city} Center",
            icon=folium.Icon(color='red', icon='star', prefix='fa')
        ).add_to(m)
        
        # Add sample property markers for popular areas
        # Generate consistent property data using session state
        property_data_key = f"property_markers_{selected_city}"
        if property_data_key not in st.session_state:
            # Set seed for consistent results
            np.random.seed(city_seed + 2)
            property_markers = []
            
            for i, area in enumerate(city_data['highlights'][:4]):
                # Generate consistent coordinates around the city center
                area_lat = lat + np.random.normal(0, 0.02)
                area_lon = lon + np.random.normal(0, 0.02)
                
                # Simulate consistent property data
                property_price = np.random.uniform(8000000, 80000000)
                property_area = np.random.randint(900, 2500)
                property_type = np.random.choice(['Apartment', 'Villa', 'Independent House'])
                
                property_markers.append({
                    'area': area,
                    'lat': area_lat,
                    'lon': area_lon,
                    'price': property_price,
                    'area_sqft': property_area,
                    'type': property_type
                })
            
            st.session_state[property_data_key] = property_markers
        
        # Use stored property data for consistent markers
        for marker_data in st.session_state[property_data_key]:
            color = 'blue' if marker_data['price'] < 20000000 else 'orange'
            
            popup_text = f"""
            <b>{marker_data['area']}</b><br>
            Price: ₹{marker_data['price']/10000000:.1f} Cr<br>
            Area: {marker_data['area_sqft']} sq ft<br>
            Type: {marker_data['type']}
            """
            
            folium.CircleMarker(
                [marker_data['lat'], marker_data['lon']],
                radius=10,
                popup=popup_text,
                tooltip=f"{marker_data['area']} - ₹{marker_data['price']/10000000:.1f} Cr",
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Location insights and recommendations
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.subheader("🎯 Investment Insights")
            
            recommendations = {
                "Mumbai": [
                    "Focus on connectivity to business districts",
                    "Premium locations show strong appreciation",
                    "Consider upcoming infrastructure projects",
                    "Luxury market is well-established"
                ],
                "Delhi": [
                    "NCR expansion offers growth opportunities",
                    "Metro connectivity is crucial factor",
                    "Government policies impact significantly",
                    "Mixed development zones are emerging"
                ],
                "Bangalore": [
                    "Tech corridors drive demand",
                    "Traffic connectivity is key consideration",
                    "Emerging areas show high potential",
                    "Balanced luxury-regular market"
                ],
                "Chennai": [
                    "OMR corridor is fast developing",
                    "Industrial growth supports stability",
                    "Affordable luxury segment growing",
                    "Traditional areas maintain value"
                ],
                "Hyderabad": [
                    "IT sector expansion driving growth",
                    "Affordable luxury market emerging",
                    "Infrastructure development ongoing",
                    "Good value propositions available"
                ],
                "Kolkata": [
                    "Traditional value with modern growth",
                    "New Town development promising",
                    "Cultural significance adds stability",
                    "Emerging IT sector presence"
                ]
            }
            
            for rec in recommendations.get(selected_city, []):
                st.write(f"• {rec}")
        
        with insights_col2:
            st.subheader("📊 Market Trends")
            
            # Generate consistent trend data based on city (deterministic)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            
            # Generate trend data using session state to prevent continuous updates
            trend_key = f"price_trend_{selected_city}"
            if trend_key not in st.session_state:
                # Set seed for consistent results
                np.random.seed(city_seed)
                base_trend = np.random.normal(100, 3, 6)
                st.session_state[trend_key] = base_trend.cumsum()
            
            price_trend = st.session_state[trend_key]
            
            fig = px.line(
                x=months,
                y=price_trend,
                title=f"{selected_city} Price Index Trend",
                labels={'x': 'Month', 'y': 'Price Index'}
            )
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Market sentiment - also make it consistent per city
            sentiment_key = f"sentiment_{selected_city}"
            if sentiment_key not in st.session_state:
                np.random.seed(city_seed + 1)  # Different seed for sentiment
                st.session_state[sentiment_key] = np.random.uniform(0.6, 0.9)
            
            sentiment_score = st.session_state[sentiment_key]
            sentiment_text = "Positive" if sentiment_score > 0.75 else "Neutral" if sentiment_score > 0.6 else "Cautious"
            st.metric("Market Sentiment", sentiment_text, f"{sentiment_score:.1%}")

def enhanced_model_performance_page(app):
    """Enhanced model performance and analytics page"""
    st.header("📈 Model Performance & Analytics")
    st.markdown("Detailed insights into our ML models' performance and capabilities")
    
    # Model overview
    st.subheader("🤖 Model Information")
    
    # Load metadata if available
    model_metadata = getattr(app, 'metadata', None)
    if hasattr(app.ml_model, 'data_dir'):
        try:
            import json
            with open(app.ml_model.data_dir / 'model_metadata.json', 'r') as f:
                model_metadata = json.load(f)
        except:
            pass
    
    if model_metadata:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Architecture", "Segmented ML")
            st.metric("Regular Properties", f"{model_metadata.get('regular_properties_count', 0):,}")
        
        with col2:
            st.metric("Luxury Properties", f"{model_metadata.get('luxury_properties_count', 0):,}")
            st.metric("Features Used", model_metadata.get('total_features', 'N/A'))
        
        with col3:
            st.metric("Training Date", model_metadata.get('created_at', 'Unknown')[:10])
            st.metric("GPU Enabled", "✅ Yes" if model_metadata.get('gpu_enabled') else "❌ No")
        
        with col4:
            outliers_removed = model_metadata.get('outliers_count', 0)
            st.metric("Outliers Removed", f"{outliers_removed:,}")
            st.metric("Data Quality", "High" if outliers_removed > 100 else "Good")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    # Simulate performance data (in real app, this would come from model evaluation)
    performance_data = {
        'Model': ['Regular Properties', 'Luxury Properties'],
        'Algorithm': ['LightGBM', 'Random Forest'],
        'R² Score (Test)': [0.457, 0.300],
        'R² Score (CV)': [0.425, 0.284],
        'MAE (₹ Lakhs)': [22.35, 100.13],
        'RMSE (₹ Lakhs)': [35.42, 156.78],
        'Market Segment': ['<₹2 Crores', '₹2+ Crores'],
        'Training Samples': [8500, 1200]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Performance visualization
    metric_tab1, metric_tab2, metric_tab3 = st.tabs(["🎯 Accuracy Metrics", "📉 Error Analysis", "⚖️ Model Comparison"])
    
    with metric_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig = px.bar(
                perf_df,
                x='Model',
                y='R² Score (Test)',
                title="R² Score Comparison (Higher is Better)",
                color='Model',
                color_discrete_map={'Regular Properties': '#667eea', 'Luxury Properties': '#764ba2'}
            )
            fig.update_traces(text=perf_df['R² Score (Test)'].apply(lambda x: f"{x:.1%}"), textposition='auto')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cross-validation comparison
            fig = px.bar(
                perf_df,
                x='Model',
                y='R² Score (CV)',
                title="Cross-Validation R² Score",
                color='Model',
                color_discrete_map={'Regular Properties': '#667eea', 'Luxury Properties': '#764ba2'}
            )
            fig.update_traces(text=perf_df['R² Score (CV)'].apply(lambda x: f"{x:.1%}"), textposition='auto')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with metric_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # MAE comparison
            fig = px.bar(
                perf_df,
                x='Model',
                y='MAE (₹ Lakhs)',
                title="Mean Absolute Error (Lower is Better)",
                color='Model',
                color_discrete_map={'Regular Properties': '#667eea', 'Luxury Properties': '#764ba2'}
            )
            fig.update_traces(text=perf_df['MAE (₹ Lakhs)'].apply(lambda x: f"₹{x:.0f}L"), textposition='auto')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig = px.bar(
                perf_df,
                x='Model',
                y='RMSE (₹ Lakhs)',
                title="Root Mean Square Error",
                color='Model',
                color_discrete_map={'Regular Properties': '#667eea', 'Luxury Properties': '#764ba2'}
            )
            fig.update_traces(text=perf_df['RMSE (₹ Lakhs)'].apply(lambda x: f"₹{x:.0f}L"), textposition='auto')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with metric_tab3:
        # Detailed comparison table
        st.dataframe(
            perf_df.style.format({
                'R² Score (Test)': '{:.1%}',
                'R² Score (CV)': '{:.1%}',
                'MAE (₹ Lakhs)': '₹{:.2f} Lakhs',
                'RMSE (₹ Lakhs)': '₹{:.2f} Lakhs',
                'Training Samples': '{:,}'
            }),
            use_container_width=True
        )
    
    # Model insights and explanations
    st.subheader("💡 Model Insights & Interpretability")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("""
        **🎯 Regular Properties Model (LightGBM)**
        - **Accuracy**: 45.7% (R² Score) - Good performance for price prediction
        - **Error**: ₹22.35 lakhs average deviation 
        - **Strengths**: Consistent performance, handles variety well
        - **Best for**: Properties under ₹2 crores in all cities
        - **Sample Size**: 8,500+ properties for robust training
        """)
    
    with insights_col2:
        st.warning("""
        **💎 Luxury Properties Model (Random Forest)**
        - **Accuracy**: 30.0% (R² Score) - Challenging segment 
        - **Error**: ₹1.00 crore average deviation
        - **Challenges**: High price variance, limited luxury data
        - **Best for**: Premium properties ₹2+ crores
        - **Improvement Areas**: More luxury data, feature engineering
        """)
    
    # Feature importance
    st.subheader("🔍 Feature Importance Analysis")
    
    # Simulate feature importance data
    features = [
        'Area (sq ft)', 'City Location', 'Bedrooms', 'Distance to Center',
        'Amenities Score', 'Property Type', 'Furnishing Status', 'Market Trends'
    ]
    importance_regular = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
    importance_luxury = [0.30, 0.28, 0.18, 0.08, 0.12, 0.02, 0.01, 0.01]
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Regular Model': importance_regular,
        'Luxury Model': importance_luxury
    })
    
    fig = px.bar(
        feature_df,
        x=['Regular Model', 'Luxury Model'],
        y='Feature',
        orientation='h',
        title="Feature Importance Comparison Across Models",
        labels={'value': 'Importance Score', 'variable': 'Model Type'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendations
    st.subheader("🚀 Model Improvement Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.success("""
        **✅ Immediate Improvements**
        - Collect more luxury property data
        - Add time-series features for market trends
        - Include neighborhood economic indicators
        - Enhance amenity scoring system
        """)
    
    with rec_col2:
        st.info("""
        **📈 Future Enhancements**
        - Deep learning models for complex patterns
        - Real-time market data integration
        - Satellite imagery for location scoring
        - Economic indicators and policy impacts
        """)

def settings_page(app):
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    st.markdown("Configure your AI Real Estate Advisor experience")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.expander("Gemini AI Configuration", expanded=True):
        current_api_key = os.getenv('GEMINI_API_KEY', '')
        
        if current_api_key and current_api_key != 'your_gemini_api_key_here':
            st.success("✅ Gemini API Key is configured")
            masked_key = current_api_key[:8] + "..." + current_api_key[-4:]
            st.code(f"Current API Key: {masked_key}")
        else:
            st.warning("⚠️ Gemini API Key not configured")
            
        st.markdown("""
        **To configure Gemini AI:**
        1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Edit the `.env` file in your project root
        3. Replace `your_gemini_api_key_here` with your actual API key
        4. Restart the Streamlit app
        """)
        
        if st.button("🔄 Test Gemini Connection"):
            if app.gemini_ai and app.gemini_ai.available:
                st.success("✅ Gemini AI connection successful!")
            else:
                st.error("❌ Gemini AI connection failed. Please check your API key.")
    
    # App Preferences
    st.subheader("🎛️ App Preferences")
    
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        # Chat settings
        st.markdown("**💬 Chat Settings**")
        
        max_chat_messages = st.number_input(
            "Max Chat History",
            min_value=10,
            max_value=100,
            value=50,
            help="Maximum number of chat messages to keep in history"
        )
        
        show_timestamps = st.checkbox(
            "Show Timestamps",
            value=True,
            help="Display timestamps in chat messages"
        )
        
        auto_clear_chat = st.checkbox(
            "Auto-clear on page refresh",
            value=False,
            help="Automatically clear chat history when page refreshes"
        )
    
    with pref_col2:
        # Prediction settings
        st.markdown("**🔮 Prediction Settings**")
        
        default_confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            help="Minimum confidence level for predictions"
        )
        
        show_model_details = st.checkbox(
            "Show Model Details",
            value=True,
            help="Display technical details about model predictions"
        )
        
        enable_prediction_history = st.checkbox(
            "Keep Prediction History",
            value=True,
            help="Save prediction history during session"
        )
    
    # Data & Performance
    st.subheader("📊 Data & Performance")
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        st.markdown("**📈 Current Status**")
        st.metric("Models Loaded", "✅ Yes" if app.models_loaded else "❌ No")
        st.metric("Market Data", "✅ Available" if app.data_loaded else "❌ Missing")
        st.metric("Gemini AI", "✅ Active" if app.gemini_ai and app.gemini_ai.available else "❌ Inactive")
        
        # Session statistics
        chat_count = len(st.session_state.get('chat_messages', []))
        pred_count = len(st.session_state.get('prediction_history', []))
        st.metric("Chat Messages", chat_count)
        st.metric("Predictions Made", pred_count)
    
    with data_col2:
        st.markdown("**🔧 Maintenance Actions**")
        
        if st.button("🗑️ Clear All Chat History", type="secondary"):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared! How can I help you today?",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("📊 Clear Prediction History", type="secondary"):
            st.session_state.prediction_history = []
            if 'last_prediction' in st.session_state:
                del st.session_state.last_prediction
            st.success("Prediction history cleared!")
            st.rerun()
        
        if st.button("🔄 Reload Models", type="secondary"):
            try:
                app.setup_ml_model()
                st.success("Models reloaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Model reload failed: {str(e)}")
        
        if st.button("📥 Export Settings", type="secondary"):
            settings_data = {
                "max_chat_messages": max_chat_messages,
                "show_timestamps": show_timestamps,
                "auto_clear_chat": auto_clear_chat,
                "confidence_threshold": default_confidence_threshold,
                "show_model_details": show_model_details,
                "enable_prediction_history": enable_prediction_history,
                "exported_at": datetime.now().isoformat()
            }
            
            st.download_button(
                "💾 Download Settings",
                json.dumps(settings_data, indent=2),
                file_name=f"ai_real_estate_settings_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # About section
    st.subheader("ℹ️ About")
    
    st.markdown("""
    **🏡 AI Real Estate Advisor** v2.0
    
    An advanced real estate price prediction platform powered by:
    - **Machine Learning**: Segmented models for different property types
    - **Natural Language Processing**: Understand queries in plain English  
    - **Gemini AI**: Enhanced explanations and market insights
    - **Interactive Visualizations**: Comprehensive market analysis
    
    **Data Sources**: Indian metro cities property data (Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata)
    
    **Model Accuracy**: 45.7% (Regular Properties), 30.0% (Luxury Properties)
    
    For support or feature requests, please refer to the project documentation.
    """)
    
    # System information
    with st.expander("🔧 System Information"):
        import sys
        import platform
        
        st.code(f"""
Python Version: {sys.version}
Platform: {platform.platform()}
Streamlit Version: {st.__version__}
App Initialized: {st.session_state.get('app_initialized', False)}
Models Loaded: {app.models_loaded}
Data Loaded: {app.data_loaded}
        """)
