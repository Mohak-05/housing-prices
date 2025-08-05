# API Configuration for Housing Prices ML Model
# Add your actual API keys here

# =============================================================================
# FREE APIs (No Keys Required) - Working Immediately!
# =============================================================================

OSM_OVERPASS_API = "https://overpass-api.de/api/interpreter"
OSM_NOMINATIM_API = "https://nominatim.openstreetmap.org"

# =============================================================================
# FREE APIs (Add Your Keys Here)
# =============================================================================

# OpenWeatherMap API - Get free key at: https://openweathermap.org/api
OPENWEATHER_API_KEY = None  # Replace with your key when you get it
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Google Places API - Optional, get at: https://console.cloud.google.com/
GOOGLE_PLACES_API_KEY = None  # Replace with your key when you get it
GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"

# Government APIs (Free registration)
RBI_API_KEY = None  # Register at: https://database.rbi.org.in/
MOSPI_API_KEY = None  # Register at: https://api.mospi.gov.in/

# =============================================================================
# Rate Limiting Configuration
# =============================================================================

RATE_LIMITS = {
    'osm_overpass': 1,      # 1 request per second
    'nominatim': 1,         # 1 request per second  
    'openweather': 60,      # 60 requests per minute
}

API_TIMEOUTS = {
    'default': 10,
    'osm_overpass': 30,
}
