# AI-Powered Location-Aware Conversational Real Estate Price Advisor

This project creates an intelligent real estate price prediction system with geospatial awareness and natural language interaction capabilities.

## Features

- Location-aware price prediction using ML models
- Geospatial intelligence with lat/long coordinates
- Natural language interaction via Gemini API
- Market insights and heatmaps
- Continuous learning from user interactions
- Streamlit frontend + FastAPI backend

## Dataset

Housing Prices in Metropolitan Areas of India by Ruchi798
https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india

## Setup Instructions

1. Create virtual environment: `python -m venv .venv`
2. Activate environment: `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Set up Kaggle API credentials
5. Run data pipeline step by step

## Project Structure

```
housing-prices/
├── data/
│   ├── raw/
│   ├── processed/
│   └── geo_enriched/
├── models/
├── src/
│   ├── data_processing/
│   ├── modeling/
│   ├── api/
│   └── frontend/
├── notebooks/
├── tests/
└── deployment/
```
