# Stock Analyzer - Full Stack Application

A comprehensive stock analysis platform with AI-powered recommendations, technical analysis, and market sentiment evaluation.

## Overview

This project consists of two main components:

1. **Backend API** (Flask): Provides endpoints for stock analysis, technical indicators, and AI recommendations
2. **Frontend Application** (React): User-friendly interface for querying stocks and visualizing analysis results

## Backend API

### Key Features

- Stock data retrieval with comprehensive technical indicators
- News sentiment analysis with weighted scoring
- AI-powered trading recommendations (supports OpenAI and Google Gemini)
- Memory-based contextual decision making
- Support for natural language stock queries

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   NEWS_API_KEY=your_newsapi_key
   OPENAI_API_KEY=your_openai_key  # Optional
   GEMINI_API_KEY=your_gemini_key  # Optional
   ```

3. Run the API:
   ```
   python app.py  # Standard version
   # or
   python app_with_cors.py  # CORS-enabled for frontend development
   ```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze a stock based on a query |
| `/stock/<ticker>` | GET | Get raw stock data for a ticker |
| `/news/<ticker>` | GET | Get news for a specific ticker |
| `/health` | GET | Health check endpoint |

## Frontend Application

### Key Features

- Clean, intuitive user interface built with React and Tailwind CSS
- Interactive stock price and volume charts
- Technical analysis display with key indicators
- AI recommendation visualization
- Real-time market news integration

### Setup

1. Navigate to the frontend directory:
   ```
   cd stock-analyzer-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

## Running the Full Stack

For convenience, a startup script is provided to run both components:

```
./start.sh
```

This will:
1. Start the backend API on port 5000
2. Start the frontend development server on port 8001
3. Automatically shut down both when you exit

## Implementation Details

### Backend Design

The backend is built with a modular structure:

- `TradingAnalyzer` class handles all analysis logic
- Robust error handling for API calls
- Fallback mechanisms for when AI services are unavailable
- SQLite database for storing trading history and memory
- CORS support for frontend integration

### Frontend Architecture

The frontend follows a component-based structure:

- Modular components for each part of the interface
- Responsive design that works on mobile and desktop
- Real-time API health monitoring
- Error handling with user-friendly messages
- Interactive data visualization with Recharts

## Deployment

### Backend

For production deployment:

1. Use Gunicorn as a WSGI server:
   ```
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. Consider using Docker with the provided Dockerfile:
   ```
   docker build -t stock-analyzer-api .
   docker run -p 5000:5000 --env-file .env stock-analyzer-api
   ```

### Frontend

For production deployment:

1. Build the frontend:
   ```
   cd stock-analyzer-frontend
   npm run build
   ```

2. Serve the static files from the `dist` directory using Nginx, Apache, or a CDN

## License

MIT