# Stock Analysis API

This API provides stock analysis and recommendations based on technical indicators, sentiment analysis, and AI suggestions.

## Features

- Stock data retrieval with technical indicators
- News sentiment analysis
- AI-powered trading recommendations
- Historical data analysis
- Memory-based contextual decision making

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   NEWS_API_KEY=your_newsapi_key
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   ```
   (Note: Either OPENAI_API_KEY or GEMINI_API_KEY is required for AI analysis)

4. Run the API:
   ```
   python app.py
   ```
   
## API Endpoints

### 1. Analyze Stock

```
POST /analyze
```

Request body:
```json
{
  "query": "Should I buy AAPL?"
}
```

Response:
```json
{
  "ticker": "AAPL",
  "currentPrice": 145.32,
  "technicalAnalysis": {
    "sma20": 143.21,
    "sma50": 140.87,
    "rsi14": 58.43,
    "support": 142.10,
    "resistance": 148.20,
    "volumeTrend": "Increasing",
    "priceTrend": "Upward",
    "momentum": "Strong",
    "volatility": 1.45,
    "riskLevel": "Medium"
  },
  "sentiment": {
    "overall": "POSITIVE",
    "score": 0.65
  },
  "recommendation": "BUY",
  "aiAnalysis": "Technical indicators show strong upward momentum with price above key moving averages...",
  "historicalData": {
    "prices": {
      "2023-05-01": 142.56,
      "2023-05-02": 144.32
    },
    "volumes": {
      "2023-05-01": 75242900,
      "2023-05-02": 82345600
    }
  },
  "news": [
    {
      "title": "Apple Reports Record Quarterly Earnings",
      "publishedAt": "2023-05-02T15:30:00Z",
      "source": "CNBC"
    }
  ]
}
```

### 2. Get Stock Data

```
GET /stock/{ticker}?period=1mo
```

Parameters:
- `ticker`: Stock symbol
- `period`: (optional) Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

### 3. Get News

```
GET /news/{ticker}?hours=24&limit=5
```

Parameters:
- `ticker`: Stock symbol
- `hours`: (optional) Look back hours
- `limit`: (optional) Maximum number of news items

### 4. Health Check

```
GET /health
```

## Using with Frontend

Example of how to call the API from your frontend (JavaScript/React):

```javascript
// Example using fetch API
const analyzeStock = async (query) => {
  try {
    const response = await fetch('http://your-api-url/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
    
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error analyzing stock:', error);
    return { error: error.message };
  }
};

// Usage
analyzeStock('What do you think about TSLA?')
  .then(result => {
    console.log(result);
    // Update your UI with the analysis result
  });
```

## Deployment

For production, it's recommended to:

1. Use Gunicorn as WSGI server:
   ```
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. Set up with Nginx or similar for production environments

3. Deploy on cloud platforms (AWS, Google Cloud, Azure) or container services (Docker, Kubernetes)

## Docker Support

A Dockerfile is included for containerization.

Build the Docker image:
```
docker build -t stock-analysis-api .
```

Run the container:
```
docker run -p 5000:5000 --env-file .env stock-analysis-api
```