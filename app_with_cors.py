from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class TradingAnalyzer:
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.cache = {"stocks": {}, "news": []}
        self.db_path = "trading_memory.db"
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self._init_database()
        self.memory_vectors = None
        self.memory_data = None
        self._load_memory_vectors()
        
        # Configure AI model based on environment variables
        self.USE_OPENAI = False
        if os.getenv("OPENAI_API_KEY"):
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.USE_OPENAI = True
        else:
            try:
                import google.generativeai as genai
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
                    self.USE_OPENAI = False
                else:
                    self.USE_OPENAI = False
            except (ImportError, Exception) as e:
                print(f"Error configuring Gemini: {str(e)}. AI suggestion will be basic.")
                self.USE_OPENAI = False

    def _init_database(self):
        """Initialize SQLite database for storing trading memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    timestamp TEXT,
                    decision TEXT,
                    price REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sentiment TEXT,
                    sentiment_score REAL,
                    news_summary TEXT,
                    context TEXT
                )
            """)
            conn.commit()

    def _store_memory(self, ticker, decision, price, sma_20, sma_50, sentiment, sentiment_score, news, context):
        """Store trading decision and context in the database"""
        timestamp = datetime.now().isoformat()
        news_summary = " ".join([n['title'] for n in news[:3]]) if news else "No news"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trading_memory (ticker, timestamp, decision, price, sma_20, sma_50, sentiment, sentiment_score, news_summary, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker, timestamp, decision, price, sma_20, sma_50, sentiment, sentiment_score, news_summary, context))
            conn.commit()
        self._load_memory_vectors()

    def _load_memory_vectors(self):
        """Load and vectorize stored memories for retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, ticker, decision, news_summary, context FROM trading_memory")
            rows = cursor.fetchall()

        if not rows:
            self.memory_vectors = None
            self.memory_data = []
            return

        self.memory_data = [
            {"id": row[0], "ticker": row[1], "decision": row[2], "news_summary": row[3], "context": row[4]}
            for row in rows
        ]
        texts = [f"{d['ticker']} {d['decision']} {d['news_summary']} {d['context']}" for d in self.memory_data]
        self.memory_vectors = self.vectorizer.fit_transform(texts).toarray()

    def _retrieve_relevant_memories(self, ticker, current_context, top_k=3):
        """Retrieve top_k relevant past memories using cosine similarity"""
        if self.memory_vectors is None or not self.memory_data:
            return []

        query_text = f"{ticker} {current_context}"
        query_vector = self.vectorizer.transform([query_text]).toarray()

        similarities = np.dot(self.memory_vectors, query_vector.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            self.memory_data[i]
            for i in top_indices
            if similarities[i] > 0.1
        ]

    def fetch_stock_data(self, ticker, period="1mo"):
        """Fetch stock data with better error handling"""
        try:
            if not ticker or not isinstance(ticker, str) or len(ticker) > 5:
                return None

            if ticker in self.cache["stocks"]:
                return self.cache["stocks"][ticker]

            data = yf.Ticker(ticker).history(period=period)
            if data.empty:
                return None

            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.cache["stocks"][ticker] = data
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def get_news(self, query, hours=24, limit=5):
        """Fetch news with fallback to sample data"""
        if not self.news_api_key:
            return self._get_sample_news()
        try:
            url = "https://newsapi.org/v2/everything"
            from_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            response = requests.get(url, params={
                'q': query,
                'from': from_time,
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'pageSize': limit,
                'domains': 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,financialpost.com'
            }).json()
            if response.get('status') != 'ok':
                return self._get_sample_news()
            self.cache["news"] = response.get('articles', [])
            return self.cache["news"]
        except Exception as e:
            print(f"News fetch error: {e}")
            return self._get_sample_news()

    def enhanced_sentiment_analysis(self, text):
        """Improved sentiment analysis with weighting and context"""
        if not text:
            return "NEUTRAL", 0
        positive_terms = {
            'up': 1, 'rise': 1.2, 'gain': 1, 'profit': 1.5, 'growth': 1.3,
            'positive': 1.2, 'surge': 1.5, 'rally': 1.3, 'boost': 1,
            'success': 1.4, 'record high': 2, 'beat': 1.5, 'exceed': 1.3,
            'strong': 1.2, 'bullish': 1.8, 'increase': 1, 'win': 1.5
        }
        negative_terms = {
            'down': 1, 'fall': 1.2, 'loss': 1.5, 'decline': 1.3,
            'negative': 1.2, 'drop': 1.3, 'plunge': 1.7, 'crisis': 2,
            'risk': 1.3, 'concern': 1.2, 'miss': 1.5, 'weak': 1.2,
            'bearish': 1.8, 'underperform': 1.6, 'decrease': 1, 'fail': 1.5
        }
        intensifiers = {'very': 1.5, 'extremely': 2, 'highly': 1.3, 'significantly': 1.4}
        diminishers = {'slightly': 0.7, 'marginally': 0.6, 'somewhat': 0.8}
        text_lower = text.lower()
        words = text_lower.split()
        positive_score = 0
        negative_score = 0
        for i, word in enumerate(words):
            for term, weight in positive_terms.items():
                if term in word:
                    modifier = 1
                    if i > 0 and words[i - 1] in intensifiers:
                        modifier = intensifiers[words[i - 1]]
                    elif i > 0 and words[i - 1] in diminishers:
                        modifier = diminishers[words[i - 1]]
                    positive_score += weight * modifier
            for term, weight in negative_terms.items():
                if term in word:
                    modifier = 1
                    if i > 0 and words[i - 1] in intensifiers:
                        modifier = intensifiers[words[i - 1]]
                    elif i > 0 and words[i - 1] in diminishers:
                        modifier = diminishers[words[i - 1]]
                    negative_score += weight * modifier
        total = positive_score + negative_score
        if total == 0:
            return "NEUTRAL", 0
        sentiment_score = (positive_score - negative_score) / total
        sentiment_score = max(-1, min(1, sentiment_score))
        if sentiment_score > 0.2:
            return "POSITIVE", sentiment_score
        elif sentiment_score < -0.2:
            return "NEGATIVE", sentiment_score
        else:
            return "NEUTRAL", sentiment_score

    def _extract_ticker(self, text):
        """Improved stock ticker extraction from user input"""
        # Company name to ticker mapping
        company_mapping = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'tesla': 'TSLA',
            'nvidia': 'NVDA',
            'amazon': 'AMZN',
            'google': 'GOOG',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
            'alphabet': 'GOOG'
        }

        text_lower = text.lower()
        for name, ticker in company_mapping.items():
            if name in text_lower:
                return ticker

        # Common tickers
        common_tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX',
                        'SPY', 'QQQ', 'BTC-USD', 'ETH-USD']

        # Split text into words and clean them
        words = [word.upper().strip('.,!?') for word in text.split()]

        # Check for exact matches first
        for word in words:
            if word in common_tickers:
                return word

        # If no exact match, look for potential tickers (3-5 letters, all uppercase)
        for word in words:
            if 3 <= len(word) <= 5 and word.isalpha() and word.isupper():
                # Skip common words that might look like tickers
                if word in ['BUY', 'SELL', 'HOLD', 'NOW', 'THE', 'AND', 'FOR', 'YOU']:
                    continue
                return word

        # If still not found, try to find the first proper noun that could be a ticker
        for word in words:
            if len(word) >= 2 and word.isalpha() and word.isupper():
                return word

        return None

    def get_ai_suggestion(self, ticker, data, news, past_memories):
        """Get AI trading suggestion with comprehensive analysis"""
        memories_text = "\n".join([
            f"Past Decision for {m['ticker']}: {m['decision']} (Context: {m['context']}, News: {m['news_summary']})"
            for m in past_memories
        ]) if past_memories else "No relevant past memories."

        # Calculate technical indicators
        rsi = data['Close'].diff()
        rsi_gain = rsi.copy()
        rsi_loss = rsi.copy()
        rsi_gain[rsi_gain < 0] = 0
        rsi_loss[rsi_loss > 0] = 0
        rsi_14 = 100 - (100 / (1 + (rsi_gain.rolling(14).mean() / -rsi_loss.rolling(14).mean())))
        current_rsi = rsi_14.iloc[-1] if len(rsi_14) > 0 else None
        rsi_str = f"{current_rsi:.2f}" if current_rsi is not None else "N/A"

        # Calculate moving averages
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        latest_close = data['Close'].iloc[-1]

        # Determine trend direction and strength
        volume_trend = 'Increasing' if data['Volume'].iloc[-5:].mean() > data['Volume'].iloc[-10:-5].mean() else 'Decreasing'
        price_trend = 'Upward' if latest_close > data['Close'].iloc[-5:].mean() else 'Downward'
        
        # Calculate momentum and trend strength
        momentum = 'Strong' if abs(latest_close - data['Close'].iloc[-5]) / data['Close'].iloc[-5] > 0.02 else 'Weak'
        trend_strength = f"{momentum} {price_trend}"

        # Support and resistance levels
        support = data['Low'].rolling(window=20).min().iloc[-1]
        resistance = data['High'].rolling(window=20).max().iloc[-1]

        # Risk assessment based on volatility and volume
        volatility = data['Close'].pct_change().std() * 100
        risk_level = 'High' if volatility > 2 else 'Medium' if volatility > 1 else 'Low'

        # Confidence assessment based on indicator alignment
        indicator_alignment = sum([
            1 if current_rsi and current_rsi > 50 and price_trend == 'Upward' else -1 if current_rsi and current_rsi < 50 and price_trend == 'Downward' else 0,
            1 if volume_trend == 'Increasing' else -1,
            1 if latest_close > sma_20.iloc[-1] > sma_50.iloc[-1] else -1 if latest_close < sma_20.iloc[-1] < sma_50.iloc[-1] else 0
        ])
        confidence_level = 'High' if abs(indicator_alignment) >= 2 else 'Medium' if abs(indicator_alignment) == 1 else 'Low'

        if self.USE_OPENAI:
            import openai
            prompt = f"""
            Analyze {ticker} stock and provide a trading recommendation with the following format:

            Analysis Summary:
            - Trend: {trend_strength}
            - Current Price: ${latest_close:.2f}
            - RSI (14): {rsi_str}
            - Support Level: ${support:.2f}
            - Resistance Level: ${resistance:.2f}
            - Volume Trend: {volume_trend}
            - Risk Level: {risk_level}
            - Confidence Level: {confidence_level}

            Technical Analysis:
            {data.tail().to_string()}

            Market News:
            {[n['title'] for n in news[:3]] if news else 'No recent news'}

            Historical Context:
            {memories_text}

            Based on this data, provide:
            1. Trend Direction: [Bullish/Bearish/Neutral]
            2. Momentum: [Strong/Weak] [Bullish/Bearish]
            3. Sentiment: [Positive/Neutral/Negative]
            4. Final Recommendation: [BUY/HOLD/SELL]
            5. One-sentence rationale combining key factors
            6. Risk Level: [Low/Medium/High]
            7. Confidence Level: [Low/Medium/High]
            """
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=250
                )
                return response.choices[0].message['content']
            except Exception as e:
                print(f"OpenAI error: {e}")
                return None
        else:
            try:
                if os.getenv("GEMINI_API_KEY"):
                    import google.generativeai as genai
                    model = genai.GenerativeModel('gemini-pro')
                    # Calculate volatility
                    volatility = (data['Close'].pct_change().std() * 100).round(2)
                    
                    prompt = f"""
                    Please analyze {ticker} stock and provide a trading recommendation in the following exact format:

                    Technical Analysis Summary:
                    - Current Price: ${latest_close:.2f}
                    - 20-day SMA: ${sma_20.iloc[-1]:.2f} ({latest_close > sma_20.iloc[-1] and 'Above' or 'Below'})
                    - 50-day SMA: ${sma_50.iloc[-1]:.2f} ({latest_close > sma_50.iloc[-1] and 'Above' or 'Below'})
                    - RSI (14): {rsi_str}
                    - Support Level: ${support:.2f}
                    - Resistance Level: ${resistance:.2f}

                    Trend Analysis:
                    - Price Trend: {price_trend}
                    - Volume Trend: {volume_trend}
                    - Momentum: {momentum}

                    Risk Assessment:
                    - Volatility: {volatility}%
                    - Risk Level: {risk_level}
                    - Market Sentiment: {sentiment} (Score: {score:.2f})

                    AI Analysis:
                    [Provide a 2-3 sentence analysis of the technical indicators, market conditions, and key factors influencing the trading decision]

                    Final Recommendation: **[BUY/HOLD/SELL]**
                    Rationale: [One clear sentence explaining the main reason for the recommendation]
                    Risk Level: [High/Medium/Low]
                    Confidence: [High/Medium/Low]

                    Please maintain this exact format and section structure in your response.
                    """
                    response = model.generate_content(prompt)
                    return response.text
                else:
                    # Basic analysis when no AI service is available
                    latest_price = data['Close'].iloc[-1] if not data.empty else 0
                    # Make a basic decision based on price trend
                    decision = "HOLD"
                    if price_trend == "Upward":
                        decision = "BUY"
                    elif price_trend == "Downward":
                        decision = "SELL"
                    return "Based on technical analysis:\n" + \
                        f"- Price is ${latest_price:.2f}\n" + \
                        f"- Volume trend is {volume_trend}\n" + \
                        f"- Price trend is {price_trend}\n\n" + \
                        f"Recommendation: {decision} based on current market conditions."
            except Exception as e:
                print(f"AI suggestion error: {e}")
                latest_price = data['Close'].iloc[-1] if not data.empty else 0
                # Make a basic decision based on price trend
                decision = "HOLD"
                if price_trend == "Upward":
                    decision = "BUY"
                elif price_trend == "Downward":
                    decision = "SELL"
                return "Based on technical analysis:\n" + \
                    f"- Price is ${latest_price:.2f}\n" + \
                    f"- Volume trend is {volume_trend}\n" + \
                    f"- Price trend is {price_trend}\n\n" + \
                    f"Recommendation: {decision} based on current market conditions."

    def _make_decision(self, price, sma_20, sma_50, sentiment, score, past_memories):
        """Decision-making logic incorporating past memories"""
        if sma_20 is None:
            return "HOLD"

        tech_score = 0
        if sma_50 is not None:
            if price > sma_20 > sma_50:
                tech_score = 1
            elif price < sma_20 < sma_50:
                tech_score = -1
        else:
            if price > sma_20:
                tech_score = 0.5
            else:
                tech_score = -0.5

        sentiment_weight = score * 1.5

        memory_score = 0
        if past_memories:
            for mem in past_memories:
                if mem['decision'] == 'BUY':
                    memory_score += 0.3
                elif mem['decision'] == 'SELL':
                    memory_score -= 0.3
            memory_score /= len(past_memories)

        total_score = tech_score + sentiment_weight + memory_score

        if total_score > 0.5:
            return "BUY"
        elif total_score < -0.5:
            return "SELL"
        else:
            return "HOLD"

    def generate_analysis(self, user_input):
        """Generate stock analysis based on user input"""
        if not user_input or not isinstance(user_input, str):
            return {"error": "Please enter a valid question about a stock"}

        ticker = self._extract_ticker(user_input)
        if not ticker:
            return {"error": "Please specify a stock ticker/symbol (e.g., 'What about AAPL?' or 'Analyze TSLA')"}

        data = self.fetch_stock_data(ticker, period="3mo")
        if data is None:
            return {"error": f"Could not fetch data for {ticker}. Please check the ticker symbol and try again."}

        news = self.get_news(ticker)
        latest_close = data['Close'].iloc[-1]
        sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else None
        sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None

        news_text = " ".join([n['title'] for n in news]) if news else ""
        sentiment, score = self.enhanced_sentiment_analysis(news_text)

        # Calculate RSI
        rsi = data['Close'].diff()
        rsi_gain = rsi.copy()
        rsi_loss = rsi.copy()
        rsi_gain[rsi_gain < 0] = 0
        rsi_loss[rsi_loss > 0] = 0
        rsi_14 = 100 - (100 / (1 + (rsi_gain.rolling(14).mean() / -rsi_loss.rolling(14).mean())))
        current_rsi = rsi_14.iloc[-1] if len(rsi_14) > 0 else None

        # Calculate trend and momentum
        volume_trend = 'Increasing' if data['Volume'].iloc[-5:].mean() > data['Volume'].iloc[-10:-5].mean() else 'Decreasing'
        price_trend = 'Upward' if latest_close > data['Close'].iloc[-5:].mean() else 'Downward'
        momentum = 'Strong' if abs(latest_close - data['Close'].iloc[-5]) / data['Close'].iloc[-5] > 0.02 else 'Weak'

        # Support and resistance levels
        support = data['Low'].rolling(window=20).min().iloc[-1]
        resistance = data['High'].rolling(window=20).max().iloc[-1]

        # Risk assessment
        volatility = data['Close'].pct_change().std() * 100
        risk_level = 'High' if volatility > 2 else 'Medium' if volatility > 1 else 'Low'

        # Format technical indicators
        sma_20_str = f"${sma_20:.2f}" if sma_20 is not None else "N/A"
        sma_20_comparison = 'Above' if sma_20 and latest_close > sma_20 else 'Below' if sma_20 else ''
        sma_50_str = f"${sma_50:.2f}" if sma_50 is not None else "N/A"
        sma_50_comparison = 'Above' if sma_50 and latest_close > sma_50 else 'Below' if sma_50 else ''
        rsi_str = f"{current_rsi:.2f}" if current_rsi is not None else "N/A"

        # Prepare analysis context
        current_context = f"Price: ${latest_close:.2f}, 20-day SMA: {sma_20_str}, 50-day SMA: {sma_50_str}, RSI: {rsi_str}, Trend: {price_trend}, Momentum: {momentum}, Risk: {risk_level}"
        past_memories = self._retrieve_relevant_memories(ticker, current_context)

        # Get AI suggestion and decision
        ai_rec = self.get_ai_suggestion(ticker, data, news, past_memories)
        if not ai_rec:
            ai_rec = "Technical analysis suggests monitoring price action and volume patterns for confirmation of trend direction."

        decision = self._make_decision(latest_close, sma_20, sma_50, sentiment, score, past_memories)

        self._store_memory(ticker, decision, latest_close, sma_20, sma_50, sentiment, score, news, current_context)

        # Format data for API response
        historical_prices = data['Close'].tail(30).to_dict()
        historical_volumes = data['Volume'].tail(30).to_dict()
        
        # Convert datetime index to string for JSON serialization
        historical_prices_formatted = {str(date): value for date, value in historical_prices.items()}
        historical_volumes_formatted = {str(date): value for date, value in historical_volumes.items()}
        
        formatted_news = []
        for n in news[:5]:
            formatted_news.append({
                'title': n.get('title', ''),
                'publishedAt': n.get('publishedAt', ''),
                'source': n.get('source', {}).get('name', '')
            })

        return {
            "ticker": ticker,
            "currentPrice": latest_close,
            "technicalAnalysis": {
                "sma20": sma_20,
                "sma50": sma_50,
                "rsi14": current_rsi,
                "support": support,
                "resistance": resistance,
                "volumeTrend": volume_trend,
                "priceTrend": price_trend,
                "momentum": momentum,
                "volatility": volatility,
                "riskLevel": risk_level
            },
            "sentiment": {
                "overall": sentiment,
                "score": score
            },
            "recommendation": decision,
            "aiAnalysis": ai_rec,
            "historicalData": {
                "prices": historical_prices_formatted,
                "volumes": historical_volumes_formatted
            },
            "news": formatted_news
        }

    def _get_sample_news(self):
        """Fallback sample news"""
        return [{
            'title': 'Markets rally after positive earnings reports',
            'publishedAt': datetime.now().isoformat(),
            'source': {'name': 'Sample News'}
        }, {
            'title': 'Tech stocks face pressure amid rate hike concerns',
            'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
            'source': {'name': 'Sample News'}
        }]

# Initialize the analyzer
analyzer = TradingAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """Endpoint to analyze a stock based on user query"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    query = data['query']
    response = analyzer.generate_analysis(query)
    return jsonify(response)

@app.route('/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    """Endpoint to get stock data for a specific ticker"""
    period = request.args.get('period', '1mo')
    data = analyzer.fetch_stock_data(ticker, period=period)
    
    if data is None:
        return jsonify({"error": f"Could not fetch data for {ticker}"}), 404
    
    # Format data for API response
    result = {
        "ticker": ticker,
        "prices": {str(date): row["Close"] for date, row in data.iterrows()},
        "volumes": {str(date): row["Volume"] for date, row in data.iterrows()}
    }
    return jsonify(result)

@app.route('/news/<ticker>', methods=['GET'])
def get_news_data(ticker):
    """Endpoint to get news for a specific ticker"""
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 5, type=int)
    
    news = analyzer.get_news(ticker, hours=hours, limit=limit)
    
    # Format news for API response
    formatted_news = []
    for n in news:
        formatted_news.append({
            'title': n.get('title', ''),
            'publishedAt': n.get('publishedAt', ''),
            'source': n.get('source', {}).get('name', '')
        })
    
    return jsonify({"ticker": ticker, "news": formatted_news})

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)