# main_api.py
"""
Production-Ready FastAPI Application for Real-Time Indian Market Investment Advisor
Integrates data ingestion, RAG system, and investment advisor engine using file-based storage
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from enum import Enum
import uvicorn
import math
import numpy as np

# Import the custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components (assuming they are in the same directory)
from enhanced_data import IndianMarketDataIngestion
from advanced_rag_system import AdvancedRAGSystem, QueryType
from investment_advisor_engine import LocalInvestmentAdvisorEngine
from logger_config import setup_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = setup_logger()

# Utility function to clean NaN/Inf values for JSON serialization
def clean_for_json(obj):
    """Recursively clean NaN, Inf, and -Inf values from data structures for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        cleaned = [clean_for_json(item) for item in obj.tolist()]
        return cleaned
    else:
        return obj

# Initialize FastAPI app
app = FastAPI(
    title="Gen-Advisor API - Indian Market Investment Advisor",
    description="Real-time AI-powered investment advisory platform for the complete Indian stock market",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_ingestion = None
rag_system = None
advisor_engine = None
websocket_manager = None

# Request/Response Models
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE.NS)")
    
class PortfolioRequest(BaseModel):
    budget: float = Field(..., description="Investment budget in INR")
    strategy: str = Field("moderate", description="Investment strategy")
    existing_portfolio: Optional[Dict[str, float]] = Field(None, description="Existing portfolio holdings")
    constraints: Optional[Dict] = Field(None, description="Investment constraints")

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    stream: bool = Field(False, description="Stream response")
    
class MarketOverviewRequest(BaseModel):
    sectors: Optional[List[str]] = Field(None, description="Specific sectors to analyze")
    
class RiskAnalysisRequest(BaseModel):
    portfolio: Dict[str, float] = Field(..., description="Portfolio holdings")
    time_horizon: int = Field(30, description="Time horizon in days")

class ScreenerRequest(BaseModel):
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    sector: Optional[str] = None
    min_volume: Optional[float] = None
    include_predictions: Optional[bool] = False

class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="The formatted (Markdown) answer from the RAG system")
    confidence: float = Field(..., description="The confidence score (0.0 to 1.0)")
    timestamp: str = Field(..., description="The time the query was processed")   
# WebSocket Manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

# Initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global data_ingestion, rag_system, advisor_engine, websocket_manager
    
    logger.info("Initializing Gen-Advisor components...")
    
    try:
        # Initialize data ingestion (file-based storage)
        data_ingestion = IndianMarketDataIngestion(enable_redis=False)
        logger.info("[OK] Data ingestion system initialized")
        
        # Initialize RAG system (file-based storage)
        rag_system = AdvancedRAGSystem(
            vector_db_type="chroma",
            enable_redis=False
        )
        logger.info("[OK] RAG system initialized")
        
        # Initialize investment advisor engine
        advisor_engine = LocalInvestmentAdvisorEngine()
        logger.info("[OK] Investment advisor engine initialized")
        
        # Initialize WebSocket manager
        websocket_manager = ConnectionManager()
        logger.info("[OK] WebSocket manager initialized")
        
        # Start background tasks
        asyncio.create_task(market_data_updater())
        asyncio.create_task(alert_monitor())
        
        logger.info("[OK] All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Background Tasks
async def market_data_updater():
    """Background task to update market data periodically"""
    while True:
        try:
            # Update top stocks every 5 minutes during market hours
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # Market hours
                logger.info("Updating market data...")
                
                # Fetch data for top stocks
                top_stocks = data_ingestion.nse_tickers[:20]
                await data_ingestion.fetch_bulk_realtime(top_stocks)
                
                # Update market breadth
                breadth = data_ingestion.fetch_market_breadth()
                
                # Broadcast updates to WebSocket clients
                if websocket_manager and breadth:
                    await websocket_manager.broadcast(json.dumps({
                        'type': 'market_update',
                        'data': breadth,
                        'timestamp': datetime.now().isoformat()
                    }))
                
                await asyncio.sleep(300)  # 5 minutes
            else:
                await asyncio.sleep(3600)  # 1 hour outside market hours
                
        except Exception as e:
            logger.error(f"Error in market data updater: {e}")
            await asyncio.sleep(60)

async def alert_monitor():
    """Monitor for price alerts and notifications"""
    while True:
        try:
            # Check for alert conditions
            # This would check user-defined alerts from a database
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in alert monitor: {e}")
            await asyncio.sleep(60)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Gen-Advisor API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "market": "/api/v1/market/*",
            "analysis": "/api/v1/analyze/*",
            "portfolio": "/api/v1/portfolio/*",
            "query": "/api/v1/query"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "data_ingestion": data_ingestion is not None,
            "rag_system": rag_system is not None,
            "advisor_engine": advisor_engine is not None
        },
        "timestamp": datetime.now().isoformat()
    }

# Market Data Endpoints
@app.get("/api/v1/market/overview")
async def get_market_overview():
    """Get comprehensive market overview"""
    try:
        # Load market data from files
        market_data = data_ingestion.get_market_data_from_file()
        
        # Get top movers
        stock_files = [f for f in os.listdir(data_ingestion.data_dir) if f.startswith('stock_')]
        top_gainers = []
        top_losers = []
        
        for file in stock_files[:50]:  # Check top 50 stocks
            file_path = os.path.join(data_ingestion.data_dir, file)
            with open(file_path, 'r') as f:
                stock_data = json.load(f)
                change = stock_data.get('change_percent')
                # Handle None, NaN, and Inf values
                if change is None:
                    continue
                try:
                    change_float = float(change)
                    if math.isnan(change_float) or math.isinf(change_float):
                        continue
                    if change_float > 0:
                        top_gainers.append({'symbol': stock_data['symbol'], 'change': change_float})
                    else:
                        top_losers.append({'symbol': stock_data['symbol'], 'change': change_float})
                except (ValueError, TypeError):
                    continue
        
        top_gainers.sort(key=lambda x: x['change'], reverse=True)
        top_losers.sort(key=lambda x: x['change'])
        
        response = {
            "market_breadth": market_data.get('market_breadth', {}),
            "sector_performance": market_data.get('sector_performance', {}),
            "top_gainers": top_gainers[:5],
            "top_losers": top_losers[:5],
            "timestamp": datetime.now().isoformat()
        }
        
        # Clean response for JSON serialization
        return clean_for_json(response)
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get real-time data for a specific stock"""
    logger.info(f"Fetching stock data for {symbol}")
    
    try:
        # Check file storage
        logger.debug(f"Checking local file storage for {symbol}")
        stock_data = data_ingestion.get_stock_data_from_file(symbol)
        
        if not stock_data:
            logger.info(f"No cached data found for {symbol}, fetching fresh data")
            stock_data = await data_ingestion.fetch_realtime_price(symbol)
        else:
            logger.info(f"Using cached data for {symbol}")
        
        if not stock_data:
            logger.error(f"No data available for {symbol}")
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get technical indicators
        logger.debug(f"Fetching technical indicators for {symbol}")
        indicators_file = os.path.join(data_ingestion.data_dir, "technical_indicators.json")
        if os.path.exists(indicators_file):
            with open(indicators_file, 'r') as f:
                all_indicators = json.load(f)
                stock_data['technical_indicators'] = all_indicators.get(symbol, {})
                logger.info(f"Added {len(stock_data['technical_indicators'])} technical indicators")
        
        logger.info(f"Successfully retrieved data for {symbol}")
        return stock_data
        
    except Exception as e:
        logger.error(f"Error getting stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/sectors")
async def get_sector_performance():
    """Get sector-wise performance"""
    try:
        sectors = data_ingestion.fetch_sector_performance()
        return {
            "sectors": sectors,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sector performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis Endpoints
@app.post("/api/v1/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Comprehensive stock analysis"""
    try:
        result = await advisor_engine.analyze_stock(request.symbol)
        return result
    except Exception as e:
        logger.error(f"Error analyzing stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze/portfolio")
async def optimize_portfolio(request: PortfolioRequest):
    """Portfolio optimization and recommendations"""
    try:
        from investment_advisor_engine import InvestmentStrategy
        
        strategy_enum = InvestmentStrategy(request.strategy.lower())
        result = await advisor_engine.optimize_portfolio(
            budget=request.budget,
            strategy=strategy_enum,
            existing_portfolio=request.existing_portfolio,
            constraints=request.constraints
        )
        
        return {
            "allocation": result.stocks,
            "strategy": result.strategy.value,
            "expected_return": result.expected_return,
            "risk_level": result.risk_level,
            "sharpe_ratio": result.sharpe_ratio,
            "recommendations": result.recommendations,
            "rebalancing_needed": result.rebalancing_needed
        }
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze/risk")
async def analyze_risk(request: RiskAnalysisRequest):
    """Risk analysis for portfolio"""
    try:
        result = await advisor_engine.analyze_risk(
            portfolio=request.portfolio,
            time_horizon=request.time_horizon
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG Query Endpoint
# Note the new `response_model` for the non-streaming part
@app.post("/api/v1/query", response_model=RAGQueryResponse)
async def process_query(request: RAGQueryRequest):
    """Process natural language queries using RAG"""
    query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[QueryID: {query_id}] Processing query: {request.query}")
    
    try:
        if request.stream:
            logger.info(f"[QueryID: {query_id}] Streaming mode enabled")
            async def stream_generator():
                logger.info(f"[QueryID: {query_id}] Starting stream generation")
                result = await rag_system.process_query(request.query, stream=True)
                async for chunk in result:
                    logger.debug(f"[QueryID: {query_id}] Streaming chunk: {len(chunk)} chars")
                    # Make sure streaming also returns a clean JSON
                    yield json.dumps({"chunk": chunk}) + "\n"
                logger.info(f"[QueryID: {query_id}] Stream completed")
            
            # StreamingResponse is correct for streaming
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        else:
            # --- START OF FIX ---
            logger.info(f"[QueryID: {query_id}] Starting RAG processing")
            
            # 1. Get the full dictionary from the RAG system
            result = await rag_system.process_query(request.query)
            
            logger.info(f"[QueryID: {query_id}] Query processed successfully")
            logger.debug(f"[QueryID: {query_id}] Response length: {len(str(result.get('response', '')))} chars")
            
            # 2. Return ONLY the clean Pydantic model
            return RAGQueryResponse(
                answer=result.get('response', 'Sorry, I could not find an answer.'),
                confidence=result.get('confidence', 0.0),
                timestamp=result.get('timestamp', datetime.now().isoformat())
            )
            # --- END OF FIX ---
            
    except Exception as e:
        logger.error(f"[QueryID: {query_id}] Error processing query: {e}")
        # Also return a clean error that matches the response model
        return RAGQueryResponse(
            answer=f"Sorry, an internal error occurred: {e}",
            confidence=0.0,
            timestamp=datetime.now().isoformat()
        )
# Screener Endpoint
@app.post("/api/v1/screener")
async def screen_stocks(request: ScreenerRequest):
    """Screen stocks based on criteria"""
    try:
        logger.info(f"Screener request: include_predictions={request.include_predictions}")
        results = []
        
        # Load all stock data
        stock_files = [f for f in os.listdir(data_ingestion.data_dir) if f.startswith('stock_')]
        logger.info(f"Found {len(stock_files)} stock files to process")
        
        for file in stock_files:
            file_path = os.path.join(data_ingestion.data_dir, file)
            with open(file_path, 'r') as f:
                stock_data = json.load(f)
                
                # Apply filters
                if request.market_cap_min and stock_data.get('market_cap', 0) < request.market_cap_min:
                    continue
                if request.market_cap_max and stock_data.get('market_cap', 0) > request.market_cap_max:
                    continue
                if request.pe_min and stock_data.get('pe_ratio', 0) < request.pe_min:
                    continue
                if request.pe_max and stock_data.get('pe_ratio', 0) > request.pe_max:
                    continue
                if request.sector and stock_data.get('sector', '').lower() != request.sector.lower():
                    continue
                if request.min_volume and stock_data.get('volume', 0) < request.min_volume:
                    continue
                
                stock_result = {
                    'symbol': stock_data['symbol'],
                    'price': stock_data.get('current_price', 0),
                    'pe_ratio': stock_data.get('pe_ratio'),
                    'market_cap': stock_data.get('market_cap', 0),
                    'sector': stock_data.get('sector', 'Unknown'),
                    'volume': stock_data.get('volume', 0)
                }
                
                # Initialize predicted_price to None - will be filled later for top stocks only
                if request.include_predictions:
                    stock_result['predicted_price'] = None
                
                results.append(stock_result)
        
        # Sort by market cap (handle None values)
        results.sort(key=lambda x: x.get('market_cap') or 0, reverse=True)
        
        # Limit results
        limited_results = results[:50]
        
        # If predictions are requested, limit to top 20 stocks to avoid timeout
        if request.include_predictions:
            logger.info(f"Predictions enabled - processing top {min(20, len(limited_results))} stocks for predictions")
            # Process predictions for top stocks only
            prediction_count = 0
            for i, stock_result in enumerate(limited_results[:20]):
                if prediction_count >= 10:  # Limit to 10 predictions to avoid timeout
                    logger.info(f"Reached prediction limit (10 stocks)")
                    break
                symbol = stock_result['symbol']
                if stock_result.get('predicted_price') is None:
                    # Try to get prediction if it wasn't set yet
                    try:
                        # Use a simpler, faster prediction for screener
                        import yfinance as yf
                        import pandas as pd
                        from forecasting.arima_lstm_combo import arima_lstm_combo
                        
                        logger.info(f"Getting prediction for {symbol} ({i+1}/10)")
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period="6mo")
                        
                        if not hist.empty and len(hist) >= 30:
                            price_series = pd.Series(hist['Close'].values, index=hist.index).dropna()
                            if len(price_series) >= 30:
                                n_lags = max(5, min(8, len(price_series) // 8))
                                predicted_prices, _ = arima_lstm_combo(
                                    series=price_series,
                                    arima_order=(1, 1, 1),
                                    n_lags=n_lags,
                                    lstm_epochs=10,  # Very fast for screener
                                    forecast_horizon=5
                                )
                                if predicted_prices is not None and len(predicted_prices) > 0:
                                    cleaned_prices = [clean_for_json(float(p)) for p in predicted_prices]
                                    next_price = cleaned_prices[0]
                                    if next_price is not None:
                                        stock_result['predicted_price'] = next_price
                                        prediction_count += 1
                                        logger.info(f"[OK] Predicted {symbol}: {stock_result['price']:.2f} -> {next_price:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not predict {symbol}: {e}")
        
        # Clean all results for JSON serialization
        cleaned_results = clean_for_json(limited_results)
        
        logger.info(f"Screener returning {len(cleaned_results)} stocks, {sum(1 for s in cleaned_results if s.get('predicted_price') is not None)} with predictions")
        
        return {
            "count": len(cleaned_results),
            "stocks": cleaned_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error screening stocks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Recommendations Endpoint
@app.get("/api/v1/recommendations/{user_profile}")
async def get_recommendations(user_profile: str):
    """Get personalized recommendations based on user profile"""
    try:
        # Map user profiles to strategies
        profile_strategy = {
            "conservative": "conservative",
            "moderate": "moderate",
            "aggressive": "aggressive",
            "growth": "growth",
            "value": "value",
            "income": "income"
        }
        
        strategy = profile_strategy.get(user_profile.lower(), "moderate")
        
        # Get portfolio recommendation
        from investment_advisor_engine_local import InvestmentStrategy
        strategy_enum = InvestmentStrategy(strategy)
        
        result = await advisor_engine.optimize_portfolio(
            budget=1000000,  # Default 10 lakhs
            strategy=strategy_enum
        )
        
        # Get top stock picks
        top_stocks = []
        for symbol, weight in list(result.stocks.items())[:5]:
            stock_analysis = await advisor_engine.analyze_stock(symbol)
            top_stocks.append({
                'symbol': symbol,
                'allocation': weight * 100,
                'recommendation': stock_analysis['action'],
                'target_price': stock_analysis['target_price'],
                'confidence': stock_analysis['confidence']
            })
        
        return {
            "user_profile": user_profile,
            "recommended_strategy": strategy,
            "top_stocks": top_stocks,
            "expected_return": result.expected_return,
            "risk_level": result.risk_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            market_data = data_ingestion.get_market_data_from_file()
            await websocket.send_json({
                "type": "market_update",
                "data": market_data,
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Batch Operations
@app.post("/api/v1/batch/analyze")
async def batch_analyze_stocks(symbols: List[str]):
    """Analyze multiple stocks in batch"""
    try:
        results = []
        for symbol in symbols[:20]:  # Limit to 20 stocks
            try:
                analysis = await advisor_engine.analyze_stock(symbol)
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return {
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Historical Data Endpoint
@app.get("/api/v1/historical/{symbol}")
async def get_historical_data(symbol: str, period: str = "1mo"):
    """Get historical data for a stock"""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to JSON-friendly format
        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data": data,
            "count": len(data)
        }
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Price Prediction Endpoint
@app.get("/api/v1/predict/{symbol}")
async def predict_stock_price(symbol: str, forecast_horizon: int = 5):
    """Predict stock prices using ARIMA-LSTM combo model"""
    try:
        import yfinance as yf
        import pandas as pd
        from forecasting.arima_lstm_combo import arima_lstm_combo
        
        logger.info(f"Predicting prices for {symbol} with horizon={forecast_horizon}")
        
        # Get current price from realtime data
        current_price = None
        try:
            stock_data = data_ingestion.get_stock_data_from_file(symbol)
            if stock_data:
                current_price = stock_data.get('current_price', 0)
        except:
            pass
        
        # Fetch historical data (3 months for better training)
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")
        
        if hist.empty or len(hist) < 50:  # Need at least 50 data points
            logger.warning(f"Insufficient historical data for {symbol}, trying 6mo")
            hist = stock.history(period="6mo")
            if hist.empty or len(hist) < 50:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient historical data for {symbol}. Need at least 50 data points."
                )
        
        # Use closing prices as the series
        price_series = pd.Series(hist['Close'].values, index=hist.index)
        
        # Run ARIMA-LSTM combo prediction
        logger.info(f"Running ARIMA-LSTM combo model on {len(price_series)} data points")
        try:
            predicted_prices, models = arima_lstm_combo(
                series=price_series,
                arima_order=(1, 1, 1),
                n_lags=min(10, len(price_series) // 5),  # Adaptive n_lags
                lstm_epochs=20,  # Reduced for faster response
                forecast_horizon=forecast_horizon
            )
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction model failed: {str(e)}"
            )
        
        # If we don't have current price, use last historical price
        if current_price is None or current_price == 0:
            current_price = float(price_series.iloc[-1])
        
        # Convert numpy array to list and clean for JSON
        predicted_prices_list = clean_for_json([float(p) for p in predicted_prices])
        
        # Get next price, defaulting to current if invalid
        next_price = predicted_prices_list[0] if predicted_prices_list and predicted_prices_list[0] is not None else current_price
        
        logger.info(f"Prediction complete for {symbol}. Current: {current_price}, Predicted: {predicted_prices_list}")
        
        # Clean entire response
        response = {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_prices": predicted_prices_list,
            "forecast_horizon": forecast_horizon,
            "predicted_next_price": next_price,
            "timestamp": datetime.now().isoformat()
        }
        
        return clean_for_json(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting prices for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# News and Sentiment Endpoint
@app.get("/api/v1/news/{symbol}")
async def get_stock_news(symbol: str):
    """Get news and sentiment for a stock"""
    try:
        # Check if we have news data for this stock
        news_file_map = {
            "RELIANCE.NS": "news_Reliance_Industries.csv",
            "TCS.NS": "news_TCS.csv",
            "INFY.NS": "news_Infosys.csv",
            "HDFCBANK.NS": "news_HDFC_Bank.csv"
        }
        
        news_file = news_file_map.get(symbol)
        if news_file:
            news_path = os.path.join("data", news_file)
            if os.path.exists(news_path):
                import pandas as pd
                news_df = pd.read_csv(news_path)
                
                # Convert to list of dicts
                news_items = []
                for _, row in news_df.head(10).iterrows():  # Top 10 news
                    news_items.append({
                        "title": row.get('title', ''),
                        "description": row.get('description', ''),
                        "url": row.get('url', ''),
                        "published_at": row.get('publishedAt', '')
                    })
                
                # Analyze sentiment
                if rag_system and news_items:
                    texts = [item['title'] + " " + item['description'] for item in news_items]
                    sentiment_analysis = rag_system.analyze_sentiment(texts[:5])
                else:
                    sentiment_analysis = {"sentiment": "neutral", "score": 0}
                
                return {
                    "symbol": symbol,
                    "news": news_items,
                    "sentiment": sentiment_analysis,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "symbol": symbol,
            "news": [],
            "sentiment": {"sentiment": "neutral", "score": 0},
            "message": "No news data available for this symbol"
        }
        
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin Endpoints
# In main.py

@app.post("/api/v1/admin/refresh-data")
async def refresh_market_data(background_tasks: BackgroundTasks):
    """Manually trigger market data refresh"""
    try:
        async def refresh_task():
            # Step 1: Fetch new data
            logger.info("Admin Refresh: Fetching new market data...")
            top_stocks = data_ingestion.nse_tickers[:50]
            await data_ingestion.fetch_bulk_realtime(top_stocks)
            
            # Step 2: Update indicators
            logger.info("Admin Refresh: Updating technical indicators...")
            data_ingestion.update_technical_indicators()
            
            # --- START OF FIX ---
            # Step 3: Refresh the RAG system's graph
            logger.info("Admin Refresh: Rebuilding RAG knowledge graph...")
            if rag_system:
                rag_system.refresh_knowledge_graph()
            # --- END OF FIX ---
            
            logger.info("Market data and RAG refresh completed")
        
        background_tasks.add_task(refresh_task)
        
        return {
            "status": "refresh_initiated",
            "message": "Market data and RAG refresh started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/v1/admin/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        # Count data files
        data_dir = "data/realtime"
        stock_files = len([f for f in os.listdir(data_dir) if f.startswith('stock_')])
        
        # Get knowledge graph stats
        graph_stats = {
            "nodes": rag_system.knowledge_graph.number_of_nodes(),
            "edges": rag_system.knowledge_graph.number_of_edges()
        } if rag_system else {}
        
        return {
            "data": {
                "stock_files": stock_files,
                "tickers_tracked": len(data_ingestion.nse_tickers) + len(data_ingestion.bse_tickers)
            },
            "knowledge_graph": graph_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    # Run the application
    print("=" * 60)
    print("Starting Gen-Advisor API Server")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Set to False in production
    )