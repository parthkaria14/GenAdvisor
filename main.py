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
        logger.info("✓ Data ingestion system initialized")
        
        # Initialize RAG system (file-based storage)
        rag_system = AdvancedRAGSystem(
            vector_db_type="chroma",
            enable_redis=False
        )
        logger.info("✓ RAG system initialized")
        
        # Initialize investment advisor engine
        advisor_engine = LocalInvestmentAdvisorEngine()
        logger.info("✓ Investment advisor engine initialized")
        
        # Initialize WebSocket manager
        websocket_manager = ConnectionManager()
        logger.info("✓ WebSocket manager initialized")
        
        # Start background tasks
        asyncio.create_task(market_data_updater())
        asyncio.create_task(alert_monitor())
        
        logger.info("✅ All components initialized successfully!")
        
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
                change = stock_data.get('change_percent', 0)
                if change > 0:
                    top_gainers.append({'symbol': stock_data['symbol'], 'change': change})
                else:
                    top_losers.append({'symbol': stock_data['symbol'], 'change': change})
        
        top_gainers.sort(key=lambda x: x['change'], reverse=True)
        top_losers.sort(key=lambda x: x['change'])
        
        return {
            "market_breadth": market_data.get('market_breadth', {}),
            "sector_performance": market_data.get('sector_performance', {}),
            "top_gainers": top_gainers[:5],
            "top_losers": top_losers[:5],
            "timestamp": datetime.now().isoformat()
        }
        
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
@app.post("/api/v1/query")
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
                    yield json.dumps({"chunk": chunk}) + "\n"
                logger.info(f"[QueryID: {query_id}] Stream completed")
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            logger.info(f"[QueryID: {query_id}] Starting RAG processing")
            result = await rag_system.process_query(request.query)
            logger.info(f"[QueryID: {query_id}] Query processed successfully")
            logger.debug(f"[QueryID: {query_id}] Response length: {len(str(result))} chars")
            return result
            
    except Exception as e:
        logger.error(f"[QueryID: {query_id}] Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Screener Endpoint
@app.post("/api/v1/screener")
async def screen_stocks(request: ScreenerRequest):
    """Screen stocks based on criteria"""
    try:
        results = []
        
        # Load all stock data
        stock_files = [f for f in os.listdir(data_ingestion.data_dir) if f.startswith('stock_')]
        
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
                
                results.append({
                    'symbol': stock_data['symbol'],
                    'price': stock_data.get('current_price', 0),
                    'pe_ratio': stock_data.get('pe_ratio'),
                    'market_cap': stock_data.get('market_cap', 0),
                    'sector': stock_data.get('sector', 'Unknown'),
                    'volume': stock_data.get('volume', 0)
                })
        
        # Sort by market cap
        results.sort(key=lambda x: x['market_cap'], reverse=True)
        
        return {
            "count": len(results),
            "stocks": results[:50],  # Return top 50 matches
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error screening stocks: {e}")
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
@app.post("/api/v1/admin/refresh-data")
async def refresh_market_data(background_tasks: BackgroundTasks):
    """Manually trigger market data refresh"""
    try:
        async def refresh_task():
            # Fetch data for top stocks
            top_stocks = data_ingestion.nse_tickers[:50]
            await data_ingestion.fetch_bulk_realtime(top_stocks)
            
            # Update indicators
            data_ingestion.update_technical_indicators()
            
            logger.info("Market data refresh completed")
        
        background_tasks.add_task(refresh_task)
        
        return {
            "status": "refresh_initiated",
            "message": "Market data refresh started in background",
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