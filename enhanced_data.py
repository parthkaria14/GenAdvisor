import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available - caching will be disabled")
import json
from bs4 import BeautifulSoup
import logging
from logger_config import setup_logger
import threading
logger = setup_logger()

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
from concurrent.futures import ThreadPoolExecutor
import schedule
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class MarketSegment(Enum):
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    PENNY = "penny"
    
@dataclass
class StockData:
    symbol: str
    exchange: str
    name: str
    sector: str
    market_cap: float
    segment: MarketSegment
    current_price: float
    change_percent: float
    volume: int
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

class IndianMarketDataIngestion:
    """
    Comprehensive data ingestion for the entire Indian stock market
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, enable_redis=True):
        self.enable_redis = enable_redis and REDIS_AVAILABLE
        if self.enable_redis:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
                self.enable_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
            logger.info("Redis caching disabled")
            
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.session = None
        
        # Complete ticker lists
        self.nse_tickers = self._load_nse_tickers()
        self.bse_tickers = self._load_bse_tickers()
        self.commodity_tickers = self._load_commodity_tickers()
        
        # Data storage
        self.data_dir = "data/realtime"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File storage paths
        self.stock_data_file = os.path.join(self.data_dir, "stock_data.json")
        self.market_breadth_file = os.path.join(self.data_dir, "market_breadth.json")
        self.sector_performance_file = os.path.join(self.data_dir, "sector_performance.json")
        self.technical_indicators_file = os.path.join(self.data_dir, "technical_indicators.json")
        
        # CSV files for structured data
        self.stocks_csv = os.path.join(self.data_dir, "stocks.csv")
        self.market_summary_csv = os.path.join(self.data_dir, "market_summary.csv")
        
    def _cache_set(self, key: str, value: str, expiry: int = 60) -> bool:
        """Set cache value if Redis is available"""
        if self.enable_redis and self.redis_client:
            try:
                self.redis_client.setex(key, expiry, value)
                return True
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
        return False
    
    def _cache_set_dict(self, key: str, data: dict, expiry: int = 60) -> bool:
        """Set cache value for dictionary data with numpy type conversion"""
        try:
            # Convert numpy types to native Python types
            converted_data = convert_numpy_types(data)
            json_str = json.dumps(converted_data)
            return self._cache_set(key, json_str, expiry)
        except Exception as e:
            logger.warning(f"Cache set dict failed: {e}")
            return False
    
    def _cache_get(self, key: str) -> Optional[str]:
        """Get cache value if Redis is available"""
        if self.enable_redis and self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        return None
    
    def _save_to_file(self, file_path: str, data: dict) -> bool:
        """Save data to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
            return False
    
    def _load_from_file(self, file_path: str) -> dict:
        """Load data from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading from {file_path}: {e}")
            return {}
    
    def _save_stock_data_to_csv(self, stock_data: List[Dict]):
        """Save stock data to CSV file"""
        try:
            if stock_data:
                df = pd.DataFrame(stock_data)
                df.to_csv(self.stocks_csv, index=False)
                logger.info(f"Saved {len(stock_data)} stock records to CSV")
        except Exception as e:
            logger.error(f"Error saving stock data to CSV: {e}")
    
    def _save_market_summary_to_csv(self, breadth: dict, sectors: dict):
        """Save market summary to CSV file"""
        try:
            summary_data = []
            timestamp = datetime.now().isoformat()
            
            # Market breadth data
            breadth_row = {
                'timestamp': timestamp,
                'type': 'market_breadth',
                'advances': breadth.get('advances', 0),
                'declines': breadth.get('declines', 0),
                'unchanged': breadth.get('unchanged', 0),
                'advance_decline_ratio': breadth.get('advance_decline_ratio', 0),
                'market_sentiment': breadth.get('market_sentiment', 'neutral')
            }
            summary_data.append(breadth_row)
            
            # Sector performance data
            for sector, data in sectors.items():
                sector_row = {
                    'timestamp': timestamp,
                    'type': 'sector_performance',
                    'sector': sector,
                    'change_percent': data.get('change_percent', 0),
                    'volume': data.get('volume', 0)
                }
                summary_data.append(sector_row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                # Append to existing file or create new one
                if os.path.exists(self.market_summary_csv):
                    df.to_csv(self.market_summary_csv, mode='a', header=False, index=False)
                else:
                    df.to_csv(self.market_summary_csv, index=False)
                logger.info(f"Saved market summary to CSV")
        except Exception as e:
            logger.error(f"Error saving market summary to CSV: {e}")
        
    def _load_nse_tickers(self) -> List[str]:
        """Load all NSE tickers including small and mid-cap"""
        # This would normally fetch from NSE API or maintain a database
        # For demonstration, including major categories
        tickers = []
        
        # Nifty 50
        # Nifty 50
        nifty_50 = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
            "ITC.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "HCLTECH.NS",
            "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
            "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIENT.NS", "ADANIPORTS.NS",
            "ASIANPAINT.NS", "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
            "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
            "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "IOC.NS",
            "JSWSTEEL.NS", "M&M.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
            "POWERGRID.NS", "SBILIFE.NS", "SHREECEM.NS", "UPL.NS"
        ]

        nifty_next_50 = [
            "ADANIGREEN.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "AUROPHARMA.NS", "BANDHANBNK.NS",
            "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "ZYDUSLIFE.NS",
            "COLPAL.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
            "HDFCAMC.NS", "ICICIPRULI.NS", "ICICIGI.NS", "INDUSIND.NS", "JINDALSTEL.NS",
            "LUPIN.NS", "MINDTREE.NS", "MOTHERSON.NS", "PAGEIND.NS", "PEL.NS",
            "PHOENIXLTD.NS", "PIIND.NS", "RECLTD.NS", "SAIL.NS", "SBILIFE.NS",
            "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SUNTV.NS", "TATACONSUMER.NS",
            "TATAELXSI.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS", "WIPRO.NS"
        ]
        
        # Include in tickers list
        tickers.extend(nifty_50)
        tickers.extend(nifty_next_50)
        
        return list(set(tickers))  # Unique tickers only
    
    def _load_bse_tickers(self) -> List[str]:
        """Load all BSE tickers"""
        # Placeholder for BSE tickers
        return [
            "500325.BO", "532540.BO", "500008.BO", "532174.BO", "500696.BO",
            "532281.BO", "500124.BO", "532555.BO", "500209.BO", "532648.BO",
            "500330.BO", "532660.BO", "500312.BO", "532755.BO", "500325.BO",
            "532843.BO", "500408.BO", "533151.BO", "500470.BO", "532540.BO"
        ]
    
    def _load_commodity_tickers(self) -> List[str]:
        """Load all commodity tickers"""
        # Placeholder for commodity tickers
        return [
            "GC=CC", "SI=CC", "HG=CC", "PL=CC", "PA=CC",
            "CL=CC", "NG=CC", "ZC=CC", "KC=CC", "CC=CC"
        ]
    
    async def fetch_bulk_realtime(self, tickers: List[str]):
        """Fetch real-time data for a bulk of tickers"""
        logger.info(f"Fetching real-time data for {len(tickers)} tickers")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for ticker in tickers:
                tasks.append(self.fetch_realtime_price(ticker))
            
            # Gather results with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in bulk fetch: {result}")
    
    async def fetch_realtime_price(self, ticker: str) -> Optional[Dict]:
        """Fetch real-time price AND fundamental data for a single ticker"""
        logger.info(f"Fetching real-time data for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            # This is a blocking call, but it's the simplest way 
            # to get all the data you need
            info = stock.info 
            
            # Use 'info' to get all data
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            if current_price == 0:
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            data = {
                'symbol': ticker,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                
                # --- START OF FIX ---
                # Populate all the data the RAG system needs
                'name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'volume': info.get('volume'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'change_percent': info.get('regularMarketChangePercent')
                # --- END OF FIX ---
            }
            
            # Save to file storage
            file_path = f"data/realtime/stock_{ticker.replace('.', '_')}.json"
            logger.debug(f"Saving data to {file_path}")
            self._save_to_file(file_path, data)
            
            logger.info(f"Successfully fetched and stored data for {ticker}")
            return data
                
        except Exception as e:
            # yfinance often fails if a ticker is invalid or delisted
            logger.error(f"Error fetching info for {ticker}: {e}")
            return None
    def fetch_and_save_news(self):
        """Fetches news for all tickers and saves to individual CSV files."""
        logger.info("Fetching news for all tickers...")
        tickers_to_fetch = self.nse_tickers[:50] + self.bse_tickers[:10] 
        
        for ticker in tickers_to_fetch:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                # Check if news is a valid list
                if news and isinstance(news, list) and len(news) > 0:
                    news_df = pd.DataFrame(news)
                    
                    # Standardize column names
                    news_df = news_df.rename(columns={
                        'uuid': 'id',
                        'providerPublishTime': 'publishedAt',
                        'relatedTickers': 'symbols',  # This is the key rename
                        'link': 'url'
                    })
                    
                    # --- START OF FIX ---
                    # Ensure ALL essential columns exist, even if empty
                    # This guarantees the 'symbols' column exists and prevents the KeyError
                    required_cols = ['title', 'publisher', 'url', 'publishedAt', 'symbols', 'description']
                    for col in required_cols:
                        if col not in news_df.columns:
                            news_df[col] = 'N/A' # Add 'N/A' for missing columns
                    # --- END OF FIX ---
                            
                    file_path = os.path.join(self.data_dir, f"news_{ticker.replace('.', '_')}.csv")
                    news_df.to_csv(file_path, index=False)
                    logger.info(f"Saved news for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to fetch news for {ticker}: {e}")
            time.sleep(1) # Rate limiting
    def fetch_market_breadth(self) -> Dict:
        """Fetch market breadth indicators"""
        try:
            advances = 0
            declines = 0
            unchanged = 0
            
            # Sample calculation (would use real-time data)
            for ticker in self.nse_tickers[:100]:  # Sample for speed
                cached = self._cache_get(f"stock:{ticker}")
                if cached:
                    data = json.loads(cached)
                    change = data.get('change_percent', 0)
                    if change > 0:
                        advances += 1
                    elif change < 0:
                        declines += 1
                    else:
                        unchanged += 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advances / declines if declines > 0 else float('inf'),
                'market_sentiment': 'bullish' if advances > declines else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {}
    
    def fetch_sector_performance(self) -> Dict:
        """Analyze sector-wise performance"""
        sectors = {}
        
        # Define sector indices
        sector_indices = {
            'Banking': 'NIFTYBEES.NS',
            'IT': 'ITBEES.NS',
            'Pharma': 'PHARMABEES.NS',
            'Auto': '^CNXAUTO',
            'FMCG': '^CNXFMCG',
            'Metal': '^CNXMETAL',
            'Realty': '^CNXREALTY',
            'Energy': '^CNXENERGY'
        }
        
        for sector, index in sector_indices.items():
            try:
                stock = yf.Ticker(index)
                hist = stock.history(period="1d")
                if not hist.empty:
                    change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                    sectors[sector] = {
                        'change_percent': float(change),  # Convert to native float
                        'volume': int(hist['Volume'].iloc[-1])  # Convert to native int
                    }
            except:
                pass
        
        return sectors
    
    def _store_fundamental_data(self, data: Dict):
        """Store fundamental data in database"""
        # Implementation would store in PostgreSQL/MongoDB
        pass
    
    async def start_realtime_monitoring(self):
        """Start real-time monitoring of all stocks"""
        logger.info("Starting real-time market monitoring...")
        
        while True:
            try:
                # Fetch data in batches
                batch_size = 50
                for i in range(0, len(self.nse_tickers), batch_size):
                    batch = self.nse_tickers[i:i+batch_size]
                    await self.fetch_bulk_realtime(batch)
                    await asyncio.sleep(1)  # Rate limiting
                
                # Update market breadth
                try:
                    breadth = self.fetch_market_breadth()
                    if breadth:  # Save to file and cache
                        self._save_to_file(self.market_breadth_file, breadth)
                        self._cache_set_dict("market:breadth", breadth, 60)
                except Exception as e:
                    logger.error(f"Error updating market breadth: {e}")
                
                # Update sector performance
                try:
                    sectors = self.fetch_sector_performance()
                    if sectors:  # Save to file and cache
                        self._save_to_file(self.sector_performance_file, sectors)
                        self._cache_set_dict("market:sectors", sectors, 300)
                        
                        # Save market summary to CSV
                        if breadth:
                            self._save_market_summary_to_csv(breadth, sectors)
                except Exception as e:
                    logger.error(f"Error updating sector performance: {e}")
                
                logger.info("Market data updated successfully")
                
                # Wait before next update (market hours consideration)
                await asyncio.sleep(30)  # Update every 30 seconds during market hours
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(60)
    
    def schedule_data_updates(self):
        """Schedule periodic data updates"""
        # Fundamental data update - daily after market close
        schedule.every().day.at("16:00").do(self.update_all_fundamentals)
        
        # Technical indicators - every hour during market hours
        schedule.every().hour.do(self.update_technical_indicators)
        
        # News and sentiment - every 30 minutes
        schedule.every(2).minutes.do(self.update_news_sentiment)
        schedule.every(2).minutes.do(self.fetch_and_save_news)
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def update_all_fundamentals(self):
        """Update fundamental data for all stocks"""
        logger.info("Updating fundamental data for all stocks...")
        
        for ticker in self.nse_tickers + self.bse_tickers:
            self.fetch_fundamental_data(ticker)
            time.sleep(0.5)  # Rate limiting
    
    def update_technical_indicators(self):
        """Update technical indicators for all stocks"""
        logger.info("Updating technical indicators...")
        
        all_indicators = {}
        for ticker in self.nse_tickers[:100]:  # Top 100 for frequent updates
            indicators = self.fetch_technical_indicators(ticker)
            if indicators:
                all_indicators[ticker] = indicators
                self._cache_set_dict(f"indicators:{ticker}", indicators, 3600)
        
        # Save all technical indicators to file
        if all_indicators:
            self._save_to_file(self.technical_indicators_file, all_indicators)
            logger.info(f"Saved technical indicators for {len(all_indicators)} stocks")
    
    def update_news_sentiment(self):
        """Update news and sentiment data"""
        # This would integrate with news APIs
        pass
    
    def get_stock_data_from_file(self, ticker: str) -> Optional[Dict]:
        """Get stock data from file if available"""
        file_path = f"data/realtime/stock_{ticker.replace('.', '_')}.json"
        return self._load_from_file(file_path)
    
    def get_market_data_from_file(self) -> Dict:
        """Get all market data from files"""
        return {
            'market_breadth': self._load_from_file(self.market_breadth_file),
            'sector_performance': self._load_from_file(self.sector_performance_file),
            'technical_indicators': self._load_from_file(self.technical_indicators_file)
        }
    
    def export_data_to_csv(self):
        """Export all data to CSV files for analysis"""
        try:
            # Export stock data
            if os.path.exists(self.stocks_csv):
                logger.info(f"Stock data already exported to {self.stocks_csv}")
            
            # Export market summary
            if os.path.exists(self.market_summary_csv):
                logger.info(f"Market summary already exported to {self.market_summary_csv}")
            
            # Export technical indicators to separate CSV
            indicators_data = self._load_from_file(self.technical_indicators_file)
            if indicators_data:
                indicators_list = []
                for ticker, indicators in indicators_data.items():
                    row = {'ticker': ticker}
                    row.update(indicators)
                    indicators_list.append(row)
                
                if indicators_list:
                    df = pd.DataFrame(indicators_list)
                    indicators_csv = os.path.join(self.data_dir, "technical_indicators.csv")
                    df.to_csv(indicators_csv, index=False)
                    logger.info(f"Technical indicators exported to {indicators_csv}")
            
            logger.info("Data export completed successfully")
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {e}")

if __name__ == "__main__":
    # Example usage - WITHOUT Redis, saving to files
    print("Starting Indian Market Data Ingestion with FILE STORAGE...")
    ingestion = IndianMarketDataIngestion(enable_redis=False)
    
    print(f"Data will be saved to: {ingestion.data_dir}")
    
    # Export existing data to CSV before starting
    ingestion.export_data_to_csv()
    
    # --- START OF FIX ---
    
    # 1. Create a separate thread for the scheduled tasks (news, fundamentals, etc.)
    # The scheduler runs in its own blocking loop, so it needs a thread.
    print("Starting scheduler thread for news and technicals...")
    scheduler_thread = threading.Thread(
        target=ingestion.schedule_data_updates,
        daemon=True  # Set as daemon so it exits when the main program exits
    )
    scheduler_thread.start()
    
    # 2. Run the async event loop for real-time price monitoring
    # This will run in the main thread.
    print("Starting real-time price monitoring loop...")
    try:
        asyncio.run(ingestion.start_realtime_monitoring())
    except KeyboardInterrupt:
        print("\nStopping data ingestion...")