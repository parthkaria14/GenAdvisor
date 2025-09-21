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
            "HDFCAMC.NS", "ICICIPRULI.NS", "ICICIGI.NS", "INDUSTOWER.NS", "JINDALSTEL.NS",
            "JUBLFOOD.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS", "UNITDSPR.NS",  # corrected from MCDOWELL
            "MOTHERSON.NS",  # replacing SAMRATPH / MOTHERSUMI
            "MUTHOOTFIN.NS", "NMDC.NS", "OFSS.NS",
            "PAGEIND.NS", "PEL.NS", "PETRONET.NS", "PIDILITIND.NS", "PNB.NS",
            "PGHH.NS", "SAIL.NS", "SBICARD.NS", "SIEMENS.NS", "SRF.NS",
            "TATACONSUM.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "UBL.NS",
            "VEDL.NS", "VOLTAS.NS", "ZEEL.NS"
        ]

        midcap_sample = [
            "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS",
            "ADANIENSOL.NS", "AJANTPHARM.NS", "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS",
            "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS"
        ]

        smallcap_sample = [
            "3MINDIA.NS", "AARTIDRUGS.NS", "AAVAS.NS", "ABSLAMC.NS", "AEGISLOG.NS",
            "AFFLE.NS", "AIAENG.NS", "AJMERA.NS", "AKZOINDIA.NS", "ALLCARGO.NS"
        ]

        
        tickers = nifty_50 + nifty_next_50 + midcap_sample + smallcap_sample
        return list(set(tickers))  # Remove duplicates
    
    def _load_bse_tickers(self) -> List[str]:
        """Load BSE-only tickers"""
        # Sample BSE tickers (would fetch from BSE API)
        return [
            "500325.BO", "532540.BO", "500034.BO", "500180.BO", "532174.BO",
            "500182.BO", "500312.BO", "500209.BO", "500696.BO", "500010.BO"
        ]
    
    def _load_commodity_tickers(self) -> List[str]:
        """Load commodity tickers from MCX"""
        return [
            "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER",
            "ZINC", "ALUMINUM", "LEAD", "NICKEL"
        ]
    
    async def fetch_realtime_price(self, ticker: str) -> Optional[Dict]:
        """Fetch real-time price for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current data
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            if current_price == 0:
                # Try to get from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            data = {
                'symbol': ticker,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'open': info.get('open', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            # Save to file storage
            self._save_to_file(f"data/realtime/stock_{ticker.replace('.', '_')}.json", data)
            
            # Also cache in Redis if available
            self._cache_set_dict(f"stock:{ticker}", data, 60)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    async def fetch_bulk_realtime(self, tickers: List[str]) -> List[Dict]:
        """Fetch real-time data for multiple tickers concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_realtime_price(ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            
            # Save all stock data to CSV
            if valid_results:
                self._save_stock_data_to_csv(valid_results)
            
            return valid_results
    
    def fetch_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Fetch detailed fundamental data"""
        try:
            stock = yf.Ticker(ticker)
            
            # Quarterly financials
            financials = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            # Key metrics
            info = stock.info
            
            fundamental_data = {
                'symbol': ticker,
                'timestamp': datetime.now().isoformat(),
                'revenue_growth': self._calculate_growth(financials, 'Total Revenue') if not financials.empty else None,
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'gross_margins': info.get('grossMargins'),
                'ebitda': info.get('ebitda'),
                'free_cashflow': info.get('freeCashflow'),
                'operating_cashflow': info.get('operatingCashflow'),
                'earnings_growth': info.get('earningsGrowth'),
                'revenue_per_share': info.get('revenuePerShare'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'enterprise_value': info.get('enterpriseValue'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda')
            }
            
            # Store in database
            self._store_fundamental_data(fundamental_data)
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return None
    
    def _calculate_growth(self, df: pd.DataFrame, metric: str) -> Optional[float]:
        """Calculate YoY growth for a metric"""
        try:
            if metric in df.index:
                values = df.loc[metric].values
                if len(values) >= 2:
                    return ((values[0] - values[1]) / abs(values[1])) * 100
        except:
            pass
        return None
    
    def fetch_technical_indicators(self, ticker: str, period: str = "1y") -> Dict:
        """Calculate technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {}
            
            # Calculate indicators
            indicators = {
                'symbol': ticker,
                'timestamp': datetime.now().isoformat(),
                'sma_20': hist['Close'].rolling(window=20).mean().iloc[-1],
                'sma_50': hist['Close'].rolling(window=50).mean().iloc[-1],
                'sma_200': hist['Close'].rolling(window=200).mean().iloc[-1],
                'ema_12': hist['Close'].ewm(span=12).mean().iloc[-1],
                'ema_26': hist['Close'].ewm(span=26).mean().iloc[-1],
                'rsi': self._calculate_rsi(hist['Close']),
                'macd': self._calculate_macd(hist['Close']),
                'bollinger_upper': self._calculate_bollinger(hist['Close'])[0],
                'bollinger_lower': self._calculate_bollinger(hist['Close'])[1],
                'atr': self._calculate_atr(hist),
                'volume_sma': hist['Volume'].rolling(window=20).mean().iloc[-1],
                'stochastic': self._calculate_stochastic(hist)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def _calculate_bollinger(self, prices: pd.Series, period: int = 20, std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper.iloc[-1], lower.iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k': k_percent.iloc[-1],
            'd': d_percent.iloc[-1]
        }
    
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
        schedule.every(30).minutes.do(self.update_news_sentiment)
        
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
    
    # Example usage - WITH Redis + files (if available)
    # print("Starting Indian Market Data Ingestion WITH Redis + file backup...")
    # ingestion = IndianMarketDataIngestion(enable_redis=True)
    
    print(f"Data will be saved to: {ingestion.data_dir}")
    print("Files created:")
    print(f"  - Individual stock data: data/realtime/stock_*.json")
    print(f"  - All stocks CSV: {ingestion.stocks_csv}")
    print(f"  - Market breadth: {ingestion.market_breadth_file}")
    print(f"  - Sector performance: {ingestion.sector_performance_file}")
    print(f"  - Technical indicators: {ingestion.technical_indicators_file}")
    print(f"  - Market summary CSV: {ingestion.market_summary_csv}")
    
    # Export existing data to CSV before starting
    ingestion.export_data_to_csv()
    
    # Run async event loop for real-time monitoring
    asyncio.run(ingestion.start_realtime_monitoring())