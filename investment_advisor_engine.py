# investment_advisor_engine_local.py
"""
Investment Advisor Engine with Local File Storage Integration
Works with data stored locally from data ingestion and RAG systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum

# Portfolio optimization
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp

# ML imports
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestmentStrategy(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    GROWTH = "growth"
    VALUE = "value"
    INCOME = "income"
    BALANCED = "balanced"

class RecommendationType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class PortfolioRecommendation:
    stocks: Dict[str, float]
    strategy: InvestmentStrategy
    expected_return: float
    risk_level: float
    sharpe_ratio: float
    recommendations: List[Dict]
    rebalancing_needed: bool
    
@dataclass
class StockRecommendation:
    symbol: str
    action: RecommendationType
    current_price: float
    target_price: float
    confidence: float
    reasons: List[str]
    risk_factors: List[str]
    time_horizon: str

class LocalInvestmentAdvisorEngine:
    """
    Investment advisory engine that works with knowledge graph from RAG system
    All data (prices, stocks, indicators, news) comes from the knowledge graph
    """
    
    def __init__(self, data_dir: str = "data/realtime", knowledge_graph=None, rag_system=None):
        self.data_dir = data_dir
        self.knowledge_graph = knowledge_graph  # Reference to RAG system's knowledge graph
        self.rag_system = rag_system  # Reference to RAG system for additional data access
        
        # Ensure data directory exists (fallback for file-based operations)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self.models_dir = "models"
        self.price_predictor = self._load_price_predictor()
        self.risk_model = self._load_risk_model()
        
        # Market parameters
        self.risk_free_rate = 0.065
        self.market_return = 0.12
        
        # Strategy parameters
        self.strategy_params = {
            InvestmentStrategy.CONSERVATIVE: {
                'equity': 0.3, 'debt': 0.6, 'gold': 0.1,
                'risk_tolerance': 0.05, 'min_return': 0.08
            },
            InvestmentStrategy.MODERATE: {
                'equity': 0.5, 'debt': 0.4, 'gold': 0.1,
                'risk_tolerance': 0.10, 'min_return': 0.10
            },
            InvestmentStrategy.AGGRESSIVE: {
                'equity': 0.7, 'debt': 0.2, 'gold': 0.1,
                'risk_tolerance': 0.15, 'min_return': 0.12
            },
            InvestmentStrategy.GROWTH: {
                'equity': 0.8, 'debt': 0.15, 'gold': 0.05,
                'risk_tolerance': 0.20, 'min_return': 0.15
            },
            InvestmentStrategy.VALUE: {
                'equity': 0.6, 'debt': 0.3, 'gold': 0.1,
                'risk_tolerance': 0.12, 'min_return': 0.11
            },
            InvestmentStrategy.INCOME: {
                'equity': 0.4, 'debt': 0.5, 'gold': 0.1,
                'risk_tolerance': 0.08, 'min_return': 0.09
            },
            InvestmentStrategy.BALANCED: {
                'equity': 0.5, 'debt': 0.4, 'gold': 0.1,
                'risk_tolerance': 0.10, 'min_return': 0.10
            }
        }
        
        # Cache for loaded data
        self.data_cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _load_price_predictor(self):
        """Load pre-trained price prediction model"""
        model_path = os.path.join(self.models_dir, 'price_predictor.pkl')
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except:
                pass
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _load_risk_model(self):
        """Load pre-trained risk assessment model"""
        model_path = os.path.join(self.models_dir, 'risk_model.pkl')
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except:
                pass
        return xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    def _load_from_file(self, file_path: str) -> Optional[Dict]:
        """Load data from JSON file with caching"""
        # Check cache first
        if file_path in self.data_cache:
            cache_time = self.cache_timestamp.get(file_path, 0)
            if (datetime.now().timestamp() - cache_time) < self.cache_ttl:
                return self.data_cache[file_path]
        
        # Load from file
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.data_cache[file_path] = data
                    self.cache_timestamp[file_path] = datetime.now().timestamp()
                    return data
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
        
        return None
    
    def _get_stock_data(self, symbol: str) -> Dict:
        """Get stock data from knowledge graph (primary source)"""
        # Primary: Query knowledge graph
        if self.knowledge_graph and self.knowledge_graph.has_node(symbol):
            node_data = self.knowledge_graph.nodes[symbol]
            if node_data.get('type') == 'stock':
                # Extract stock data from graph node attributes
                stock_data = {
                    'symbol': symbol,
                    'name': node_data.get('name', symbol),
                    'sector': node_data.get('sector', 'Unknown'),
                    'current_price': node_data.get('price', 0),
                    'change_percent': node_data.get('change_percent', 0),
                    'volume': node_data.get('volume', 0),
                    'market_cap': node_data.get('market_cap', 0),
                    'pe_ratio': node_data.get('pe_ratio'),
                    'pb_ratio': node_data.get('pb_ratio'),
                    'dividend_yield': node_data.get('dividend_yield'),
                    'beta': node_data.get('beta', 1.0),
                    'fifty_two_week_high': node_data.get('fifty_two_week_high'),
                    'fifty_two_week_low': node_data.get('fifty_two_week_low'),
                    'open': node_data.get('open'),
                    'high': node_data.get('high'),
                    'low': node_data.get('low'),
                    'close': node_data.get('close')
                }
                # Also get technical indicators from graph
                stock_data['rsi'] = node_data.get('rsi')
                stock_data['macd'] = node_data.get('macd')
                stock_data['sma_20'] = node_data.get('sma_20')
                stock_data['sma_50'] = node_data.get('sma_50')
                return stock_data
        
        # Fallback: Try RAG system's file_data if available
        if self.rag_system and hasattr(self.rag_system, 'file_data'):
            stock_data = self.rag_system.file_data.get('stocks', {}).get(symbol)
            if stock_data:
                return stock_data
        
        # Last resort: Read from files (legacy support)
        stock_file = os.path.join(self.data_dir, f"stock_{symbol.replace('.', '_')}.json")
        data = self._load_from_file(stock_file)
        if data:
            return data
        
        csv_file = os.path.join(self.data_dir, "stocks.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                stock_row = df[df['symbol'] == symbol]
                if not stock_row.empty:
                    return stock_row.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Error reading CSV for {symbol}: {e}")
        
        return {}
    
    def _get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators from knowledge graph (primary source)"""
        # Primary: Query knowledge graph
        if self.knowledge_graph and self.knowledge_graph.has_node(symbol):
            node_data = self.knowledge_graph.nodes[symbol]
            if node_data.get('type') == 'stock':
                indicators = {
                    'rsi': node_data.get('rsi'),
                    'macd': node_data.get('macd'),
                    'sma_20': node_data.get('sma_20'),
                    'sma_50': node_data.get('sma_50'),
                    'bollinger_upper': node_data.get('bollinger_upper'),
                    'bollinger_lower': node_data.get('bollinger_lower')
                }
                # Filter out None values
                return {k: v for k, v in indicators.items() if v is not None}
        
        # Fallback: Try RAG system's file_data
        if self.rag_system and hasattr(self.rag_system, 'file_data'):
            indicators = self.rag_system.file_data.get('technical_indicators', {}).get(symbol, {})
            if indicators:
                return indicators
        
        # Last resort: Read from files (legacy support)
        indicators_file = os.path.join(self.data_dir, "technical_indicators.json")
        all_indicators = self._load_from_file(indicators_file)
        if all_indicators:
            return all_indicators.get(symbol, {})
        
        return {}
    
    def _get_market_breadth(self) -> Dict:
        """Get market breadth data from knowledge graph (primary source)"""
        # Primary: Query knowledge graph
        if self.knowledge_graph and self.knowledge_graph.has_node('market_breadth'):
            node_data = self.knowledge_graph.nodes['market_breadth']
            if node_data.get('type') == 'market_indicator':
                return {
                    'advances': node_data.get('advances', 0),
                    'declines': node_data.get('declines', 0),
                    'market_sentiment': node_data.get('sentiment', 'neutral'),
                    'timestamp': node_data.get('timestamp', ''),
                    'advance_decline_ratio': (
                        node_data.get('advances', 0) / max(node_data.get('declines', 1), 1)
                    )
                }
        
        # Fallback: Try RAG system's file_data
        if self.rag_system and hasattr(self.rag_system, 'file_data'):
            breadth_data = self.rag_system.file_data.get('market_breadth', {})
            if breadth_data:
                return breadth_data
        
        # Last resort: Read from files (legacy support)
        breadth_file = os.path.join(self.data_dir, "market_breadth.json")
        return self._load_from_file(breadth_file) or {}
    
    def _get_sector_performance(self) -> Dict:
        """Get sector performance from knowledge graph (primary source)"""
        sector_data = {}
        
        # Primary: Query knowledge graph for all sector nodes
        if self.knowledge_graph:
            # Get all sector nodes and their related stocks
            for node in self.knowledge_graph.nodes():
                node_data = self.knowledge_graph.nodes[node]
                if node_data.get('type') == 'sector':
                    # Get stocks in this sector
                    sector_stocks = [
                        neighbor for neighbor in self.knowledge_graph.neighbors(node)
                        if self.knowledge_graph.nodes[neighbor].get('type') == 'stock'
                    ]
                    
                    # Calculate sector aggregate metrics
                    total_change = 0
                    total_volume = 0
                    stock_count = 0
                    
                    for stock_symbol in sector_stocks:
                        stock_data = self.knowledge_graph.nodes[stock_symbol]
                        total_change += stock_data.get('change_percent', 0)
                        total_volume += stock_data.get('volume', 0)
                        stock_count += 1
                    
                    if stock_count > 0:
                        sector_data[node] = {
                            'average_change': total_change / stock_count,
                            'total_volume': total_volume,
                            'stock_count': stock_count,
                            'stocks': sector_stocks[:10]  # Top 10 stocks
                        }
        
        # Fallback: Try RAG system's file_data
        if not sector_data and self.rag_system and hasattr(self.rag_system, 'file_data'):
            sector_data = self.rag_system.file_data.get('sector_performance', {})
        
        # Last resort: Read from files (legacy support)
        if not sector_data:
            sectors_file = os.path.join(self.data_dir, "sector_performance.json")
            sector_data = self._load_from_file(sectors_file) or {}
        
        return sector_data
    
    async def analyze_stock(self, symbol: str) -> Dict:
        """Comprehensive analysis of a single stock"""
        try:
            # Get stock data from local storage
            stock_data = self._get_stock_data(symbol)
            if not stock_data:
                return {
                    "success": False,
                    "error": f"No data available for {symbol}"
                }
            
            # Technical analysis
            technical_indicators = self._get_technical_indicators(symbol)
            technical_signals = self._analyze_technical(technical_indicators)
            
            # Fundamental analysis
            fundamental_score = self._analyze_fundamentals(stock_data)
            
            # Sentiment analysis (simplified without Redis)
            sentiment_score = self._get_sentiment_score(symbol)
            
            # Price prediction
            predicted_price = self._predict_price(symbol, stock_data, technical_indicators)
            current_price = float(stock_data.get('current_price', 0))
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price if current_price > 0 else 0
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                expected_return,
                technical_signals,
                fundamental_score,
                sentiment_score
            )
            
            # Identify reasons and risks
            reasons = self._identify_buy_reasons(
                symbol, expected_return, technical_signals, 
                fundamental_score, sentiment_score, stock_data
            )
            
            risk_factors = self._identify_risk_factors(
                symbol, stock_data, technical_indicators
            )
            
            return {
                "success": True,
                "symbol": symbol,
                "action": recommendation.value,
                "current_price": current_price,
                "target_price": predicted_price,
                "expected_return": expected_return * 100,
                "confidence": self._calculate_confidence(
                    technical_signals, fundamental_score, sentiment_score
                ),
                "reasons": reasons,
                "risk_factors": risk_factors,
                "time_horizon": "30 days",
                "technical_indicators": technical_indicators,
                "fundamental_score": fundamental_score,
                "sentiment_score": sentiment_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_technical(self, indicators: Dict) -> Dict:
        """Perform technical analysis"""
        if not indicators:
            return {'signal': 'neutral', 'strength': 0.5}
        
        signals = []
        
        # RSI signal
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            signals.append(1)  # Oversold - Buy
        elif rsi > 70:
            signals.append(-1)  # Overbought - Sell
        else:
            signals.append(0)
        
        # MACD signal
        macd = indicators.get('macd', {})
        if isinstance(macd, dict):
            if macd.get('histogram', 0) > 0:
                signals.append(1)
            elif macd.get('histogram', 0) < 0:
                signals.append(-1)
            else:
                signals.append(0)
        
        # Moving average signal
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        
        if sma_20 > sma_50 and sma_20 > 0:
            signals.append(1)  # Bullish
        elif sma_20 < sma_50 and sma_20 > 0:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)
        
        # Aggregate signals
        avg_signal = np.mean(signals) if signals else 0
        
        if avg_signal > 0.3:
            return {'signal': 'buy', 'strength': abs(avg_signal)}
        elif avg_signal < -0.3:
            return {'signal': 'sell', 'strength': abs(avg_signal)}
        else:
            return {'signal': 'neutral', 'strength': 0.5}
    
    def _analyze_fundamentals(self, stock_data: Dict) -> float:
        """Analyze fundamental metrics"""
        score = 0.5  # Neutral baseline
        
        # PE Ratio analysis
        pe = stock_data.get('pe_ratio')
        if pe and 0 < pe < 15:
            score += 0.1  # Undervalued
        elif pe and pe > 30:
            score -= 0.1  # Overvalued
        
        # PB Ratio
        pb = stock_data.get('pb_ratio')
        if pb and 0 < pb < 1:
            score += 0.1  # Below book value
        elif pb and pb > 3:
            score -= 0.1  # Expensive
        
        # Dividend Yield
        div_yield = stock_data.get('dividend_yield')
        if div_yield and div_yield > 0.03:
            score += 0.05  # Good dividend
        
        # Market Cap (prefer large cap for stability)
        market_cap = stock_data.get('market_cap', 0)
        if market_cap > 1000000000000:  # > 1 Trillion INR
            score += 0.05  # Large cap premium
        
        # Beta (volatility)
        beta = stock_data.get('beta', 1.0)
        if 0.8 < beta < 1.2:
            score += 0.05  # Moderate volatility
        elif beta > 1.5:
            score -= 0.05  # High volatility
        
        return min(max(score, 0), 1)  # Clamp between 0 and 1
    
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score from knowledge graph news nodes (primary source)"""
        sentiment_scores = []
        
        # Primary: Query knowledge graph for news nodes connected to this stock
        if self.knowledge_graph and self.knowledge_graph.has_node(symbol):
            # Find news nodes connected to this stock
            for neighbor in self.knowledge_graph.neighbors(symbol):
                neighbor_data = self.knowledge_graph.nodes[neighbor]
                if neighbor_data.get('type') == 'news':
                    sentiment = neighbor_data.get('sentiment', 'neutral')
                    # Convert sentiment string to numeric score
                    sentiment_map = {
                        'very_positive': 1.0,
                        'positive': 0.5,
                        'neutral': 0.0,
                        'negative': -0.5,
                        'very_negative': -1.0
                    }
                    score = sentiment_map.get(sentiment.lower(), 0.0)
                    sentiment_scores.append(score)
        
        # Fallback: Try RAG system's file_data
        if not sentiment_scores and self.rag_system and hasattr(self.rag_system, 'file_data'):
            # Search news data for mentions of this symbol
            news_data = self.rag_system.file_data.get('news_data', [])
            for news_item in news_data:
                # Simple check if symbol is mentioned
                content = f"{news_item.get('title', '')} {news_item.get('content', '')}"
                if symbol.replace('.NS', '').replace('.BO', '') in content:
                    sentiment = news_item.get('sentiment', 'neutral')
                    sentiment_map = {
                        'positive': 0.3,
                        'neutral': 0.0,
                        'negative': -0.3
                    }
                    sentiment_scores.append(sentiment_map.get(sentiment.lower(), 0.0))
        
        # Calculate average sentiment
        if sentiment_scores:
            return sum(sentiment_scores) / len(sentiment_scores)
        
        return 0.0  # Neutral if no news data
    
    def _predict_price(self, symbol: str, stock_data: Dict, technical_indicators: Dict) -> float:
        """Predict future stock price"""
        try:
            current_price = float(stock_data.get('current_price', 0))
            
            if current_price == 0:
                return 0
            
            # Calculate momentum
            momentum = self._calculate_momentum(stock_data)
            
            # Get fundamental score
            fundamental_score = self._analyze_fundamentals(stock_data)
            
            # Technical trend
            tech_trend = 0
            if technical_indicators:
                rsi = technical_indicators.get('rsi', 50)
                tech_trend = (50 - rsi) / 100  # Convert RSI to trend factor
            
            # Expected return calculation
            base_return = 0.01  # 1% monthly base
            momentum_factor = momentum * 0.02  # Up to 2% from momentum
            fundamental_factor = (fundamental_score - 0.5) * 0.03  # Up to ±1.5% from fundamentals
            tech_factor = tech_trend * 0.01  # Up to ±1% from technical
            
            expected_return = base_return + momentum_factor + fundamental_factor + tech_factor
            predicted_price = current_price * (1 + expected_return)
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return 0
    
    def _calculate_momentum(self, stock_data: Dict) -> float:
        """Calculate price momentum"""
        current = float(stock_data.get('current_price', 0))
        week_low = float(stock_data.get('fifty_two_week_low', current))
        week_high = float(stock_data.get('fifty_two_week_high', current))
        
        if week_high > week_low:
            momentum = (current - week_low) / (week_high - week_low)
            return min(max(momentum, -1), 1)
        return 0
    
    def _generate_recommendation(
        self,
        expected_return: float,
        technical_signals: Dict,
        fundamental_score: float,
        sentiment_score: float
    ) -> RecommendationType:
        """Generate buy/sell/hold recommendation"""
        
        # Weight different factors
        weights = {
            'return': 0.3,
            'technical': 0.25,
            'fundamental': 0.25,
            'sentiment': 0.2
        }
        
        # Convert signals to scores
        tech_score = 1 if technical_signals['signal'] == 'buy' else (
            -1 if technical_signals['signal'] == 'sell' else 0
        )
        
        # Calculate weighted score
        score = (
            expected_return * 10 * weights['return'] +
            tech_score * weights['technical'] +
            (fundamental_score - 0.5) * 2 * weights['fundamental'] +
            sentiment_score * weights['sentiment']
        )
        
        # Generate recommendation based on score
        if score > 0.5:
            return RecommendationType.STRONG_BUY
        elif score > 0.2:
            return RecommendationType.BUY
        elif score < -0.5:
            return RecommendationType.STRONG_SELL
        elif score < -0.2:
            return RecommendationType.SELL
        else:
            return RecommendationType.HOLD
    
    def _identify_buy_reasons(
        self,
        symbol: str,
        expected_return: float,
        technical_signals: Dict,
        fundamental_score: float,
        sentiment_score: float,
        stock_data: Dict
    ) -> List[str]:
        """Identify reasons for recommendation"""
        reasons = []
        
        if expected_return > 0.1:
            reasons.append(f"Expected return of {expected_return*100:.1f}% in next 30 days")
        
        if technical_signals['signal'] == 'buy':
            reasons.append("Positive technical indicators (RSI, MACD, Moving Averages)")
        
        if fundamental_score > 0.7:
            reasons.append("Strong fundamental metrics (PE, PB, ROE)")
        
        if sentiment_score > 0.3:
            reasons.append("Positive market sentiment and news flow")
        
        div_yield = stock_data.get('dividend_yield', 0)
        if div_yield and div_yield > 0.03:
            reasons.append(f"Attractive dividend yield of {div_yield*100:.1f}%")
        
        pe = stock_data.get('pe_ratio')
        if pe and 10 < pe < 20:
            reasons.append(f"Reasonable valuation with PE ratio of {pe:.1f}")
        
        return reasons if reasons else ["Based on overall market conditions"]
    
    def _identify_risk_factors(
        self,
        symbol: str,
        stock_data: Dict,
        technical_indicators: Dict
    ) -> List[str]:
        """Identify risk factors"""
        risks = []
        
        # Volatility risk
        beta = stock_data.get('beta', 1)
        if beta > 1.5:
            risks.append(f"High volatility (Beta: {beta:.2f})")
        
        # Valuation risk
        pe = stock_data.get('pe_ratio')
        if pe and pe > 30:
            risks.append(f"High valuation (PE: {pe:.1f})")
        
        # Technical risk
        if technical_indicators:
            rsi = technical_indicators.get('rsi', 50)
            if rsi > 70:
                risks.append("Overbought conditions (RSI > 70)")
            elif rsi < 30:
                risks.append("Oversold conditions (RSI < 30)")
        
        # Sector risk
        sector = stock_data.get('sector', '')
        if sector in ['Energy', 'Commodities']:
            risks.append(f"Sector-specific risks in {sector}")
        
        # Market risk
        market_breadth = self._get_market_breadth()
        if market_breadth.get('market_sentiment') == 'bearish':
            risks.append("Overall bearish market sentiment")
        
        risks.append("General market volatility and economic conditions")
        
        return risks
    
    def _calculate_confidence(
        self,
        technical_signals: Dict,
        fundamental_score: float,
        sentiment_score: float
    ) -> float:
        """Calculate confidence score for recommendation"""
        
        # Base confidence
        confidence = 0.5
        
        # Technical signal strength
        confidence += technical_signals.get('strength', 0.5) * 0.2
        
        # Fundamental strength
        confidence += abs(fundamental_score - 0.5) * 0.3
        
        # Sentiment clarity
        confidence += abs(sentiment_score) * 0.2
        
        return min(max(confidence, 0), 1)
    
    async def optimize_portfolio(
        self,
        budget: float,
        strategy: InvestmentStrategy,
        existing_portfolio: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict] = None
    ) -> PortfolioRecommendation:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        
        # Get strategy parameters
        params = self.strategy_params[strategy]
        
        # Get universe of stocks based on strategy
        stock_universe = await self._get_stock_universe(strategy)
        
        # Fetch return and risk data
        returns_data, risk_matrix = await self._get_returns_and_risk(stock_universe)
        
        if len(returns_data) == 0:
            raise ValueError("Insufficient data for portfolio optimization")
        
        # Optimize allocation
        weights = self._optimize_allocation(
            returns_data,
            risk_matrix,
            params['risk_tolerance'],
            params['min_return'],
            constraints
        )
        
        # Create stock allocation
        allocation = {}
        for i, symbol in enumerate(stock_universe):
            if weights[i] > 0.01:  # Only include if > 1% allocation
                allocation[symbol] = weights[i]
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * returns_data)
        portfolio_risk = np.sqrt(weights @ risk_matrix @ weights.T)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Generate specific recommendations
        recommendations = await self._generate_portfolio_recommendations(
            allocation,
            existing_portfolio,
            budget
        )
        
        # Check if rebalancing needed
        rebalancing_needed = self._check_rebalancing_needed(
            existing_portfolio,
            allocation
        )
        
        return PortfolioRecommendation(
            stocks=allocation,
            strategy=strategy,
            expected_return=portfolio_return,
            risk_level=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            recommendations=recommendations,
            rebalancing_needed=rebalancing_needed
        )
    
    async def _get_stock_universe(self, strategy: InvestmentStrategy) -> List[str]:
        """Get relevant stocks based on strategy from knowledge graph (primary source)"""
        available_stocks = []
        
        # Primary: Query knowledge graph for all stock nodes
        if self.knowledge_graph:
            for node in self.knowledge_graph.nodes():
                node_data = self.knowledge_graph.nodes[node]
                if node_data.get('type') == 'stock':
                    available_stocks.append(node)
        elif self.rag_system and hasattr(self.rag_system, 'file_data'):
            # Fallback: Get from RAG system's file_data
            available_stocks = list(self.rag_system.file_data.get('stocks', {}).keys())
        else:
            # Last resort: Load from CSV
            csv_file = os.path.join(self.data_dir, "stocks.csv")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    available_stocks = df['symbol'].tolist()
                except Exception as e:
                    logger.warning(f"Error reading stocks CSV: {e}")
            
            # If no CSV, check individual files
            if not available_stocks:
                stock_files = [f for f in os.listdir(self.data_dir) if f.startswith('stock_')]
                for file in stock_files[:20]:
                    symbol = file.replace('stock_', '').replace('.json', '').replace('_', '.')
                    available_stocks.append(symbol)
        
        # Base universe - prioritize large caps
        base_universe = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "AXISBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS"
        ]
        
        # Filter to only available stocks from knowledge graph
        universe = [s for s in base_universe if s in available_stocks]
        
        # Add strategy-specific stocks if available in knowledge graph
        if strategy == InvestmentStrategy.GROWTH:
            growth_stocks = ["ADANIENT.NS", "ADANIGREEN.NS"]
            universe.extend([s for s in growth_stocks if s in available_stocks and s not in universe])
        elif strategy == InvestmentStrategy.VALUE:
            value_stocks = ["COALINDIA.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS"]
            universe.extend([s for s in value_stocks if s in available_stocks and s not in universe])
        elif strategy == InvestmentStrategy.INCOME:
            income_stocks = ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"]
            universe.extend([s for s in income_stocks if s in available_stocks and s not in universe])
        
        return universe[:15] if universe else available_stocks[:15]  # Limit to 15 stocks for optimization
    
    async def _get_returns_and_risk(
        self,
        stock_universe: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate expected returns and covariance matrix from local data"""
        
        returns = []
        
        # Fetch predicted returns for each stock
        for symbol in stock_universe:
            stock_data = self._get_stock_data(symbol)
            technical_indicators = self._get_technical_indicators(symbol)
            
            if stock_data:
                predicted_price = self._predict_price(symbol, stock_data, technical_indicators)
                current_price = float(stock_data.get('current_price', 0))
                
                if current_price > 0:
                    expected_return = (predicted_price - current_price) / current_price
                    returns.append(expected_return)
                else:
                    returns.append(0)
            else:
                returns.append(0)
        
        returns = np.array(returns)
        
        # Create correlation matrix (simplified - in production, use historical data)
        n_stocks = len(stock_universe)
        correlation = np.eye(n_stocks) * 0.5  # Start with 0.5 correlation
        np.fill_diagonal(correlation, 1.0)  # Diagonal = 1
        
        # Adjust correlation based on sectors
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                stock_i = self._get_stock_data(stock_universe[i])
                stock_j = self._get_stock_data(stock_universe[j])
                
                if stock_i and stock_j:
                    # Same sector = higher correlation
                    if stock_i.get('sector') == stock_j.get('sector'):
                        correlation[i, j] = correlation[j, i] = 0.7
                    else:
                        correlation[i, j] = correlation[j, i] = 0.3
        
        # Convert to covariance
        std_devs = np.array([0.2] * n_stocks)  # Assume 20% volatility
        
        # Adjust volatility based on beta
        for i, symbol in enumerate(stock_universe):
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                beta = stock_data.get('beta', 1.0)
                std_devs[i] = 0.2 * beta  # Scale volatility by beta
        
        covariance = correlation * np.outer(std_devs, std_devs)
        
        return returns, covariance
    
    def _optimize_allocation(
        self,
        returns: np.ndarray,
        risk_matrix: np.ndarray,
        risk_tolerance: float,
        min_return: float,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Perform portfolio optimization using cvxpy"""
        
        n_assets = len(returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Expected portfolio return
        portfolio_return = returns @ weights
        
        # Portfolio risk (variance)
        portfolio_risk = cp.quad_form(weights, risk_matrix)
        
        # Objective: Maximize Sharpe ratio (approximated)
        objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_risk)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # No short selling
            weights <= 0.3,  # Max 30% in single stock
            portfolio_return >= min_return  # Minimum return constraint
        ]
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.SCS)
            
            if problem.status == cp.OPTIMAL:
                return weights.value
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
    
    async def _generate_portfolio_recommendations(
        self,
        new_allocation: Dict[str, float],
        existing_portfolio: Optional[Dict[str, float]],
        budget: float
    ) -> List[Dict]:
        """Generate specific buy/sell recommendations for portfolio"""
        
        recommendations = []
        
        if not existing_portfolio:
            # New portfolio - all buys
            for symbol, weight in new_allocation.items():
                amount = budget * weight
                stock_data = self._get_stock_data(symbol)
                current_price = float(stock_data.get('current_price', 0))
                
                if current_price > 0:
                    shares = int(amount / current_price)
                    recommendations.append({
                        'action': 'BUY',
                        'symbol': symbol,
                        'shares': shares,
                        'amount': shares * current_price,
                        'allocation': weight * 100,
                        'reason': 'Initial portfolio allocation'
                    })
        else:
            # Rebalancing existing portfolio
            for symbol, target_weight in new_allocation.items():
                current_weight = existing_portfolio.get(symbol, 0)
                diff = target_weight - current_weight
                
                if abs(diff) > 0.02:  # Only rebalance if difference > 2%
                    stock_data = self._get_stock_data(symbol)
                    current_price = float(stock_data.get('current_price', 0))
                    
                    if current_price > 0:
                        amount = abs(budget * diff)
                        shares = int(amount / current_price)
                        
                        recommendations.append({
                            'action': 'BUY' if diff > 0 else 'SELL',
                            'symbol': symbol,
                            'shares': shares,
                            'amount': shares * current_price,
                            'current_allocation': current_weight * 100,
                            'target_allocation': target_weight * 100,
                            'reason': 'Portfolio rebalancing'
                        })
        
        return recommendations
    
    def _check_rebalancing_needed(
        self,
        existing_portfolio: Optional[Dict[str, float]],
        target_allocation: Dict[str, float]
    ) -> bool:
        """Check if portfolio rebalancing is needed"""
        
        if not existing_portfolio:
            return False
        
        # Check if any allocation differs by more than 5%
        for symbol, target_weight in target_allocation.items():
            current_weight = existing_portfolio.get(symbol, 0)
            if abs(target_weight - current_weight) > 0.05:
                return True
        
        return False
    
    async def analyze_risk(
        self,
        portfolio: Dict[str, float],
        time_horizon: int = 30
    ) -> Dict:
        """Comprehensive risk analysis for portfolio"""
        
        # Calculate Value at Risk (VaR)
        var_95 = self._calculate_var(portfolio, confidence=0.95)
        var_99 = self._calculate_var(portfolio, confidence=0.99)
        
        # Calculate Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio)
        
        # Calculate Beta
        portfolio_beta = self._calculate_portfolio_beta(portfolio)
        
        # Stress testing
        stress_results = await self._stress_test_portfolio(portfolio)
        
        # Correlation analysis
        correlation_risk = self._analyze_correlation_risk(portfolio)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'portfolio_beta': portfolio_beta,
            'stress_test': stress_results,
            'correlation_risk': correlation_risk,
            'risk_rating': self._calculate_risk_rating(var_95, portfolio_beta),
            'recommendations': self._generate_risk_recommendations(
                var_95, portfolio_beta, correlation_risk
            )
        }
    
    def _calculate_var(
        self,
        portfolio: Dict[str, float],
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk"""
        
        # Get portfolio volatility from constituent stocks
        portfolio_volatility = 0
        
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                beta = stock_data.get('beta', 1.0)
                stock_volatility = 0.2 * beta  # Base 20% scaled by beta
                portfolio_volatility += (weight * stock_volatility) ** 2
        
        portfolio_volatility = np.sqrt(portfolio_volatility)
        
        # Daily VaR
        z_score = stats.norm.ppf(1 - confidence)
        daily_var = portfolio_volatility / np.sqrt(252)  # Convert to daily
        
        return abs(z_score * daily_var)
    
    def _calculate_max_drawdown(self, portfolio: Dict[str, float]) -> float:
        """Calculate maximum drawdown"""
        # Simplified calculation based on portfolio composition
        avg_beta = 0
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                beta = stock_data.get('beta', 1.0)
                avg_beta += weight * beta
        
        # Estimate max drawdown based on portfolio beta
        if avg_beta > 1.5:
            return 0.25  # 25% max drawdown for high beta
        elif avg_beta > 1.2:
            return 0.20  # 20% for moderate-high beta
        elif avg_beta > 0.8:
            return 0.15  # 15% for moderate beta
        else:
            return 0.10  # 10% for low beta
    
    def _calculate_portfolio_beta(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio beta"""
        
        weighted_beta = 0
        
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                beta = stock_data.get('beta', 1.0)
                weighted_beta += weight * beta
        
        return weighted_beta
    
    async def _stress_test_portfolio(
        self,
        portfolio: Dict[str, float]
    ) -> Dict:
        """Perform stress testing on portfolio"""
        
        scenarios = {
            'market_crash': -0.20,  # 20% market drop
            'sector_crisis': -0.30,  # 30% sector drop
            'black_swan': -0.40,  # 40% extreme event
            'recession': -0.15,  # 15% recession scenario
            'rate_hike': -0.10,  # 10% interest rate shock
        }
        
        results = {}
        
        for scenario, impact in scenarios.items():
            portfolio_impact = 0
            
            for symbol, weight in portfolio.items():
                stock_data = self._get_stock_data(symbol)
                if stock_data:
                    beta = stock_data.get('beta', 1.0)
                    
                    # Stock impact = market impact * beta
                    stock_impact = impact * beta
                    
                    # Sector-specific adjustments
                    sector = stock_data.get('sector', '')
                    if scenario == 'sector_crisis' and sector in ['Banking', 'Financial Services']:
                        stock_impact *= 1.5  # Extra impact on financials
                    elif scenario == 'rate_hike' and sector in ['Real Estate', 'Infrastructure']:
                        stock_impact *= 1.3  # Rate sensitive sectors
                    
                    portfolio_impact += weight * stock_impact
            
            results[scenario] = {
                'impact': portfolio_impact,
                'description': f"Portfolio would lose {abs(portfolio_impact)*100:.1f}% in {scenario}",
                'severity': 'high' if abs(portfolio_impact) > 0.25 else 'moderate' if abs(portfolio_impact) > 0.15 else 'low'
            }
        
        return results
    
    def _analyze_correlation_risk(self, portfolio: Dict[str, float]) -> str:
        """Analyze correlation risk in portfolio"""
        
        # Check sector concentration
        sectors = {}
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                sector = stock_data.get('sector', 'Unknown')
                sectors[sector] = sectors.get(sector, 0) + weight
        
        # Find dominant sector
        max_sector_weight = max(sectors.values()) if sectors else 0
        
        # Check market cap concentration
        market_cap_distribution = {
            'large': 0,
            'mid': 0,
            'small': 0
        }
        
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            if stock_data:
                market_cap = stock_data.get('market_cap', 0)
                if market_cap > 1000000000000:  # > 1T INR
                    market_cap_distribution['large'] += weight
                elif market_cap > 100000000000:  # > 100B INR
                    market_cap_distribution['mid'] += weight
                else:
                    market_cap_distribution['small'] += weight
        
        # Determine risk level
        if max_sector_weight > 0.4 or market_cap_distribution['small'] > 0.3:
            return 'high'
        elif max_sector_weight > 0.25 or market_cap_distribution['small'] > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_rating(self, var: float, beta: float) -> str:
        """Calculate overall risk rating"""
        
        risk_score = (var * 10) + (beta - 1) * 0.5
        
        if risk_score < 0.5:
            return 'low'
        elif risk_score < 1.0:
            return 'medium'
        elif risk_score < 1.5:
            return 'high'
        else:
            return 'very_high'
    
    def _generate_risk_recommendations(
        self,
        var: float,
        beta: float,
        correlation_risk: str
    ) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if var > 0.05:
            recommendations.append(f"High VaR of {var*100:.1f}% - Consider reducing position sizes or adding hedges")
        
        if beta > 1.5:
            recommendations.append("Portfolio has high market sensitivity (Beta > 1.5) - Add defensive stocks or gold ETFs")
        elif beta < 0.8:
            recommendations.append("Low beta portfolio may underperform in bull markets - Consider adding growth stocks")
        
        if correlation_risk == 'high':
            recommendations.append("High concentration risk detected - Diversify across sectors and market caps")
        elif correlation_risk == 'medium':
            recommendations.append("Moderate concentration - Consider broadening sector allocation")
        
        # Market-specific recommendations
        market_breadth = self._get_market_breadth()
        if market_breadth.get('market_sentiment') == 'bearish':
            recommendations.append("Bearish market conditions - Consider increasing cash allocation or defensive positions")
        
        recommendations.append("Regular monthly rebalancing recommended to maintain target allocation")
        recommendations.append("Set stop-loss orders at 10-15% below purchase price for risk management")
        
        return recommendations
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary from knowledge graph (primary source)"""
        try:
            market_breadth = self._get_market_breadth()
            sector_performance = self._get_sector_performance()
            
            # Find top gainers and losers from knowledge graph
            top_gainers = []
            top_losers = []
            most_active = []
            
            # Primary: Query knowledge graph for all stock nodes
            if self.knowledge_graph:
                for node in self.knowledge_graph.nodes():
                    node_data = self.knowledge_graph.nodes[node]
                    if node_data.get('type') == 'stock':
                        symbol = node
                        price = node_data.get('price', 0)
                        change_pct = node_data.get('change_percent', 0)
                        volume = node_data.get('volume', 0)
                        
                        if price > 0:  # Only include stocks with valid price
                            stock_info = {
                                'symbol': symbol,
                                'price': price,
                                'change_percent': change_pct,
                                'volume': volume,
                                'name': node_data.get('name', symbol),
                                'sector': node_data.get('sector', 'Unknown')
                            }
                            
                            if change_pct > 0:
                                top_gainers.append(stock_info)
                            else:
                                top_losers.append(stock_info)
                            
                            most_active.append(stock_info)
            else:
                # Fallback: Use RAG system's file_data
                if self.rag_system and hasattr(self.rag_system, 'file_data'):
                    stocks = self.rag_system.file_data.get('stocks', {})
                    for symbol, stock_data in stocks.items():
                        price = stock_data.get('current_price', 0)
                        change_pct = stock_data.get('change_percent', 0)
                        volume = stock_data.get('volume', 0)
                        
                        if price > 0:
                            stock_info = {
                                'symbol': symbol,
                                'price': price,
                                'change_percent': change_pct,
                                'volume': volume,
                                'name': stock_data.get('name', symbol),
                                'sector': stock_data.get('sector', 'Unknown')
                            }
                            
                            if change_pct > 0:
                                top_gainers.append(stock_info)
                            else:
                                top_losers.append(stock_info)
                            
                            most_active.append(stock_info)
                else:
                    # Last resort: Read from files
                    stock_files = [f for f in os.listdir(self.data_dir) if f.startswith('stock_')]
                    for file in stock_files[:100]:
                        file_path = os.path.join(self.data_dir, file)
                        stock_data = self._load_from_file(file_path)
                        
                        if stock_data:
                            symbol = stock_data.get('symbol', '')
                            price = stock_data.get('current_price', 0)
                            volume = stock_data.get('volume', 0)
                            open_price = stock_data.get('open', price)
                            
                            if open_price > 0:
                                change_pct = ((price - open_price) / open_price) * 100
                                stock_info = {
                                    'symbol': symbol,
                                    'price': price,
                                    'change_percent': change_pct,
                                    'volume': volume
                                }
                                
                                if change_pct > 0:
                                    top_gainers.append(stock_info)
                                else:
                                    top_losers.append(stock_info)
                                
                                most_active.append(stock_info)
            
            # Sort and limit results
            top_gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            top_losers.sort(key=lambda x: x['change_percent'])
            most_active.sort(key=lambda x: x['volume'], reverse=True)
            
            return {
                'market_breadth': market_breadth,
                'sector_performance': sector_performance,
                'top_gainers': top_gainers[:10],
                'top_losers': top_losers[:10],
                'most_active': most_active[:10],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_engine():
        print("Testing Local Investment Advisor Engine...")
        print("=" * 60)
        
        # Initialize engine
        engine = LocalInvestmentAdvisorEngine()
        
        # Test 1: Analyze a stock
        print("\n1. Analyzing RELIANCE.NS...")
        result = await engine.analyze_stock("RELIANCE.NS")
        if result['success']:
            print(f"   Action: {result['action']}")
            print(f"   Current Price: ₹{result['current_price']:.2f}")
            print(f"   Target Price: ₹{result['target_price']:.2f}")
            print(f"   Expected Return: {result['expected_return']:.1f}%")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print(f"   Reasons: {result['reasons'][:2]}")
        else:
            print(f"   Error: {result.get('error')}")
        
        # Test 2: Portfolio optimization
        print("\n2. Optimizing portfolio...")
        portfolio = await engine.optimize_portfolio(
            budget=1000000,  # 10 lakhs
            strategy=InvestmentStrategy.MODERATE
        )
        print(f"   Strategy: {portfolio.strategy.value}")
        print(f"   Expected Return: {portfolio.expected_return*100:.1f}%")
        print(f"   Risk Level: {portfolio.risk_level*100:.1f}%")
        print(f"   Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
        print(f"   Top Allocations:")
        for symbol, weight in list(portfolio.stocks.items())[:5]:
            print(f"      {symbol}: {weight*100:.1f}%")
        
        # Test 3: Risk analysis
        print("\n3. Analyzing portfolio risk...")
        test_portfolio = {"RELIANCE.NS": 0.3, "TCS.NS": 0.3, "HDFCBANK.NS": 0.4}
        risk = await engine.analyze_risk(test_portfolio)
        print(f"   VaR (95%): {risk['var_95']*100:.1f}%")
        print(f"   Portfolio Beta: {risk['portfolio_beta']:.2f}")
        print(f"   Max Drawdown: {risk['max_drawdown']*100:.1f}%")
        print(f"   Risk Rating: {risk['risk_rating']}")
        print(f"   Key Recommendations:")
        for rec in risk['recommendations'][:3]:
            print(f"      - {rec}")
        
        # Test 4: Market summary
        print("\n4. Getting market summary...")
        summary = engine.get_market_summary()
        if summary:
            breadth = summary.get('market_breadth', {})
            print(f"   Market Sentiment: {breadth.get('market_sentiment', 'N/A')}")
            print(f"   Advances: {breadth.get('advances', 0)}")
            print(f"   Declines: {breadth.get('declines', 0)}")
            
            print(f"   Top Gainers: {len(summary.get('top_gainers', []))}")
            print(f"   Top Losers: {len(summary.get('top_losers', []))}")
        
        print("\n" + "=" * 60)
        print("Testing completed!")
    
    # Run tests
    asyncio.run(test_engine())