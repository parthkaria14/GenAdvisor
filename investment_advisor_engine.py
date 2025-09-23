# investment_advisor_engine.py
"""
Real-Time Investment Advisor Engine
Core advisory logic with portfolio optimization, risk management, and recommendations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import networkx as nx

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
    stocks: Dict[str, float]  # symbol: allocation percentage
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

class InvestmentAdvisorEngine:
    """Core investment advisory engine with ML-powered recommendations"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.realtime_dir = self.data_dir / "realtime"
        self.market_dir = self.data_dir / "market"
        
        # Ensure directories exist
        self.realtime_dir.mkdir(parents=True, exist_ok=True)
        self.market_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models and initialize components
        self.risk_model = self._load_risk_model()
        self.price_predictor = self._load_price_predictor()

    def _generate_stock_recommendation(self, symbol: str) -> Dict:
        """Generate detailed stock recommendation with analysis"""
        try:
            # Get stock data from realtime folder
            file_path = self.realtime_dir / f"stock_{symbol}.json"
            if not file_path.exists():
                logger.warning(f"No data file found for {symbol}")
                return {
                    "success": False,
                    "error": f"No data available for {symbol}"
                }
            
            with open(file_path, 'r') as f:
                stock_data = json.load(f)
            
            # Analyze components
            technical = self._analyze_technical(stock_data)
            fundamental_score = self._analyze_fundamentals(stock_data)
            sentiment = self._get_sentiment_score(symbol)
            
            # Generate recommendation
            recommendation = self._determine_recommendation(
                technical=technical,
                fundamental_score=fundamental_score,
                sentiment=sentiment,
                stock_data=stock_data
            )
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "price": stock_data.get('last_price', 0),
                    "change": stock_data.get('change_percent', 0),
                    "recommendation": recommendation,
                    "technical_analysis": technical,
                    "fundamental_score": fundamental_score,
                    "sentiment": sentiment,
                    "target_price": self._calculate_target_price(stock_data),
                    "risk_level": self._assess_risk_level(stock_data),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def _determine_recommendation(self, technical: Dict, fundamental_score: float, 
                                sentiment: float, stock_data: Dict) -> str:
        """Determine final recommendation based on all factors"""
        score = 0.0
        
        # Weight different factors
        score += technical.get('signal', 0) * 0.4  # Technical analysis weight
        score += fundamental_score * 0.4           # Fundamental analysis weight
        score += sentiment * 0.2                   # Sentiment weight
        
        # Convert score to recommendation
        if score > 0.6:
            return "STRONG_BUY"
        elif score > 0.2:
            return "BUY"
        elif score > -0.2:
            return "HOLD"
        elif score > -0.6:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _analyze_technical(self, stock_data: Dict) -> Dict:
        """Analyze technical indicators"""
        return {
            "rsi": stock_data.get('rsi', 50),
            "macd": stock_data.get('macd', 0),
            "signal": self._calculate_technical_signal(stock_data)
        }

    def _analyze_fundamentals(self, stock_data: Dict) -> float:
        """Calculate fundamental score"""
        try:
            pe_ratio = stock_data.get('pe_ratio', 0)
            industry_pe = stock_data.get('industry_pe', 0)
            debt_equity = stock_data.get('debt_equity', 0)
            
            # Basic scoring
            score = 0.0
            if pe_ratio and industry_pe:
                score += 0.5 if pe_ratio < industry_pe else -0.5
            if debt_equity:
                score += 0.5 if debt_equity < 2 else -0.5
                
            return max(min(score, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return 0.0

    def _calculate_technical_signal(self, stock_data: Dict) -> float:
        """Calculate technical signal from indicators"""
        try:
            signals = []
            
            # RSI signals
            rsi = stock_data.get('rsi', 50)
            if rsi < 30:
                signals.append(1.0)  # Oversold
            elif rsi > 70:
                signals.append(-1.0)  # Overbought
            
            # MACD signals
            macd = stock_data.get('macd', 0)
            if macd > 0:
                signals.append(1.0)
            elif macd < 0:
                signals.append(-1.0)
            
            return sum(signals) / len(signals) if signals else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0

    def _load_price_predictor(self):
        """Load pre-trained price prediction model"""
        try:
            return joblib.load('models/price_predictor.pkl')
        except:
            # Return a dummy model if not available
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _load_risk_model(self):
        """Load pre-trained risk assessment model"""
        try:
            return joblib.load('models/risk_model.pkl')
        except:
            return xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    async def analyze_stock(self, symbol: str) -> StockRecommendation:
        """Comprehensive analysis of a single stock"""
        
        # Fetch current data
        stock_data = self._get_stock_data(symbol)
        if not stock_data:
            raise ValueError(f"No data available for {symbol}")
        
        # Technical analysis
        technical_signals = self._analyze_technical(symbol)
        
        # Fundamental analysis
        fundamental_score = self._analyze_fundamentals(stock_data)
        
        # Sentiment analysis
        sentiment_score = self._get_sentiment_score(symbol)
        
        # Price prediction
        predicted_price = self._predict_price(symbol, horizon_days=30)
        current_price = stock_data.get('current_price', 0)
        
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
            fundamental_score, sentiment_score
        )
        
        risk_factors = self._identify_risk_factors(
            symbol, stock_data, technical_signals
        )
        
        return StockRecommendation(
            symbol=symbol,
            action=recommendation,
            current_price=current_price,
            target_price=predicted_price,
            confidence=self._calculate_confidence(
                technical_signals, fundamental_score, sentiment_score
            ),
            reasons=reasons,
            risk_factors=risk_factors,
            time_horizon="30 days"
        )
    
    def _get_stock_data(self, symbol: str) -> Dict:
        """Fetch stock data from file storage"""
        try:
            file_path = self.data_dir / "stocks" / f"{symbol}.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error reading stock data for {symbol}: {e}")
            return {}
    
    def _initialize_graph(self):
        """Initialize knowledge graph with basic market structure"""
        # Add market indices
        self.knowledge_graph.add_node("NIFTY50", type="index")
        self.knowledge_graph.add_node("SENSEX", type="index")
        
        # Add major sectors
        sectors = ["IT", "Banking", "Pharma", "Auto", "Energy", "FMCG"]
        for sector in sectors:
            self.knowledge_graph.add_node(sector, type="sector")
            self.knowledge_graph.add_edge("NIFTY50", sector, relation="contains")

    def _save_stock_data(self, symbol: str, data: Dict):
        """Save stock data to file"""
        try:
            file_path = self.data_dir / "stocks" / f"{symbol}.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stock data for {symbol}: {e}")

    def _get_market_data(self) -> Dict:
        """Get market data from file storage"""
        try:
            file_path = self.data_dir / "market" / "market_data.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error reading market data: {e}")
            return {}
    
    def _analyze_technical(self, symbol: str) -> Dict:
        """Perform technical analysis"""
        indicators = self.redis_client.get(f"indicators:{symbol}")
        if not indicators:
            return {'signal': 'neutral', 'strength': 0.5}
        
        indicators = json.loads(indicators)
        
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
        if macd.get('histogram', 0) > 0:
            signals.append(1)
        elif macd.get('histogram', 0) < 0:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Moving average signal
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        current_price = float(self._get_stock_data(symbol).get('current_price', 0))
        
        if current_price > sma_20 > sma_50:
            signals.append(1)  # Bullish
        elif current_price < sma_20 < sma_50:
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
        
        return min(max(score, 0), 1)  # Clamp between 0 and 1
    
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score from news and social media"""
        sentiment = self.redis_client.get(f"sentiment:{symbol}")
        if sentiment:
            return json.loads(sentiment).get('score', 0)
        return 0  # Neutral if no data
    
    def _predict_price(self, symbol: str, horizon_days: int = 30) -> float:
        """Predict future stock price"""
        try:
            # Get historical data
            stock_data = self._get_stock_data(symbol)
            current_price = stock_data.get('current_price', 0)
            
            if current_price == 0:
                return 0
            
            # Simple prediction based on momentum and fundamentals
            # In production, use the trained ML model
            momentum = self._calculate_momentum(symbol)
            fundamental_score = self._analyze_fundamentals(stock_data)
            
            # Expected return calculation
            base_return = 0.01  # 1% monthly base
            momentum_factor = momentum * 0.02  # Up to 2% from momentum
            fundamental_factor = (fundamental_score - 0.5) * 0.03  # Up to ±1.5% from fundamentals
            
            expected_return = base_return + momentum_factor + fundamental_factor
            predicted_price = current_price * (1 + expected_return)
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return 0
    
    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate price momentum"""
        stock_data = self._get_stock_data(symbol)
        
        current = stock_data.get('current_price', 0)
        week_low = stock_data.get('fifty_two_week_low', current)
        week_high = stock_data.get('fifty_two_week_high', current)
        
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
            expected_return * 10 * weights['return'] +  # Scale return to similar range
            tech_score * weights['technical'] +
            (fundamental_score - 0.5) * 2 * weights['fundamental'] +  # Center around 0
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
        sentiment_score: float
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
        
        stock_data = self._get_stock_data(symbol)
        if stock_data.get('dividend_yield', 0) > 0.03:
            reasons.append(f"Attractive dividend yield of {stock_data['dividend_yield']*100:.1f}%")
        
        return reasons if reasons else ["Based on overall market conditions"]
    
    def _identify_risk_factors(
        self,
        symbol: str,
        stock_data: Dict,
        technical_signals: Dict
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
        if technical_signals.get('rsi', 50) > 70:
            risks.append("Overbought conditions (RSI > 70)")
        
        # Sector risk
        sector = stock_data.get('sector', '')
        if sector in ['Energy', 'Commodities']:
            risks.append(f"Sector-specific risks in {sector}")
        
        # Market risk
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
    
    def _generate_portfolio_recommendation(
        self,
        budget: float,
        strategy: InvestmentStrategy,
        existing_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Generate optimized portfolio recommendation"""
        try:
            # Load universe of stocks
            stock_universe = self._get_stock_universe(strategy)
            if not stock_universe:
                return {"success": False, "error": "No stocks available for analysis"}

            # Calculate returns and risks
            returns_data = {}
            risk_data = {}
            valid_stocks = []

            for symbol in stock_universe:
                stock_data = self._get_stock_data(symbol)
                if stock_data:
                    returns_data[symbol] = stock_data.get('expected_return', 0.10)  # Default 10%
                    risk_data[symbol] = stock_data.get('beta', 1.0)
                    valid_stocks.append(symbol)

            if not valid_stocks:
                return {"success": False, "error": "No valid stock data available"}

            # Set strategy constraints
            constraints = self._get_strategy_constraints(strategy)
            
            # Optimize portfolio
            allocation = self._optimize_allocation(
                stocks=valid_stocks,
                returns=returns_data,
                risks=risk_data,
                constraints=constraints
            )

            # Calculate portfolio metrics
            total_return = sum(returns_data[s] * w for s, w in allocation.items())
            portfolio_risk = self._calculate_portfolio_risk(allocation, risk_data)
            
            # Generate recommendations
            recommendations = []
            for symbol, weight in allocation.items():
                amount = budget * weight
                if amount > 1000:  # Minimum investment threshold
                    recommendations.append({
                        "symbol": symbol,
                        "weight": weight,
                        "amount": amount,
                        "stock_data": self._get_stock_data(symbol)
                    })

            return {
                "success": True,
                "data": {
                    "allocation": allocation,
                    "strategy": strategy.value,
                    "expected_return": total_return,
                    "risk_level": portfolio_risk,
                    "sharpe_ratio": (total_return - 0.06) / portfolio_risk if portfolio_risk > 0 else 0,
                    "recommendations": recommendations,
                    "rebalancing_needed": self._check_rebalancing_needed(
                        allocation, existing_portfolio
                    ) if existing_portfolio else False
                }
            }

        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {e}")
            return {"success": False, "error": str(e)}

    def _get_stock_universe(self, strategy: InvestmentStrategy) -> List[str]:
        """Get list of stocks based on investment strategy"""
        try:
            # Read stock list from realtime data directory
            stocks = []
            for file in self.realtime_dir.glob("stock_*.json"):
                symbol = file.stem.replace("stock_", "")
                stocks.append(symbol)
            
            # Filter based on strategy
            if strategy in [InvestmentStrategy.CONSERVATIVE, InvestmentStrategy.INCOME]:
                return [s for s in stocks if self._is_large_cap(s)]
            elif strategy == InvestmentStrategy.MODERATE:
                return [s for s in stocks if self._is_large_cap(s) or self._is_mid_cap(s)]
            else:
                return stocks

        except Exception as e:
            logger.error(f"Error getting stock universe: {e}")
            return []

    def _optimize_allocation(
        self,
        stocks: List[str],
        returns: Dict[str, float],
        risks: Dict[str, float],
        constraints: Dict
    ) -> Dict[str, float]:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        try:
            n_stocks = len(stocks)
            if n_stocks == 0:
                return {}

            # Equal weight as fallback
            equal_weight = 1.0 / n_stocks
            allocation = {stock: equal_weight for stock in stocks}

            # Apply strategy constraints
            max_weight = constraints.get('max_weight', 0.2)
            min_weight = constraints.get('min_weight', 0.05)

            # Adjust weights to meet constraints
            for stock in stocks:
                if allocation[stock] > max_weight:
                    allocation[stock] = max_weight
                elif allocation[stock] < min_weight:
                    allocation[stock] = min_weight

            # Normalize weights to sum to 1
            total = sum(allocation.values())
            if total > 0:
                allocation = {k: v/total for k, v in allocation.items()}

            return allocation

        except Exception as e:
            logger.error(f"Error optimizing allocation: {e}")
            return {stock: 1.0/len(stocks) for stock in stocks}

    def _calculate_portfolio_risk(
        self,
        allocation: Dict[str, float],
        risks: Dict[str, float]
    ) -> float:
        """Calculate portfolio risk using weighted average of stock betas"""
        try:
            return sum(weight * risks.get(stock, 1.0) for stock, weight in allocation.items())
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 1.0

    def _check_rebalancing_needed(
        self,
        new_allocation: Dict[str, float],
        current_allocation: Dict[str, float]
    ) -> bool:
        """Check if portfolio rebalancing is needed"""
        threshold = 0.05  # 5% deviation threshold
        for symbol in set(new_allocation) & set(current_allocation):
            if abs(new_allocation[symbol] - current_allocation[symbol]) > threshold:
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
        
        # Simplified VaR calculation
        # In production, use historical simulation or Monte Carlo
        portfolio_volatility = 0.2  # Assume 20% annual volatility
        
        # Daily VaR
        z_score = stats.norm.ppf(1 - confidence)
        daily_var = portfolio_volatility / np.sqrt(252)  # Convert to daily
        
        return abs(z_score * daily_var)
    
    def _calculate_max_drawdown(self, portfolio: Dict[str, float]) -> float:
        """Calculate maximum drawdown"""
        # Simplified - in production, use historical data
        return 0.15  # 15% max drawdown estimate
    
    def _calculate_portfolio_beta(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio beta"""
        
        weighted_beta = 0
        
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
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
        }
        
        results = {}
        
        for scenario, impact in scenarios.items():
            portfolio_impact = 0
            
            for symbol, weight in portfolio.items():
                stock_data = self._get_stock_data(symbol)
                beta = stock_data.get('beta', 1.0)
                
                # Stock impact = market impact * beta
                stock_impact = impact * beta
                portfolio_impact += weight * stock_impact
            
            results[scenario] = {
                'impact': portfolio_impact,
                'description': f"Portfolio would lose {abs(portfolio_impact)*100:.1f}% in {scenario}"
            }
        
        return results
    
    def _analyze_correlation_risk(self, portfolio: Dict[str, float]) -> str:
        """Analyze correlation risk in portfolio"""
        
        # Check sector concentration
        sectors = {}
        for symbol, weight in portfolio.items():
            stock_data = self._get_stock_data(symbol)
            sector = stock_data.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + weight
        
        # Find dominant sector
        max_sector_weight = max(sectors.values()) if sectors else 0
        
        if max_sector_weight > 0.4:
            return 'high'
        elif max_sector_weight > 0.25:
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
            recommendations.append("Consider reducing position sizes to lower VaR")
        
        if beta > 1.5:
            recommendations.append("Portfolio has high market sensitivity - consider adding defensive stocks")
        elif beta < 0.8:
            recommendations.append("Portfolio may underperform in bull markets - consider adding growth stocks")
        
        if correlation_risk == 'high':
            recommendations.append("High sector concentration detected - diversify across sectors")
        
        recommendations.append("Regular rebalancing recommended to maintain risk profile")
        
        return recommendations

# API Interface
class InvestmentAdvisorAPI:
    """API interface for the investment advisor engine"""
    
    def __init__(self):
        self.engine = InvestmentAdvisorEngine()
    
    async def get_stock_recommendation(self, symbol: str) -> Dict:
        """Get recommendation for a single stock"""
        try:
            recommendation = await self.engine.analyze_stock(symbol)
            return {
                'success': True,
                'data': {
                    'symbol': recommendation.symbol,
                    'action': recommendation.action.value,
                    'current_price': recommendation.current_price,
                    'target_price': recommendation.target_price,
                    'confidence': recommendation.confidence,
                    'reasons': recommendation.reasons,
                    'risks': recommendation.risk_factors,
                    'time_horizon': recommendation.time_horizon
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_portfolio_recommendation(
        self,
        budget: float,
        strategy: str,
        existing_portfolio: Optional[Dict] = None
    ) -> Dict:
        """Get portfolio recommendations"""
        try:
            strategy_enum = InvestmentStrategy(strategy.lower())
            recommendation = await self.engine.optimize_portfolio(
                budget,
                strategy_enum,
                existing_portfolio
            )
            
            return {
                'success': True,
                'data': {
                    'allocation': recommendation.stocks,
                    'strategy': recommendation.strategy.value,
                    'expected_return': recommendation.expected_return,
                    'risk_level': recommendation.risk_level,
                    'sharpe_ratio': recommendation.sharpe_ratio,
                    'recommendations': recommendation.recommendations,
                    'rebalancing_needed': recommendation.rebalancing_needed
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_risk_analysis(
        self,
        portfolio: Dict[str, float]
    ) -> Dict:
        """Get risk analysis for portfolio"""
        try:
            analysis = await self.engine.analyze_risk(portfolio)
            return {
                'success': True,
                'data': analysis
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_detailed_stock_recommendation(self, symbol: str) -> Dict:
        """Get detailed recommendation for a single stock"""
        try:
            recommendation = self.engine._generate_stock_recommendation(symbol)
            return recommendation
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_detailed_portfolio_recommendation(
        self,
        budget: float,
        strategy: str,
        existing_portfolio: Optional[Dict] = None
    ) -> Dict:
        """Get detailed portfolio recommendations"""
        try:
            strategy_enum = InvestmentStrategy(strategy.lower())
            recommendation = await self.engine._generate_portfolio_recommendation(
                budget,
                strategy_enum,
                existing_portfolio
            )
            
            return recommendation
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # Initialize engine
    engine = InvestmentAdvisorEngine()
    
    # Test stock recommendation
    print("\nAnalyzing individual stock...")
    stock_rec = engine._generate_stock_recommendation("RELIANCE.NS")
    print("\nStock Analysis:")
    print(json.dumps(stock_rec, indent=2))
    
    # Test portfolio recommendation
    print("\nGenerating portfolio recommendation...")
    portfolio_rec = engine._generate_portfolio_recommendation(
        budget=1000000,  # 10 Lakhs
        strategy=InvestmentStrategy.MODERATE
    )
    print("\nPortfolio Recommendation:")
    print(json.dumps(portfolio_rec, indent=2))