# GenAdvisor - AI-Powered Indian Market Investment Advisor

## Overview
GenAdvisor is a production-ready FastAPI application that provides real-time investment advisory for the Indian stock market. It combines advanced data ingestion, RAG (Retrieval-Augmented Generation) system, and an AI-powered investment advisor engine to deliver comprehensive market insights and personalized investment recommendations.

## Features
- Real-time market data tracking
- AI-powered stock analysis
- Portfolio optimization
- Risk analysis
- Natural language querying
- Stock screening
- News and sentiment analysis
- WebSocket support for live updates
- Comprehensive REST API

## Tech Stack
- FastAPI
- Python 3.10+
- Chroma DB for vector storage
- yfinance for market data
- WebSocket for real-time updates
- Pydantic for data validation
- File-based storage system

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/GenAdvisor.git
cd GenAdvisor
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create .env file with required configurations
```bash
cp .env.example .env
```

## Usage

### Starting the Server
```bash
python main.py
```
The server will start at `http://localhost:8000`

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

#### Market Data
- GET `/api/v1/market/overview` - Get market overview
- GET `/api/v1/market/stock/{symbol}` - Get stock details
- GET `/api/v1/market/sectors` - Get sector performance

#### Analysis
- POST `/api/v1/analyze/stock` - Analyze specific stock
- POST `/api/v1/analyze/portfolio` - Portfolio optimization
- POST `/api/v1/analyze/risk` - Risk analysis

#### Natural Language Queries
- POST `/api/v1/query` - Process natural language queries

#### Screener
- POST `/api/v1/screener` - Screen stocks based on criteria

#### Real-time Updates
- WebSocket `/ws` - Real-time market updates

## Project Structure
```
GenAdvisor/
├── main.py                    # Main FastAPI application
├── enhanced_data.py          # Data ingestion system
├── advanced_rag_system.py    # RAG implementation
├── investment_advisor_engine.py # Advisory logic
├── requirements.txt          # Project dependencies
├── .env                     # Environment variables
└── data/                    # Data storage
    ├── realtime/           # Real-time market data
    ├── knowledge_base/     # RAG knowledge base
    └── chroma/             # Vector database
```

## API Examples

### Portfolio Optimization
```json
POST /api/v1/analyze/portfolio
{
    "budget": 1000000,
    "strategy": "moderate",
    "existing_portfolio": {
        "RELIANCE.NS": 0.3,
        "TCS.NS": 0.2
    }
}
```

### Stock Analysis
```json
POST /api/v1/analyze/stock
{
    "symbol": "RELIANCE.NS"
}
```

## Error Handling
The API implements comprehensive error handling with appropriate HTTP status codes and detailed error messages.

## Performance Considerations
- Uses file-based storage for better reliability
- Implements background tasks for data updates
- Supports data caching
- Handles rate limiting for external APIs

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments
- FastAPI framework
- yfinance for market data
- Indian stock market data providers

## Support
For support, email support@genadvisor.com or open an issue in the repository.