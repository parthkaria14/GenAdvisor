# advanced_rag_system.py
"""
Advanced RAG System with Vector Database for Indian Market Intelligence
Integrates real-time data with financial knowledge base
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# Add this with the other imports at the top of the file
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Graph RAG imports
import networkx as nx
from collections import defaultdict

# Vector database
import chromadb
from chromadb.config import Settings
import pinecone

# ML imports
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
load_dotenv()

# Redis for caching (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available - using file-based storage")

from logger_config import setup_logger
logger = setup_logger()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    STOCK_ANALYSIS = "stock_analysis"
    PORTFOLIO_ADVICE = "portfolio_advice"
    MARKET_OVERVIEW = "market_overview"
    SECTOR_ANALYSIS = "sector_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    NEWS_SENTIMENT = "news_sentiment"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class RAGContext:
    query: str
    query_type: QueryType
    relevant_docs: List[Document]
    market_data: Dict
    technical_indicators: Dict
    sentiment_score: float
    confidence_score: float
    graph_results: List[Dict] = None  # Graph RAG results

class AdvancedRAGSystem:
    """
    Production-ready RAG system for Indian market investment advisory
    """
    
    def __init__(
        self,
        vector_db_type: str = "chroma",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model: str = "openai/gpt-oss-120b",  # Default Groq model
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_redis: bool = False
):
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_db_type = vector_db_type
        self.vector_store = self._initialize_vector_store()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=llm_model,  # mixtral-8x7b-32768 or llama2-70b-4096
            temperature=0.1,
            max_tokens=4096,
            api_key=os.getenv("GROQ_API_KEY")  # Make sure to set this environment variable
    )
        
        # Initialize specialized models
        self.sentiment_model = self._initialize_sentiment_model()
        self.ner_model = self._initialize_ner_model()
        
        # Redis for caching (optional)
        self.enable_redis = REDIS_AVAILABLE
        if self.enable_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using file-based storage.")
                self.enable_redis = False
                self.redis_client = None
        else:
            self.enable_redis = bool(enable_redis) and REDIS_AVAILABLE
            if self.enable_redis:
                logger.info("Redis enabled for caching")
                      # initialize redis client here if you want, e.g. redis.Redis(...)
            else:
                if enable_redis and not REDIS_AVAILABLE:
                    logger.warning("enable_redis=True requested but Redis package not available; using file-based storage")
                else:
                    logger.info("Using file-based storage (Redis disabled)")
            
        # File-based data paths
        self.data_dir = "data"
        self.realtime_dir = os.path.join(self.data_dir, "realtime")
        self.knowledge_base_dir = os.path.join(self.data_dir, "knowledge_base")
        
        # Ensure directories exist
        os.makedirs(self.realtime_dir, exist_ok=True)
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Document splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        
        # Load file-based data
        self._load_file_data()
        
        # Initialize Graph RAG
        self.knowledge_graph = nx.Graph()
        self._build_knowledge_graph()
    def refresh_knowledge_graph(self):
        """
        Reloads all file-based data and rebuilds the knowledge graph.
        """
        logger.info("Refreshing knowledge graph...")
        try:
            # Step 1: Reload all data from files
            self._load_file_data()
            logger.info("File data reloaded.")
            
            # Step 2: Rebuild the graph with the new data
            self._build_knowledge_graph()
            logger.info("Knowledge graph rebuilt successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh knowledge graph: {e}")
            return False
    def _load_file_data(self):
        """Load data from JSON and CSV files"""
        self.file_data = {
            'stocks': {},
            'market_breadth': {},
            'sector_performance': {},
            'technical_indicators': {},
            'news_data': []
        }
        
        try:
            # Load stocks data
            stocks_csv_path = os.path.join(self.realtime_dir, "stocks.csv")
            if os.path.exists(stocks_csv_path):
                stocks_df = pd.read_csv(stocks_csv_path)
                for _, row in stocks_df.iterrows():
                    ticker = row.get('ticker', '')
                    if ticker:
                        self.file_data['stocks'][ticker] = row.to_dict()
                logger.info(f"Loaded {len(self.file_data['stocks'])} stocks from CSV")
            
            # Load individual stock JSON files
            for filename in os.listdir(self.realtime_dir):
                if filename.startswith('stock_') and filename.endswith('.json'):
                    ticker = filename.replace('stock_', '').replace('.json', '').replace('_', '.')
                    file_path = os.path.join(self.realtime_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            self.file_data['stocks'][ticker] = data
                    except Exception as e:
                        logger.warning(f"Error loading {filename}: {e}")
            
            # Load market breadth data
            breadth_path = os.path.join(self.realtime_dir, "market_breadth.json")
            if os.path.exists(breadth_path):
                with open(breadth_path, 'r') as f:
                    self.file_data['market_breadth'] = json.load(f)
            
            # Load sector performance data
            sectors_path = os.path.join(self.realtime_dir, "sector_performance.json")
            if os.path.exists(sectors_path):
                with open(sectors_path, 'r') as f:
                    self.file_data['sector_performance'] = json.load(f)
            
            # Load technical indicators
            indicators_path = os.path.join(self.realtime_dir, "technical_indicators.json")
            if os.path.exists(indicators_path):
                with open(indicators_path, 'r') as f:
                    self.file_data['technical_indicators'] = json.load(f)
            
            # Load news data from existing CSV files
            news_files = [
                "news_Indian_stock_market.csv",
                "news_HDFC_Bank.csv", 
                "news_Infosys.csv",
                "news_Nifty_50.csv",
                "news_Reliance_Industries.csv",
                "news_TCS.csv"
            ]
            
            for news_file in news_files:
                news_path = os.path.join(self.data_dir, news_file)
                if os.path.exists(news_path):
                    try:
                        news_df = pd.read_csv(news_path)
                        for _, row in news_df.iterrows():
                            self.file_data['news_data'].append(row.to_dict())
                    except Exception as e:
                        logger.warning(f"Error loading {news_file}: {e}")
            
            logger.info(f"Loaded file data: {len(self.file_data['stocks'])} stocks, "
                       f"{len(self.file_data['news_data'])} news articles")
                       
        except Exception as e:
            logger.error(f"Error loading file data: {e}")
    
    def _build_knowledge_graph(self):
        """Build a knowledge graph from the loaded data"""
        try:
            # Clear existing graph
            self.knowledge_graph.clear()
            
            # Add market index nodes
            self.knowledge_graph.add_node("NIFTY50", type="index", level="broad_market")
            self.knowledge_graph.add_node("SENSEX", type="index", level="broad_market")
            
            # Add stock nodes with rich metadata
            for ticker, stock_data in self.file_data['stocks'].items():
                self.knowledge_graph.add_node(
                    ticker,
                    type='stock',
                    name=stock_data.get('name', ticker),
                    sector=stock_data.get('sector', 'Unknown'),
                    price=stock_data.get('current_price', 0),
                    change_percent=stock_data.get('change_percent', 0),
                    volume=stock_data.get('volume', 0),
                    market_cap=stock_data.get('market_cap', 0),
                    pe_ratio=stock_data.get('pe_ratio'),
                    beta=stock_data.get('beta', 1.0)
                )
                
                # Connect stocks to sectors
                sector = stock_data.get('sector', 'Unknown')
                if sector != 'Unknown':
                    if not self.knowledge_graph.has_node(sector):
                        self.knowledge_graph.add_node(sector, type='sector')
                    self.knowledge_graph.add_edge(ticker, sector, relation='belongs_to')
                    
                    # Connect sector to market indices
                    if not self.knowledge_graph.has_edge(sector, 'NIFTY50'):
                        self.knowledge_graph.add_edge(sector, 'NIFTY50', relation='component_of')
            
            # Add market breadth relationships
            if self.file_data['market_breadth']:
                breadth_data = self.file_data['market_breadth']
                self.knowledge_graph.add_node(
                    'market_breadth',
                    type='market_indicator',
                    advances=breadth_data.get('advances', 0),
                    declines=breadth_data.get('declines', 0),
                    sentiment=breadth_data.get('market_sentiment', 'neutral'),
                    timestamp=breadth_data.get('timestamp', '')
                )
                
                # Connect market breadth to indices
                self.knowledge_graph.add_edge('market_breadth', 'NIFTY50', relation='indicates')
                self.knowledge_graph.add_edge('market_breadth', 'SENSEX', relation='indicates')
            
            # Add technical indicators as node attributes
            for ticker, indicators in self.file_data.get('technical_indicators', {}).items():
                if self.knowledge_graph.has_node(ticker):
                    nx.set_node_attributes(self.knowledge_graph, {
                        ticker: {
                            'rsi': indicators.get('rsi'),
                            'macd': indicators.get('macd', {}).get('histogram'),
                            'sma_20': indicators.get('sma_20'),
                            'sma_50': indicators.get('sma_50')
                        }
                    })
            
            # Add news relationships
            for news_item in self.file_data['news_data']:
                title = news_item.get('title', '')
                news_id = f"news_{hash(title)}"
                
                self.knowledge_graph.add_node(
                    news_id,
                    type='news',
                    title=title,
                    sentiment=news_item.get('sentiment', 'neutral'),
                    date=news_item.get('date', ''),
                    source=news_item.get('source', '')
                )
                
                # Extract and link entities
                # Extract and link entities
                # Use the *better* NER-based extractor
                extracted = self.extract_entities(f"{title} {news_item.get('content', '')}")
                entities_to_link = extracted.get('companies', []) + extracted.get('sectors', [])
                
                for entity_symbol in entities_to_link:
                    if self.knowledge_graph.has_node(entity_symbol):
                        self.knowledge_graph.add_edge(news_id, entity_symbol, relation='mentions')
            
            logger.info(f"Built knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")    
    
    def _graph_retrieve(self, query: str, max_nodes: int = 10) -> List[Dict]:
        """Retrieve relevant information using graph traversal"""
        try:
            query_entities = self._extract_entities_from_text(query)
            relevant_nodes = []
            
            # Find nodes directly related to query entities
            for entity in query_entities:
                if entity in self.knowledge_graph:
                    # Get the node and its neighbors
                    node_data = self.knowledge_graph.nodes[entity]
                    neighbors = list(self.knowledge_graph.neighbors(entity))
                    
                    relevant_nodes.append({
                        'node': entity,
                        'data': node_data,
                        'type': 'direct_match',
                        'relevance_score': 1.0
                    })
                    
                    # Add neighboring nodes
                    for neighbor in neighbors[:3]:  # Limit to top 3 neighbors
                        neighbor_data = self.knowledge_graph.nodes[neighbor]
                        relevant_nodes.append({
                            'node': neighbor,
                            'data': neighbor_data,
                            'type': 'related',
                            'relevance_score': 0.7
                        })
            
            # If no direct matches, do semantic search on graph nodes
            if not relevant_nodes:
                query_lower = query.lower()
                for node in self.knowledge_graph.nodes():
                    node_data = self.knowledge_graph.nodes[node]
                    
                    # Calculate relevance based on text similarity
                    relevance = 0
                    if 'name' in node_data:
                        if any(word in node_data['name'].lower() for word in query_lower.split()):
                            relevance += 0.5
                    
                    if 'sector' in node_data:
                        if any(word in node_data['sector'].lower() for word in query_lower.split()):
                            relevance += 0.3
                    
                    if relevance > 0:
                        relevant_nodes.append({
                            'node': node,
                            'data': node_data,
                            'type': 'semantic_match',
                            'relevance_score': relevance
                        })
            
            # Sort by relevance and return top results
            relevant_nodes.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_nodes[:max_nodes]
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
    
    def _get_cache(self, key: str):
        """Get cached result from Redis or file"""
        if self.enable_redis and self.redis_client:
            try:
                cached_result = self.redis_client.get(key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Fallback to file cache
        cache_file = os.path.join(self.knowledge_base_dir, f"cache_{hash(key)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"File cache get failed: {e}")
        
        return None
    
    def _set_cache(self, key: str, data: Any, expiry: int = 300):
        """Set cache result in Redis or file"""
        # Try Redis first
        if self.enable_redis and self.redis_client:
            try:
                self.redis_client.setex(key, expiry, json.dumps(data, default=str))
                return True
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # Fallback to file cache
        try:
            cache_file = os.path.join(self.knowledge_base_dir, f"cache_{hash(key)}.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)
            return True
        except Exception as e:
            logger.warning(f"File cache set failed: {e}")
            return False
    
    def _get_stock_data(self, company: str) -> Optional[Dict]:
        """Get stock data from file or Redis"""
        # Try Redis first
        if self.enable_redis and self.redis_client:
            try:
                stock_data = self.redis_client.get(f"stock:{company}")
                if stock_data:
                    return json.loads(stock_data)
            except Exception as e:
                logger.warning(f"Redis stock data get failed: {e}")
        
        # Fallback to file data
        return self.file_data['stocks'].get(company)
    
    def _get_sector_data(self, sector: str) -> Optional[Dict]:
        """Get sector data from file or Redis"""
        # Try Redis first
        if self.enable_redis and self.redis_client:
            try:
                sector_data = self.redis_client.get(f"sector:{sector}")
                if sector_data:
                    return json.loads(sector_data)
            except Exception as e:
                logger.warning(f"Redis sector data get failed: {e}")
        
        # Fallback to file data
        return self.file_data['sector_performance'].get(sector)
    
    def _get_market_breadth(self) -> Optional[Dict]:
        """Get market breadth data from file or Redis"""
        # Try Redis first
        if self.enable_redis and self.redis_client:
            try:
                market_breadth = self.redis_client.get("market:breadth")
                if market_breadth:
                    return json.loads(market_breadth)
            except Exception as e:
                logger.warning(f"Redis market breadth get failed: {e}")
        
        # Fallback to file data
        return self.file_data['market_breadth']
    
    def _get_technical_indicators(self, company: str) -> Optional[Dict]:
        """Get technical indicators from file or Redis"""
        # Try Redis first
        if self.enable_redis and self.redis_client:
            try:
                indicators = self.redis_client.get(f"indicators:{company}")
                if indicators:
                    return json.loads(indicators)
            except Exception as e:
                logger.warning(f"Redis technical indicators get failed: {e}")
        
        # Fallback to file data
        return self.file_data['technical_indicators'].get(company)
    
    def _initialize_vector_store(self):
        """Initialize vector database (Chroma or Pinecone)"""
        if self.vector_db_type == "chroma":
            # Chroma DB setup
            persist_directory = "data/chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            

            client = chromadb.PersistentClient(path="data/chroma")

            
            return Chroma(
                collection_name="indian_market_knowledge",
                embedding_function=self.embeddings,
                client=client,
                persist_directory=persist_directory
            )
        
        elif self.vector_db_type == "pinecone":
            # Pinecone setup
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV", "us-west-2")
            )
            
            index_name = "indian-market-index"
            
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  # Dimension of embeddings
                    metric="cosine"
                )
            
            return Pinecone.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings
            )
    
    def _initialize_sentiment_model(self):
        """Initialize financial sentiment analysis model"""
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def _initialize_ner_model(self):
        """Initialize NER model for entity extraction"""
        return pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
    
    async def ingest_documents(self, document_paths: List[str]):
        """Ingest and index documents into vector store"""
        all_documents = []
        
        for path in document_paths:
            logger.info(f"Ingesting document: {path}")
            
            # Determine loader based on file type
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.endswith('.csv'):
                loader = CSVLoader(path)
            else:
                loader = TextLoader(path)
            
            # Load and split documents
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in split_docs:
                doc.metadata.update({
                    'source': path,
                    'ingested_at': datetime.now().isoformat(),
                    'doc_type': self._classify_document(doc.page_content)
                })
            
            all_documents.extend(split_docs)
        
        # Add to vector store in batches
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            self.vector_store.add_documents(batch)
            logger.info(f"Indexed batch {i//batch_size + 1}")
        
        # Persist if using Chroma
        if self.vector_db_type == "chroma":
            self.vector_store.persist()
        
        logger.info(f"Successfully indexed {len(all_documents)} document chunks")
    
    def _classify_document(self, content: str) -> str:
        """Classify document type based on content"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['earnings', 'revenue', 'profit', 'loss']):
            return 'financial_report'
        elif any(term in content_lower for term in ['rbi', 'monetary', 'policy', 'inflation']):
            return 'regulatory'
        elif any(term in content_lower for term in ['technical', 'rsi', 'macd', 'bollinger']):
            return 'technical_analysis'
        elif any(term in content_lower for term in ['news', 'report', 'announce']):
            return 'news'
        else:
            return 'general'
    
    def classify_query(self, query: str) -> QueryType:
        """Classify user query to determine processing strategy"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['stock', 'share', 'company', 'analyze']):
            return QueryType.STOCK_ANALYSIS
        elif any(term in query_lower for term in ['portfolio', 'allocate', 'diversify']):
            return QueryType.PORTFOLIO_ADVICE
        elif any(term in query_lower for term in ['market', 'nifty', 'sensex', 'index']):
            return QueryType.MARKET_OVERVIEW
        elif any(term in query_lower for term in ['sector', 'industry', 'banking', 'it', 'pharma']):
            return QueryType.SECTOR_ANALYSIS
        elif any(term in query_lower for term in ['technical', 'chart', 'pattern', 'indicator']):
            return QueryType.TECHNICAL_ANALYSIS
        elif any(term in query_lower for term in ['fundamental', 'earning', 'valuation', 'pe']):
            return QueryType.FUNDAMENTAL_ANALYSIS
        elif any(term in query_lower for term in ['news', 'sentiment', 'announcement']):
            return QueryType.NEWS_SENTIMENT
        elif any(term in query_lower for term in ['risk', 'volatility', 'var', 'beta']):
            return QueryType.RISK_ASSESSMENT
        else:
            return QueryType.STOCK_ANALYSIS
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query (companies, sectors, etc.)"""
        
        extracted = {
            'companies': [],
            'sectors': [],
            'metrics': [],
            'time_periods': []
        }
        
        # --- START OF IMPROVED LOGIC ---
        
        # 1. Extract NER Entities
        try:
            entities = self.ner_model(query)
            org_names = [e['word'] for e in entities if e['entity_group'] == 'ORG']
        except Exception as e:
            logger.warning(f"NER model failed: {e}")
            org_names = []

        # 2. Find matching company nodes in graph
        # We search both tickers and company names
        all_stock_nodes = self.file_data['stocks'].keys()
        
        for name in org_names:
            name_lower = name.lower()
            for ticker in all_stock_nodes:
                stock_data = self.file_data['stocks'][ticker]
                stock_name = stock_data.get('name', '').lower()
                ticker_clean = ticker.replace('.NS', '').replace('.BO', '').lower()
                
                # Match on ticker or partial name
                if (name_lower == ticker_clean or 
                    name_lower in stock_name or 
                    (len(name_lower) > 3 and name_lower in ticker_clean)):
                    
                    if ticker not in extracted['companies']:
                        extracted['companies'].append(ticker)

        # 3. Handle queries that just use the ticker
        for word in query.upper().split():
            if word in all_stock_nodes and word not in extracted['companies']:
                 extracted['companies'].append(word)
        
        # --- END OF IMPROVED LOGIC ---

        # 4. Extract sectors (your existing logic is good)
        all_sectors = [
            node for node, data in self.knowledge_graph.nodes(data=True) 
            if data.get('type') == 'sector'
        ]
        query_lower = query.lower()
        for sector in all_sectors:
            if sector.lower() in query_lower and sector not in extracted['sectors']:
                extracted['sectors'].append(sector)
        
        # 5. Extract metrics
        metrics = ['pe', 'pb', 'roe', 'debt', 'margin', 'growth', 'dividend', 'rsi', 'macd']
        for metric in metrics:
            if metric in query_lower:
                extracted['metrics'].append(metric)
        
        return extracted
    
    async def retrieve_context(
        self,
        query: str,
        query_type: QueryType,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Advanced retrieval with hybrid search and re-ranking"""
        
        # Check cache first
        cache_key = f"rag:{query}:{query_type.value}"
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Semantic search
        semantic_results = self.vector_store.similarity_search_with_score(
            query,
            k=k*2,  # Get more for re-ranking
            filter=filters
        )
        
        # Keyword search (if supported by vector store)
        keyword_results = self._keyword_search(query, k=k)
        
        # Combine and deduplicate
        all_results = semantic_results + keyword_results
        seen = set()
        unique_results = []
        
        for doc, score in all_results:
            doc_id = doc.metadata.get('source', '') + doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc, score))
        
        # Re-rank based on query type and relevance
        reranked = self._rerank_documents(
            query,
            unique_results,
            query_type
        )
        
        # Cache results
        self._set_cache(cache_key, [doc.dict() for doc in reranked[:k]], 300)
        
        return reranked[:k]
    
    def _keyword_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform keyword-based search"""
        # This would integrate with Elasticsearch or similar
        # For now, return empty list
        return []
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        query_type: QueryType
    ) -> List[Document]:
        """Re-rank documents based on relevance and query type"""
        
        # Score each document
        scored_docs = []
        for doc, base_score in documents:
            # Calculate relevance scores
            content_score = self._calculate_content_relevance(query, doc.page_content)
            recency_score = self._calculate_recency_score(doc.metadata.get('ingested_at'))
            type_score = self._calculate_type_relevance(query_type, doc.metadata.get('doc_type'))
            
            # Weighted combination
            final_score = (
                base_score * 0.4 +
                content_score * 0.3 +
                recency_score * 0.2 +
                type_score * 0.1
            )
            
            scored_docs.append((doc, final_score))
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]
    
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """Calculate content relevance score"""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & content_terms)
        return overlap / len(query_terms)
    
    def _calculate_recency_score(self, timestamp: Optional[str]) -> float:
        """Calculate recency score for documents"""
        if not timestamp:
            return 0.5
        
        try:
            doc_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - doc_time).days
            
            # Exponential decay
            return np.exp(-age_days / 30)  # Half-life of 30 days
        except:
            return 0.5
    
    def _calculate_type_relevance(self, query_type: QueryType, doc_type: Optional[str]) -> float:
        """Calculate document type relevance for query"""
        if not doc_type:
            return 0.5
        
        relevance_matrix = {
            QueryType.FUNDAMENTAL_ANALYSIS: {
                'financial_report': 1.0,
                'regulatory': 0.6,
                'news': 0.4,
                'technical_analysis': 0.2
            },
            QueryType.TECHNICAL_ANALYSIS: {
                'technical_analysis': 1.0,
                'news': 0.4,
                'financial_report': 0.3,
                'regulatory': 0.2
            },
            QueryType.NEWS_SENTIMENT: {
                'news': 1.0,
                'regulatory': 0.5,
                'financial_report': 0.3,
                'technical_analysis': 0.2
            }
        }
        
        return relevance_matrix.get(query_type, {}).get(doc_type, 0.5)
    
    def _calculate_aggregate_sentiment(self, graph_results: List[Dict]) -> float:
        """
        Calculate aggregate sentiment from graph results
        Returns a float between -1 (very negative) and 1 (very positive)
        """
        try:
            sentiments = []
            weights = []
            
            for result in graph_results:
                # Get sentiment value based on result type
                if result['type'] == 'news':
                    sentiment = self._convert_sentiment_to_score(
                        result['data'].get('sentiment', 'neutral')
                    )
                    sentiments.append(sentiment)
                    weights.append(0.7)  # News sentiment weight
                    
                elif result['type'] == 'market_context':
                    market_sentiment = self._convert_sentiment_to_score(
                        result['data'].get('sentiment', 'neutral')
                    )
                    sentiments.append(market_sentiment)
                    weights.append(1.0)  # Market sentiment weight
                    
                elif result['type'] == 'stock_info':
                    # Calculate technical sentiment
                    tech_data = result['data']
                    if 'rsi' in tech_data:
                        rsi = float(tech_data['rsi'])
                        rsi_sentiment = -1.0 if rsi > 70 else 1.0 if rsi < 30 else 0.0
                        sentiments.append(rsi_sentiment)
                        weights.append(0.5)  # Technical indicator weight
                    
                    if 'macd' in tech_data:
                        macd = float(tech_data['macd'])
                        macd_sentiment = 1.0 if macd > 0 else -1.0 if macd < 0 else 0.0
                        sentiments.append(macd_sentiment)
                        weights.append(0.5)
            
            # Calculate weighted average if we have sentiments
            if sentiments and weights:
                return sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
            
            return 0.0  # Neutral if no sentiment data
            
        except Exception as e:
            logger.error(f"Error calculating aggregate sentiment: {e}")
            return 0.0

    def _convert_sentiment_to_score(self, sentiment: str) -> float:
        """Convert string sentiment to numeric score"""
        sentiment_map = {
            'very_positive': 1.0,
            'positive': 0.5,
            'neutral': 0.0,
            'negative': -0.5,
            'very_negative': -1.0
        }
        return sentiment_map.get(sentiment.lower(), 0.0)

    def _calculate_confidence_score(self, graph_results: List[Dict], vector_results: List[Document]) -> float:
        """
        Calculate overall confidence score for the response
        Returns a float between 0 and 1
        """
        try:
            factors = []
            
            # Graph coverage
            if graph_results:
                coverage = min(len(graph_results) / 5, 1.0)  # Normalize to max 1.0
                factors.append(coverage * 0.4)  # Graph results weight
            
            # Vector results relevance
            if vector_results:
                # Use metadata scores if available
                scores = [getattr(doc.metadata, 'score', 0.5) for doc in vector_results]
                avg_score = sum(scores) / len(scores)
                factors.append(avg_score * 0.3)  # Vector results weight
            
            # Data freshness
            current_time = datetime.now()
            timestamps = []
            for result in graph_results:
                if 'timestamp' in result['data']:
                    try:
                        ts = datetime.fromisoformat(result['data']['timestamp'])
                        age_hours = (current_time - ts).total_seconds() / 3600
                        freshness = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
                        timestamps.append(freshness)
                    except (ValueError, TypeError):
                        continue
            
            if timestamps:
                avg_freshness = sum(timestamps) / len(timestamps)
                factors.append(avg_freshness * 0.3)  # Freshness weight
            
            # Calculate final confidence
            if factors:
                return sum(factors)
            return 0.5  # Default moderate confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    async def generate_response(
        self,
        query: str,
        context: RAGContext,
        stream: bool = False
    ) -> str:
        """Generate final response using LLM with context"""
        
        # Create prompt with context
        prompt = self._create_prompt(query, context)
        
        # Generate response
        if stream:
            return self._stream_response(prompt)
        else:
            response = self.llm.invoke(prompt)
            return response.content
    
    def _create_prompt(self, query: str, context: RAGContext) -> str:
        """Create detailed prompt with all context"""
        
        # Format retrieved documents
        doc_context = "\n\n".join([
            f"Source {i+1}: {doc.page_content[:500]}"
            for i, doc in enumerate(context.relevant_docs)
        ])
        
        # Format market data
        market_context = json.dumps(context.market_data, indent=2) if context.market_data else "No market data available"
        
        # Format Graph RAG results
        graph_context = ""
        if context.graph_results:
            graph_context = "\n\nGraph Knowledge Base Results:\n"
            for i, result in enumerate(context.graph_results[:5]):  # Top 5 results
                node_data = result['data']
                graph_context += f"Node {i+1}: {result['node']} ({result['type']})\n"
                graph_context += f"  Type: {node_data.get('type', 'unknown')}\n"
                if 'name' in node_data:
                    graph_context += f"  Name: {node_data['name']}\n"
                if 'sector' in node_data:
                    graph_context += f"  Sector: {node_data['sector']}\n"
                if 'price' in node_data:
                    graph_context += f"  Price: {node_data['price']}\n"
                if 'change_percent' in node_data:
                    graph_context += f"  Change: {node_data['change_percent']}%\n"
                graph_context += f"  Relevance: {result['relevance_score']:.2f}\n\n"
        
        # Format technical indicators
        tech_context = json.dumps(context.technical_indicators, indent=2) if context.technical_indicators else "No technical data available"
        
        prompt = f"""You are an expert Indian market investment advisor. Answer the following query using the provided context.

Query: {query}
Query Type: {context.query_type.value}

Retrieved Knowledge:
{doc_context}

Graph Knowledge Base:
{graph_context}

Current Market Data:
{market_context}

Technical Indicators:
{tech_context}

Market Sentiment: {context.sentiment_score}

Instructions:
1. Provide a comprehensive answer based on the context
2. Include specific data points and numbers
3. Give actionable recommendations
4. Mention any risks or considerations
5. Be specific to the Indian market context
6. Use INR for currency references

Response:"""
        
        return prompt
    
    async def _stream_response(self, prompt: str):
        """Stream response from LLM"""
        async for chunk in self.llm.astream(prompt):
            yield chunk.content
    
    async def process_query(self, query: str, stream: bool = False) -> Dict:
        """Enhanced query processing with graph and vector integration"""
        query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[RAG-{query_id}] Processing query: {query}")
        
        # Classify query and extract entities
        query_type = self.classify_query(query)
        entities = self.extract_entities(query)
        logger.info(f"[RAG-{query_id}] Query type: {query_type}, Entities: {entities}")
        
        # Get graph-based results
        logger.debug(f"[RAG-{query_id}] Querying knowledge graph")
        graph_results = self._graph_query(query_type, entities)
        logger.info(f"[RAG-{query_id}] Found {len(graph_results)} graph nodes")
        
        # Get vector-based results
        logger.debug(f"[RAG-{query_id}] Retrieving context from vector store")
        relevant_docs = await self.retrieve_context(query, query_type, k=5)
        logger.info(f"[RAG-{query_id}] Retrieved {len(relevant_docs)} relevant documents")
        
        # Get market context and technical indicators
        logger.debug(f"[RAG-{query_id}] Extracting market context and indicators from graph")
        market_context = {}
        technical_indicators = {} # <-- Initialize as empty dict
        
        for result in graph_results:
            if result['type'] in ['market_context', 'sector_info']:
                market_context[result['node']] = result['data']
            
            # --- START OF FIX ---
            # Check if this is a stock node and extract its indicators
            if result['type'] == 'stock_info' and 'data' in result:
                stock_data = result['data']
                stock_symbol = result['node']
                
                # Extract indicators stored as attributes
                indicators = {
                    'rsi': stock_data.get('rsi'),
                    'macd': stock_data.get('macd'),
                    'sma_20': stock_data.get('sma_20'),
                    'sma_50': stock_data.get('sma_50')
                }
                # Add to the main dict, filtering out None values
                technical_indicators[stock_symbol] = {k: v for k, v in indicators.items() if v is not None}
            # --- END OF FIX ---

        # Calculate scores
        sentiment_score = self._calculate_aggregate_sentiment(graph_results)
        confidence_score = self._calculate_confidence_score(graph_results, relevant_docs)
        logger.info(f"[RAG-{query_id}] Sentiment: {sentiment_score:.2f}, Confidence: {confidence_score:.2f}")
        
        # Create context
        context = RAGContext(
            query=query,
            query_type=query_type,
            relevant_docs=relevant_docs,
            market_data=market_context,
            technical_indicators=technical_indicators, # <-- Pass the populated dict
            sentiment_score=sentiment_score,
            confidence_score=confidence_score,
            graph_results=graph_results
        )
        
        # Generate response
        logger.info(f"[RAG-{query_id}] Generating response using LLM")
        response = await self.generate_response(query, context, stream)
        
        logger.info(f"[RAG-{query_id}] Query processing completed")
        return {
            'query': query,
            'response': response,
            'entities': entities,
            'graph_results': [r['node'] for r in graph_results],
            'vector_results': [doc.metadata.get('source') for doc in relevant_docs],
            'confidence': context.confidence_score,
            'timestamp': datetime.now().isoformat()
        }    
    def _graph_query(self, query_type: QueryType, entities: Dict[str, List[str]], max_depth: int = 2) -> List[Dict]:
        """Execute graph-based queries based on query type and entities"""
        results = []
        seen_nodes = set()

        try:
            # Get starting nodes based on entities
            start_nodes = []
            for company in entities.get('companies', []):
                if self.knowledge_graph.has_node(company):
                    start_nodes.append(company)
            
            for sector_name in entities.get('sectors', []):
                matching_sectors = [
                    node for node in self.knowledge_graph.nodes()
                    if self.knowledge_graph.nodes[node].get('type') == 'sector'
                    and sector_name.lower() in node.lower()
                ]
                start_nodes.extend(matching_sectors)
            
            # Add general market context
            if self.knowledge_graph.has_node('market_breadth') and 'market_breadth' not in seen_nodes:
                results.append({
                    'node': 'market_breadth',
                    'data': self.knowledge_graph.nodes['market_breadth'],
                    'type': 'market_context',
                    'relevance_score': 0.7
                })
                seen_nodes.add('market_breadth')

            # Perform graph traversal
            for node in start_nodes:
                if node in seen_nodes:
                    continue
                
                node_data = self.knowledge_graph.nodes[node]
                node_type = node_data.get('type')
                
                # Add the starting node
                results.append({
                    'node': node,
                    'data': node_data,
                    'type': f"{node_type}_info",
                    'relevance_score': 1.0
                })
                seen_nodes.add(node)
                
                # --- START OF IMPROVED 2-HOP TRAVERSAL ---
                
                # Get direct neighbors (1-hop)
                for neighbor in self.knowledge_graph.neighbors(node):
                    if neighbor in seen_nodes:
                        continue
                        
                    neighbor_data = self.knowledge_graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get('type')

                    # 1. If Stock -> Get related News and its Sector
                    if node_type == 'stock':
                        if neighbor_type == 'news':
                            results.append({
                                'node': neighbor,
                                'data': neighbor_data,
                                'type': 'related_news',
                                'relevance_score': 0.8
                            })
                            seen_nodes.add(neighbor)
                        
                        elif neighbor_type == 'sector':
                            # This is the 2-hop link. Add the sector.
                            if neighbor not in seen_nodes:
                                results.append({
                                    'node': neighbor,
                                    'data': neighbor_data,
                                    'type': 'sector_info',
                                    'relevance_score': 0.9
                                })
                                seen_nodes.add(neighbor)
                            
                            # 2. (2-HOP) From Sector -> Get other stocks
                            for sector_neighbor in self.knowledge_graph.neighbors(neighbor):
                                if sector_neighbor in seen_nodes or sector_neighbor == node:
                                    continue
                                
                                sector_neighbor_data = self.knowledge_graph.nodes[sector_neighbor]
                                if sector_neighbor_data.get('type') == 'stock':
                                    results.append({
                                        'node': sector_neighbor,
                                        'data': sector_neighbor_data,
                                        'type': 'peer_stock', # Peer comparison
                                        'relevance_score': 0.6
                                    })
                                    seen_nodes.add(sector_neighbor)
            
                    # 1. If Sector -> Get Stocks in sector
                    elif node_type == 'sector':
                        if neighbor_type == 'stock':
                            if neighbor not in seen_nodes:
                                results.append({
                                    'node': neighbor,
                                    'data': neighbor_data,
                                    'type': 'sector_stock',
                                    'relevance_score': 0.9
                                })
                                seen_nodes.add(neighbor)
                # --- END OF IMPROVED 2-HOP TRAVERSAL ---

            # Deduplicate and sort results
            final_results = []
            final_seen = set()
            for res in results:
                if res['node'] not in final_seen:
                    final_results.append(res)
                    final_seen.add(res['node'])
            
            final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return final_results[:20] # Limit to top 20 relevant nodes
            
        except Exception as e:
            logger.error(f"Error in graph query: {e}")
            return []
# Initialize the RAG system
if __name__ == "__main__":
    print("Initializing Advanced RAG System with Graph RAG and File-based Storage...")
    
    # Initialize with file-based storage (Redis optional)
    rag_system = AdvancedRAGSystem(
        vector_db_type="chroma",
        enable_redis=False  # Use file-based storage
    )
    
    print(f"Knowledge Graph: {rag_system.knowledge_graph.number_of_nodes()} nodes, {rag_system.knowledge_graph.number_of_edges()} edges")
    print(f"Loaded Data: {len(rag_system.file_data['stocks'])} stocks, {len(rag_system.file_data['news_data'])} news articles")
    
    # Example usage
    async def test_query():
        queries = [
            "What are the best mid-cap stocks to invest in for next 6 months considering current market conditions?",
            "How is Reliance Industries performing and what are the technical indicators?",
            "What is the current market sentiment and sector performance?",
            "Should I invest in HDFC Bank given the RBI policy changes?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*50}")
            print(f"Query {i}: {query}")
            print(f"{'='*50}")
            
            result = await rag_system.process_query(query)
            
            print(f"Response: {result['response'][:500]}...")
            print(f"Entities: {result['entities']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Graph Results: {len(result.get('graph_results', []))} nodes found")
    
    asyncio.run(test_query())