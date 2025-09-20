import os
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time

# --- Configuration ---
load_dotenv()
DATA_DIR = "data"
ASSETS = ["^NSEI", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "GOLDBEES.NS"]
START_DATE = "2010-01-01"

# --- API Keys & URLs ---
# IMPORTANT: Add your NewsAPI key to your .env file. You can get a free key from newsapi.org
# NEWS_API_KEY="your_key_here"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
RBI_DATA_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=54006" # Example URL, needs to be updated for real-time data
SEBI_FILINGS_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=19&smid=13" # Insider trading


def fetch_yfinance_data(symbol, start_date):
    """Fetches historical price data from Yahoo Finance."""
    print(f"Fetching historical price data for {symbol}...")
    try:
        df = yf.download(symbol, start=start_date, progress=False)
        if df.empty:
            print(f"  - No price data found for {symbol}.")
            return None
        df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Adj Close": "adjusted_close", "Volume": "volume"
        }, inplace=True)
        return df
    except Exception as e:
        print(f"  - Error fetching price data for {symbol}: {e}")
        return None

def fetch_fundamental_data(symbol):
    """Fetches fundamental data (financials, balance sheet) for a stock."""
    print(f"Fetching fundamental data for {symbol}...")
    if ".NS" not in symbol: # Only fetch for stocks, not indices
        print(f"  - Skipping fundamentals for index {symbol}.")
        return
        
    try:
        ticker = yf.Ticker(symbol)
        
        # Quarterly Financials
        financials = ticker.quarterly_financials
        if not financials.empty:
            financials.to_csv(os.path.join(DATA_DIR, f"{symbol}_financials.csv"))
            print(f"  - Saved quarterly financials for {symbol}.")
        
        # Quarterly Balance Sheet
        balance_sheet = ticker.quarterly_balance_sheet
        if not balance_sheet.empty:
            balance_sheet.to_csv(os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv"))
            print(f"  - Saved quarterly balance sheet for {symbol}.")
            
        time.sleep(1) # Be respectful to the API
    except Exception as e:
        print(f"  - Error fetching fundamental data for {symbol}: {e}")

def fetch_news_data(query):
    """Fetches news articles related to a query from NewsAPI."""
    print(f"Fetching news for '{query}'...")
    if not NEWS_API_KEY:
        print("  - NEWS_API_KEY not found in .env. Skipping news fetch.")
        return
        
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        if articles:
            news_df = pd.DataFrame(articles)
            news_df = news_df[['title', 'description', 'content', 'publishedAt', 'url']]
            news_df.to_csv(os.path.join(DATA_DIR, f"news_{query.replace(' ', '_')}.csv"), index=False)
            print(f"  - Saved {len(articles)} news articles for '{query}'.")
    except requests.exceptions.RequestException as e:
        print(f"  - Error fetching news for {query}: {e}")

def scrape_rbi_data():
    """Scrapes key macroeconomic indicators from RBI. (Demonstration purposes)"""
    print("Scraping RBI data...")
    try:
        # NOTE: This is a simplified example. Real-world scraping of RBI needs robust parsers.
        response = requests.get(RBI_DATA_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the relevant table or text. This will change based on the RBI website structure.
        # For this example, we'll just save the text content.
        content = soup.get_text()
        with open(os.path.join(DATA_DIR, "rbi_macro_data.txt"), "w", encoding="utf-8") as f:
            f.write(content)
        print("  - Saved RBI page content.")
    except Exception as e:
        print(f"  - Error scraping RBI data: {e}")


def run_ingestion():
    """Runs the full, multi-source data ingestion pipeline."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    
    # 1. Fetch Price and Fundamental Data for all assets
    for asset in ASSETS:
        sanitized_asset_name = asset.replace('^', '')
        
        # Price Data
        price_df = fetch_yfinance_data(asset, START_DATE)
        if price_df is not None:
            price_df.to_csv(os.path.join(DATA_DIR, f"{sanitized_asset_name}.csv"))
            print(f"  - Successfully saved price data for {asset}")
        
        # Fundamental Data
        fetch_fundamental_data(sanitized_asset_name)

    # 2. Fetch News Data for general market and specific companies
    news_queries = ["Indian stock market", "Nifty 50", "Reliance Industries", "HDFC Bank", "Infosys", "TCS"]
    for query in news_queries:
        fetch_news_data(query)
        time.sleep(1) # Be respectful to the API

    # 3. Fetch Macroeconomic Data
    scrape_rbi_data()
    
    # 4. Scrape SEBI Filings (placeholder for a more complex scraper)
    # A real implementation would require a dedicated scraper for the SEBI website.
    print("SEBI filings scraping would be implemented here in a full-scale version.")

    print("\n--- Data Ingestion Complete ---")


if __name__ == "__main__":
    run_ingestion()

