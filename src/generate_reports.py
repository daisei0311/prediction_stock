import os
import sys
import json
import datetime
import glob
import requests
import pandas as pd
import yfinance as yf
import yaml
from settings import OUTPUT_DIR, RAW_DATA_DIR, CONFIG_FILE

# Deepseek API Configuration
def load_api_key():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('deepseek_api_key')
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or load_api_key()
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-reasoner"

def load_latest_forecast(date_str=None):
    """
    Load the forecast CSV for the given date or the latest available.
    """
    if date_str:
        file_path = os.path.join(OUTPUT_DIR, f"forecast_{date_str}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Forecast file not found: {file_path}")
    else:
        # Find the latest forecast file
        files = glob.glob(os.path.join(OUTPUT_DIR, "forecast_*.csv"))
        if not files:
            raise FileNotFoundError("No forecast files found in output directory.")
        file_path = max(files, key=os.path.getctime)
        print(f"Using latest forecast file: {file_path}")

    return pd.read_csv(file_path)

def get_top_tickers(df, n=5):
    """
    Get the top n tickers from the forecast dataframe.
    Assumes 'Rank' or 'Score' column exists.
    """
    if "Rank" in df.columns:
        return df.sort_values("Rank").head(n)["Code"].tolist()
    elif "Score" in df.columns:
        return df.sort_values("Score", ascending=False).head(n)["Code"].tolist()
    else:
        raise ValueError("Forecast dataframe must have 'Rank' or 'Score' column.")

def get_stock_data_summary(ticker):
    """
    Get stock data summary for the prompt.
    Reads from local CSV for price history and yfinance for financial metrics.
    """
    # 1. Price History (Last 2 months) from local CSV
    csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    price_summary = "No local price data available."
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Filter last 2 months
            latest_date = df['Date'].max()
            two_months_ago = latest_date - pd.DateOffset(months=2)
            recent_df = df[df['Date'] >= two_months_ago]
            
            # Create a summary string (Date, Close, Volume, and maybe some indicators if present)
            # We'll just provide the raw rows for the model to analyze
            price_summary = recent_df.to_csv(index=False)
        except Exception as e:
            price_summary = f"Error reading local price data: {e}"

    # 2. Financial Metrics & News from yfinance
    financial_summary = "No financial data available."
    news_summary = "No recent news available."
    company_name = ticker
    
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        company_name = info.get('longName', ticker)
        
        # Extract specific metrics requested
        metrics = {
            'company_name': company_name,
            'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            'total_revenue': info.get('totalRevenue'),
            'net_income_to_common': info.get('netIncomeToCommon'),
            'revenue_growth': info.get('revenueGrowth'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry')
        }
        
        # Get 5 years of historical financials
        historical_financials = {}
        try:
            # Financials (Income Statement)
            fin = yf_ticker.financials
            if not fin.empty:
                # Take last 5 columns (years) and convert Timestamp to string
                cols = fin.columns[:5]
                fin_subset = fin[cols].copy()
                fin_subset.columns = [str(col.date()) if hasattr(col, 'date') else str(col) for col in fin_subset.columns]
                historical_financials['income_statement'] = fin_subset.to_dict()
            
            # Balance Sheet
            bs = yf_ticker.balance_sheet
            if not bs.empty:
                cols = bs.columns[:5]
                bs_subset = bs[cols].copy()
                bs_subset.columns = [str(col.date()) if hasattr(col, 'date') else str(col) for col in bs_subset.columns]
                historical_financials['balance_sheet'] = bs_subset.to_dict()
                
            # Cash Flow
            cf = yf_ticker.cashflow
            if not cf.empty:
                cols = cf.columns[:5]
                cf_subset = cf[cols].copy()
                cf_subset.columns = [str(col.date()) if hasattr(col, 'date') else str(col) for col in cf_subset.columns]
                historical_financials['cash_flow'] = cf_subset.to_dict()
                
        except Exception as e:
            print(f"Warning: Error fetching historical financials for {ticker}: {e}")
            historical_financials['error'] = str(e)

        financial_summary = json.dumps({
            "current_metrics": metrics, 
            "historical_financials": historical_financials
        }, indent=2, default=str)
        
        # News
        try:
            news = yf_ticker.news
            if news:
                news_items = []
                for item in news[:5]: # Top 5 news
                    news_items.append(f"- {item.get('title')} ({item.get('publisher')}) - {item.get('link')}")
                news_summary = "\n".join(news_items)
        except Exception as e:
            print(f"Warning: Error fetching news for {ticker}: {e}")
            news_summary = f"Error fetching news: {e}"
            
    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        financial_summary = f"Error fetching yfinance data: {e}"
        news_summary = f"Error fetching news: {e}"

    return price_summary, financial_summary, news_summary, company_name

def generate_report(ticker, company_name, price_data, financial_data, news_data):
    """
    Call Deepseek API to generate the report.
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

    system_prompt = f"""
あなたは、企業（{company_name}、シンボル: {ticker}）のパフォーマンスを株価、テクニカル指標、および財務指標に基づいて評価することを専門とするファンダメンタルアナリストです。あなたのタスクは、指定された株式のファンダメンタル分析に関する包括的な要約を提供することです。

使用可能なツール（以下のデータが提供されています）：
1. **get_stock_prices**: 最新の株価、過去の価格データ（直近2ヶ月）、およびテクニカル指標。
2. **get_financial_metrics**: 売上高、EPS、P/E、負債比率、ROE、ROA、成長率などの主要な財務指標。また、過去5年間の財務諸表データ（損益計算書、貸借対照表、キャッシュフロー計算書）も含まれています。
3. **news**: 企業の最新ニュース。

### あなたのタスク：
1. **データを分析する**: 提供されたデータ（株価、財務、ニュース）を評価し、潜在的な抵抗線、主要なトレンド、強み、または懸念点を特定します。特に過去5年間の財務データのトレンド（売上、利益の成長など）に注目してください。
2. **要約を提供する**: 以下を強調する簡潔で構造化された要約を作成します：
   - 最近の株価動向、トレンド、および潜在的な抵抗線。
   - テクニカル指標から得られる重要な洞察。
   - 財務指標に基づく財務の健全性とパフォーマンス（過去5年のトレンドを含む）。
   - Webサイトから見つけたニュースに基づくセンチメントや影響。

### 制約条件：
 - 提供されたデータのみを使用してください。
 - ニュースデータをWebサイトから必ず引っ張り、自分なりの考え（投資判断について）を述べること。
 - データが不足している場合、その旨を要約で明確に記載してください。

### 出力フォーマット：
以下のJSON形式で応答してください（Markdownコードブロックなしで、純粋なJSONとして）：
{{
"stock": "{ticker}",
"company_name": "{company_name}",
"price_analysis": "<株価動向の詳細な分析>",
"technical_analysis": "<全てのテクニカル指標に基づく時系列分析の詳細>",
"financial_analysis": "<財務指標（5年間のトレンド含む）に基づく詳細な分析>",
"final Summary": "<上記の分析に基づく総合的な結論>",
"Asked Question Answer": "<上記の詳細と分析に基づく質問への回答>"
}}

応答は客観的で簡潔、かつ実用的なものにしてください。
"""

    user_content = f"""
Please generate the report for {company_name} ({ticker}) based on the following data:

[Stock Prices (Last 2 Months)]
{price_data}

[Financial Metrics & 5-Year History]
{financial_data}

[Recent News]
{news_data}
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "stream": True
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        full_content = ""
        reasoning_content = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            # Handle reasoning content (for deepseek-reasoner)
                            if 'reasoning_content' in delta and delta['reasoning_content']:
                                reasoning_content += delta['reasoning_content']
                            # Handle regular content
                            if 'content' in delta and delta['content']:
                                full_content += delta['content']
                    except json.JSONDecodeError:
                        continue
        
        # For deepseek-reasoner, we only want the final content, not the reasoning
        return full_content if full_content else reasoning_content
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating report: {e}"

def save_report(date_str, ticker, report_content):
    """
    Save the report to data/output/{date_str}/{ticker}_report.txt
    """
    # Create directory if not exists
    report_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(report_dir, exist_ok=True)
    
    file_path = os.path.join(report_dir, f"{ticker}_report.txt") # Changed to .txt to store raw output (likely JSON)
    
    # If the content is JSON, maybe we want to save as .json?
    # The prompt asks for JSON format output.
    # But the user said "5 files will be created".
    # I'll save as .json if it looks like json, otherwise .txt
    
    ext = ".txt"
    try:
        json.loads(report_content)
        ext = ".json"
    except:
        pass
        
    file_path = os.path.join(report_dir, f"{ticker}_report{ext}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Saved report for {ticker} to {file_path}")

def main():
    try:
        # Determine date
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Load forecast
        print(f"Loading forecast data...")
        df = load_latest_forecast() # Uses latest if not specified, or we could pass today_str
        
        # Get top 5
        top_tickers = get_top_tickers(df, n=5)
        print(f"Top 5 Tickers: {top_tickers}")
        
        for ticker in top_tickers:
            print(f"Processing {ticker}...")
            
            # Get Data
            price_data, financial_data, news_data, company_name = get_stock_data_summary(ticker)
            
            # Generate Report
            print(f"Generating report for {company_name} ({ticker}) with Deepseek AI...")
            report = generate_report(ticker, company_name, price_data, financial_data, news_data)
            
            # Save Report
            save_report(today_str, ticker, report)
            
        print("All reports generated successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
