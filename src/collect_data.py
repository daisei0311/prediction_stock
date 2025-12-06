import yfinance as yf
import pandas as pd
import os
import datetime
import json
from settings import DATA_DIR, TICKERS_CONFIG_FILE, RAW_DATA_DIR

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

# 保存先ディレクトリ (settings.pyから取得)
# DATA_DIR = "./data/raw"  <-- settings.pyのRAW_DATA_DIRを使う

# ティッカー設定ファイル (settings.pyから取得)
# TICKERS_CONFIG_FILE = "tickers.yaml"

# 期間設定
# 期間設定
START_DATE = "2015-01-01"
# END_DATEは明日（今日を含むため）
END_DATE = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

# ---------------------------------------------------------
# Ticker設定の読み込み
# ---------------------------------------------------------
def load_tickers_config():
    """
    tickers.yamlから銘柄設定を読み込む
    Returns:
        dict: ティッカー設定
    """
    import yaml
    
    if not os.path.exists(TICKERS_CONFIG_FILE):
        print(f"エラー: {TICKERS_CONFIG_FILE} が見つかりません")
        return None
    
    with open(TICKERS_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_tickers():
    """
    YAML設定からティッカーリストと名前マッピングを準備
    Returns:
        tuple: (ALL_TICKERS, INDICES, SECTOR_ETFS, COMPANY_NAMES, CORE30_TICKERS, MY_WATCHLIST)
    """
    config = load_tickers_config()
    if not config:
        return [], {}, {}, {}, [], []
    
    # 個別銘柄リスト
    core30 = list(config.get('core30', {}).keys())
    watchlist = list(config.get('watchlist', {}).keys())
    all_tickers = list(set(core30 + watchlist))
    
    # 指数マッピング
    indices_config = config.get('indices', {})
    indices = {}
    for ticker, info in indices_config.items():
        # ファイル名として使用する名前（name_enの最初の単語など）
        if ticker == '^N225':
            indices[ticker] = 'Nikkei225'
        elif ticker == 'USDJPY=X':
            indices[ticker] = 'USDJPY'
        elif ticker == '^TNX':
            indices[ticker] = 'US10Y'
        elif ticker == '^GSPC':
            indices[ticker] = 'SP500'
        elif ticker == '^VIX':
            indices[ticker] = 'VIX'
    
    # セクターマッピング
    sectors_config = config.get('sectors', {})
    sector_etfs = {}
    for ticker, info in sectors_config.items():
        # ファイル名として使用する名前
        if ticker == '1615.T':
            sector_etfs[ticker] = 'Sector_Bank'
        elif ticker == '1622.T':
            sector_etfs[ticker] = 'Sector_Auto'
        elif ticker == '1625.T':
            sector_etfs[ticker] = 'Sector_Electric'
        elif ticker == '1621.T':
            sector_etfs[ticker] = 'Sector_Pharma'
        elif ticker == '1629.T':
            sector_etfs[ticker] = 'Sector_Wholesale'
        elif ticker == '1623.T':
            sector_etfs[ticker] = 'Sector_Steel'
        elif ticker == '1626.T':
            sector_etfs[ticker] = 'Sector_Comm'
    
    # 会社名マッピング（全ての銘柄）
    company_names = {}
    for category in ['core30', 'watchlist', 'indices', 'sectors']:
        for ticker, info in config.get(category, {}).items():
            company_names[ticker] = {
                'name': info.get('name', ticker),
                'name_en': info.get('name_en', ticker)
            }
    
    return all_tickers, indices, sector_etfs, company_names, core30, watchlist

# ティッカー設定を読み込む
ALL_TICKERS, INDICES, SECTOR_ETFS, COMPANY_NAMES, CORE30_TICKERS, MY_WATCHLIST = setup_tickers()

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def create_directory(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"フォルダを作成しました: {directory}")
    else:
        print(f"フォルダを確認しました: {directory}")

def download_and_save(ticker, name, start, end):
    """
    データをダウンロードしてCSV保存する（差分更新対応）
    Args:
        ticker: ティッカーシンボル
        name: 保存ファイル名（拡張子なし）
        start: 開始日（新規ダウンロード時のみ使用）
        end: 終了日
    """
    save_path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
    
    # ファイル存在チェック
    if os.path.exists(save_path):
        # ケースB: 既存ファイルがある場合、差分更新
        try:
            # 既存データを読み込み
            # まずは普通に読み込む
            try:
                existing_df = pd.read_csv(save_path, index_col=0)
            except:
                print(f"[{ticker}] 既存ファイル読み込み失敗。再ダウンロードします。")
                download_full(ticker, name, start, end, save_path)
                return

            # インデックス名が 'Date' でない、またはカラムに "('" が含まれる場合は壊れている/古い形式とみなす
            is_corrupted = False
            
            # 古いyfinance形式 (3行ヘッダー) のチェック
            if existing_df.index.name != 'Date':
                 try:
                     # 3行スキップで読み直してみる
                     temp_df = pd.read_csv(save_path, skiprows=[1, 2], index_col=0)
                     if temp_df.index.name == 'Date':
                         existing_df = temp_df
                     else:
                         # それでもダメなら壊れている可能性
                         is_corrupted = True
                 except:
                     is_corrupted = True
            
            # カラム名のクリーニングとチェック
            if not is_corrupted:
                # 不要なカラム（タプル文字列表現など）を特定
                valid_cols = []
                for col in existing_df.columns:
                    col_str = str(col)
                    if "('" not in col_str and "')" not in col_str and "Unnamed" not in col_str:
                        valid_cols.append(col)
                
                if len(valid_cols) < len(existing_df.columns):
                    print(f"[{ticker}] 不要なカラムを削除してクリーンアップします。")
                    existing_df = existing_df[valid_cols]
                
                # 必須カラムがあるか確認
                if 'Close' not in existing_df.columns:
                    is_corrupted = True
                else:
                    # CloseがNaNの行を削除 (データ欠損行のクリーニング)
                    # これにより、前回不完全な状態で保存された行があれば削除され、再ダウンロード対象になる
                    existing_df = existing_df.dropna(subset=['Close'])

            if is_corrupted:
                print(f"[{ticker}] ファイル形式が不正のため、全期間再ダウンロードします。")
                download_full(ticker, name, start, end, save_path)
                return

            # インデックスをDatetimeIndexに変換
            existing_df.index = pd.to_datetime(existing_df.index)
            
            # タイムゾーン情報を削除（比較のため）
            if hasattr(existing_df.index, 'tz') and existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)
            
            # 最終データの日付を取得
            last_date = existing_df.index.max()
            
            # 翌日から現在までをダウンロード
            next_day = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # APIリミット回避: 最終データが今日の日付と同じならスキップ
            today = datetime.datetime.now().date()
            if last_date.date() >= today:
                print(f"[{ticker}] 最新データ済み (最終日: {last_date.date()}) - スキップ")
                return
            
            print(f"[{ticker}] 差分更新中... (最終日: {last_date.date()})")
            
            # 差分データをダウンロード
            df_new = yf.download(ticker, start=next_day, end=end, progress=False)
            
            if df_new.empty:
                print(f"差分なし（最新）: {save_path}")
                # クリーンアップしたかもしれないので保存し直す
                existing_df.to_csv(save_path)
                return
            
            # MultiIndexカラムをフラット化 (例: ('Close', '7203.T') -> 'Close')
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)

            # タイムゾーン情報を削除
            if hasattr(df_new.index, 'tz') and df_new.index.tz is not None:
                df_new.index = df_new.index.tz_localize(None)
            
            # 必要な列のみ抽出
            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            existing_cols = [c for c in required_cols if c in df_new.columns]
            df_new = df_new[existing_cols]

            # 既存データと結合
            df = pd.concat([existing_df, df_new])
            
            # 重複排除（最新のデータを優先）
            df = df[~df.index.duplicated(keep='last')]
            
            # 日付でソート
            df = df.sort_index()
            
            # 数値列を丸める (5桁)
            df = df.round(5)
            
            # CSV保存
            df.to_csv(save_path)
            print(f"{len(df_new)}日分追加更新: {save_path}")
            
        except Exception as e:
            print(f"差分更新エラー [{ticker}]: {e}")
            print(f"全期間再ダウンロードします...")
            # エラー時は全期間再ダウンロード
            download_full(ticker, name, start, end, save_path)
    else:
        # ケースA: ファイルがない場合、新規ダウンロード
        download_full(ticker, name, start, end, save_path)

def download_full(ticker, name, start, end, save_path):
    """
    全期間のデータをダウンロードして保存
    """
    print(f"[{ticker}] データを取得中...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"警告: データが取得できませんでした。ティッカー={ticker}")
            return
        
        # MultiIndexカラムをフラット化 (例: ('Close', '7203.T') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # タイムゾーン情報を削除
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 必要な列のみ抽出
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        # 数値列を丸める (5桁)
        df = df.round(5)
        
        # CSV保存
        df.to_csv(save_path)
        print(f"新規ダウンロード完了: {save_path}")
        
    except Exception as e:
        print(f"エラー: {ticker} のダウンロードに失敗しました。{e}")
        print(f"詳細: {e}")

def is_market_open():
    """
    市場が開いているか判定する
    東京証券取引所の営業時間: 平日 9:00-15:00 (JST)
    Returns:
        bool: 市場が開いていればTrue
    """
    import pytz
    from datetime import datetime
    
    # 日本時間を取得
    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.now(jst)
    
    # 土日は休場
    if now.weekday() >= 5:  # 5=土曜日, 6=日曜日
        return False
    
    # 営業時間チェック (9:00-15:00)
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def save_latest_market_data():
    """
    全てのティッカーの最新データをJSONファイルに保存
    ダッシュボード用のデータを生成
    """
    print("\n--- 最新市場データの保存 ---")
    
    latest_data = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": [],
        "indices": [],
        "sectors": []
    }
    
    # 個別銘柄データの取得
    print("個別銘柄の最新データを取得中...")
    for ticker in ALL_TICKERS:
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0)
                if not df.empty:
                    latest = df.iloc[-1]
                    latest_date = df.index[-1]
                    
                    # 前日比を計算（2行以上ある場合）
                    change_pct = 0.0
                    if len(df) >= 2:
                        prev_close = df.iloc[-2]['Close']
                        curr_close = latest['Close']
                        change_pct = ((curr_close - prev_close) / prev_close) * 100
                    
                    # Volumeの処理: NaNの場合は0にする
                    volume_val = 0
                    if 'Volume' in latest and pd.notna(latest['Volume']):
                        volume_val = int(latest['Volume'])
                    
                    ticker_data = {
                        "code": ticker,
                        "name": COMPANY_NAMES.get(ticker, {}).get('name', ticker),
                        "name_en": COMPANY_NAMES.get(ticker, {}).get('name_en', ticker),
                        "date": latest_date,
                        "price": round(float(latest['Close']), 5),
                        "open": round(float(latest['Open']), 5),
                        "high": round(float(latest['High']), 5),
                        "low": round(float(latest['Low']), 5),
                        "volume": volume_val,
                        "change_pct": round(change_pct, 5)
                    }
                    latest_data["tickers"].append(ticker_data)
            except Exception as e:
                print(f"警告: {ticker} のデータ読み込みエラー: {e}")
    
    # 指数データの取得
    print("指数データの最新データを取得中...")
    for ticker, name in INDICES.items():
        csv_path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0)
                if not df.empty:
                    latest = df.iloc[-1]
                    latest_date = df.index[-1]
                    
                    # 前日比を計算
                    change_pct = 0.0
                    if len(df) >= 2:
                        prev_close = df.iloc[-2]['Close']
                        curr_close = latest['Close']
                        change_pct = ((curr_close - prev_close) / prev_close) * 100
                    
                    index_data = {
                        "ticker": ticker,
                        "name": COMPANY_NAMES.get(ticker, {}).get('name', name),
                        "name_en": COMPANY_NAMES.get(ticker, {}).get('name_en', name),
                        "date": latest_date,
                        "price": round(float(latest['Close']), 5),
                        "open": round(float(latest['Open']), 5),
                        "high": round(float(latest['High']), 5),
                        "low": round(float(latest['Low']), 5),
                        "change_pct": round(change_pct, 5)
                    }
                    latest_data["indices"].append(index_data)
            except Exception as e:
                print(f"警告: {name} のデータ読み込みエラー: {e}")
    
    # セクターデータの取得
    print("セクターデータの最新データを取得中...")
    for ticker, name in SECTOR_ETFS.items():
        csv_path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0)
                if not df.empty:
                    latest = df.iloc[-1]
                    latest_date = df.index[-1]
                    
                    # 前日比を計算
                    change_pct = 0.0
                    if len(df) >= 2:
                        prev_close = df.iloc[-2]['Close']
                        curr_close = latest['Close']
                        change_pct = ((curr_close - prev_close) / prev_close) * 100
                    
                    # Volumeの処理: NaNの場合は0にする
                    volume_val = 0
                    if 'Volume' in latest and pd.notna(latest['Volume']):
                        volume_val = int(latest['Volume'])
                    
                    sector_data = {
                        "ticker": ticker,
                        "name": COMPANY_NAMES.get(ticker, {}).get('name', name),
                        "name_en": COMPANY_NAMES.get(ticker, {}).get('name_en', name),
                        "date": latest_date,
                        "price": round(float(latest['Close']), 5),
                        "open": round(float(latest['Open']), 5),
                        "high": round(float(latest['High']), 5),
                        "low": round(float(latest['Low']), 5),
                        "volume": volume_val,
                        "change_pct": round(change_pct, 5)
                    }
                    latest_data["sectors"].append(sector_data)
            except Exception as e:
                print(f"警告: {name} のデータ読み込みエラー: {e}")
    
    # JSONファイルに保存
    output_path = os.path.join(RAW_DATA_DIR, "latest_market_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(latest_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n最新市場データを保存しました: {output_path}")
    print(f"  - 個別銘柄: {len(latest_data['tickers'])}件")
    print(f"  - 指数: {len(latest_data['indices'])}件")
    print(f"  - セクター: {len(latest_data['sectors'])}件")

def export_company_names():
    """
    会社名マッピングをJSON形式でエクスポート
    """
    output_path = os.path.join(RAW_DATA_DIR, "company_names.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(COMPANY_NAMES, f, ensure_ascii=False, indent=2)
    
    print(f"会社名マッピングを保存しました: {output_path}")
    print(f"  - 総銘柄数: {len(COMPANY_NAMES)}件")

def detect_new_companies():
    """
    company_names.jsonに存在するが、CSVファイルが存在しない個別銘柄を検出
    これらは新規追加された銘柄として扱い、2015年からの全データを取得する
    
    注: 指数やセクターETFは既に差分更新されているため、個別銘柄のみをチェック
    
    Returns:
        list: 新規追加された個別銘柄のティッカーリスト
    """
    new_companies = []
    
    # 個別銘柄のみチェック
    for ticker in ALL_TICKERS:
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            new_companies.append(ticker)
    
    return new_companies

def download_new_company_data(new_companies, start_date, end_date):
    """
    新規追加された個別銘柄のデータを2015年から取得
    
    Args:
        new_companies: detect_new_companies()の戻り値（ティッカーのリスト）
        start_date: 開始日（通常は2015-01-01）
        end_date: 終了日
    """
    if len(new_companies) == 0:
        print("新規追加された銘柄はありません。")
        return
    
    print(f"\n{'='*60}")
    print(f"新規追加銘柄を検出しました: {len(new_companies)}件")
    print(f"これらの銘柄は{start_date}からの全データを取得します。")
    print(f"{'='*60}\n")
    
    # 個別銘柄の新規ダウンロード
    for ticker in new_companies:
        company_name = COMPANY_NAMES.get(ticker, {}).get('name', ticker)
        print(f"[新規] {ticker} ({company_name})")
        save_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        download_full(ticker, ticker, start_date, end_date, save_path)
    
    print(f"\n{'='*60}")
    print(f"新規銘柄のデータ取得が完了しました。")
    print(f"{'='*60}\n")

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

def main():
    print("=== データ収集を開始します ===")
    
    # 保存ディレクトリの準備
    create_directory(RAW_DATA_DIR)
    
    # 0. 新規追加銘柄の検出と全期間データ取得
    print("\n--- 新規追加銘柄の検出 ---")
    new_companies = detect_new_companies()
    
    # 新規銘柄がある場合は2015年からの全データを取得
    if new_companies:
        download_new_company_data(new_companies, START_DATE, END_DATE)
    else:
        print("新規追加された銘柄はありません。既存銘柄の差分更新を行います。\n")
    
    # 1. 個別銘柄の取得（差分更新）
    print("\n--- 個別銘柄 (Core30 + MyList) の差分更新 ---")
    existing_count = 0
    for ticker in ALL_TICKERS:
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            # 既存銘柄は差分更新
            download_and_save(ticker, ticker, START_DATE, END_DATE)
            existing_count += 1
    print(f"既存銘柄の更新完了: {existing_count}件")
        
    # 2. 指数・マクロ指標の取得（差分更新）
    print("\n--- 市場指数・マクロ指標の差分更新 ---")
    for ticker, name in INDICES.items():
        download_and_save(ticker, name, START_DATE, END_DATE)
        
    # 3. セクターETFの取得（差分更新）
    print("\n--- セクターETFの差分更新 ---")
    for ticker, name in SECTOR_ETFS.items():
        download_and_save(ticker, name, START_DATE, END_DATE)
    
    # 4. 会社名マッピングのエクスポート
    print("\n--- 会社名マッピングのエクスポート ---")
    export_company_names()
    
    # 5. 最新データをJSON形式で保存（ダッシュボード用）
    # 市場が開いている時間帯のみ実行
    if is_market_open():
        print("\n市場営業時間中のため、最新データをJSON形式で保存します。")
        save_latest_market_data()
    else:
        print("\n市場営業時間外のため、JSON保存をスキップします。")
        
    print("\n=== すべての処理が完了しました ===")

if __name__ == "__main__":
    main()
