import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import skew, kurtosis
from src.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, INFLUENCER_SENTIMENT_FILE

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

# RAW_DATA_DIR = "./data/raw"
# PROCESSED_DATA_DIR = "./data/processed"
TARGET_HORIZON = 20  # 予測期間 (20営業日後)

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def calculate_rsi(series, period=14):
    """RSI (Relative Strength Index) を計算する"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_csv_safe(file_path):
    """yfinance形式のCSVを安全に読み込む"""
    try:
        # まずは標準的なCSVとして読み込む (header=0)
        df = pd.read_csv(file_path, index_col=0)
        
        # インデックス名がDateでない場合、または古いyfinance形式(3行ヘッダー)の可能性がある場合
        if df.index.name != 'Date':
            # 3行スキップで再試行 (古い形式の互換性維持)
            try:
                temp_df = pd.read_csv(file_path, skiprows=[1, 2], index_col=0)
                if temp_df.index.name == 'Date':
                    df = temp_df
                    # カラム名がPriceになっている場合の修正
                    if "Price" in df.columns:
                        df.rename(columns={"Price": "Date"}, inplace=True)
            except:
                pass

        # Dateカラムをdatetimeに変換
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # タイムゾーン削除
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # データが右にシフトしている場合の補正 (yfinanceの追記バグ対応)
        # 例: Date, NaN, NaN, ..., Close, High, ...
        if "Close" in df.columns and df["Close"].isna().sum() > 0:
            # CloseがNaNの行を特定
            # nan_rows = df[df["Close"].isna()]
            # 簡易的に、CloseがNaNの行は削除する
            df = df.dropna(subset=["Close"])
            
        # 重複削除
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    except Exception as e:
        print(f"読み込みエラー: {file_path}, {e}")
        return None

def remove_correlated_features(df, threshold=0.95):
    """
    相関が高すぎる特徴量を削除する
    Args:
        df: データフレーム
        threshold: 相関係数の閾値 (デフォルト: 0.95)
    Returns:
        削除後のデータフレーム
    """
    # 特徴量のみを抽出 (Date, Code, Target, Close以外)
    feature_cols = [c for c in df.columns if c not in ["Date", "Code", "Target", "Close"]]
    
    # 相関行列を計算
    corr_matrix = df[feature_cols].corr().abs()
    
    # 上三角行列を取得 (重複を避けるため)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 閾値を超える相関を持つ特徴量を特定
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    if to_drop:
        print(f"\n--- 高相関特徴量の削除 (閾値: {threshold}) ---")
        print(f"削除する特徴量: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print(f"\n--- 高相関特徴量なし (閾値: {threshold}) ---")
    
    return df

def load_market_data():
    """市場データ (日経平均, ドル円, 米国債, VIX, セクターETF) を読み込み、整形して返す"""
    print("--- 市場データの読み込み ---")
    
    # 日経平均 (Target計算用 & 特徴量用)
    n225 = load_csv_safe(os.path.join(RAW_DATA_DIR, "Nikkei225.csv"))
    # 終値のみを使用
    n225_close = n225["Close"]
    
    # ドル円 (特徴量用)
    usdjpy = load_csv_safe(os.path.join(RAW_DATA_DIR, "USDJPY.csv"))
    usdjpy = usdjpy["Close"].shift(1)
    
    # 米国10年債利回り (特徴量用)
    us10y = load_csv_safe(os.path.join(RAW_DATA_DIR, "US10Y.csv"))
    us10y = us10y["Close"].shift(1)
    
    # VIX (特徴量用)
    vix = load_csv_safe(os.path.join(RAW_DATA_DIR, "VIX.csv"))
    vix = vix["Close"].shift(1)
    
    # 市場データの特徴量作成
    market_features = pd.DataFrame(index=n225.index)
    market_features["USDJPY_Change"] = usdjpy.pct_change(fill_method=None)
    market_features["US10Y_Change"] = us10y.pct_change(fill_method=None)
    
    # VIX特徴量
    # VIXは日付が米国時間なので、日本時間とズレる可能性があるが、
    # 単純に結合するとNaNが増えるため、ffillで対応する
    market_features["VIX_Close"] = vix
    market_features["VIX_Change"] = vix.pct_change(fill_method=None)
    market_features["Is_High_VIX"] = (vix > 20).astype(int)
    
    # セクターETF特徴量
    sector_files = glob.glob(os.path.join(RAW_DATA_DIR, "Sector_*.csv"))
    for f in sector_files:
        sector_name = os.path.basename(f).replace(".csv", "")
        df_sector = load_csv_safe(f)
        if df_sector is not None:
            # 5日リターンと20日リターン
            ret_5 = df_sector["Close"].pct_change(5,fill_method=None)
            ret_20 = df_sector["Close"].pct_change(20,fill_method=None)
            
            market_features[f"{sector_name}_Return_5d"] = ret_5
            market_features[f"{sector_name}_Return_20d"] = ret_20
    
    # インフルエンサーセンチメント特徴量
    sentiment_path = INFLUENCER_SENTIMENT_FILE
    if os.path.exists(sentiment_path):
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['Date'])
        sentiment_df = sentiment_df.set_index('Date')
        
        # Influencer_Sentiment列のみを使用（Sample_Titleは除外）
        sentiment_df = sentiment_df[['Influencer_Sentiment']]
        
        # market_featuresと結合
        market_features = market_features.join(sentiment_df, how='left')
        
        # センチメントのNaNは中立（0.0）で埋める（過去データを保持）
        market_features['Influencer_Sentiment'] = market_features['Influencer_Sentiment'].fillna(0.0)
        
        print(f"インフルエンサーセンチメント読み込み完了: {len(sentiment_df)}日分")
    else:
        print("警告: influencer_sentiment.csv が見つかりません")
        market_features['Influencer_Sentiment'] = 0.0
    
    # 欠損値補完 (前日埋め -> 0埋め)
    # VIXやセクターETFの休日ズレを補完
    # ※センチメントは既に0埋め済みなので影響なし
    market_features = market_features.ffill().fillna(0)
    
    return n225_close, market_features

def process_stock_data(file_path, n225_close, market_features):
    """個別銘柄のデータを読み込み、特徴量とTargetを作成する"""
    ticker = os.path.basename(file_path).replace(".csv", "")
    
    # 市場指標ファイルはスキップ
    if ticker in ["Nikkei225", "USDJPY", "US10Y", "SP500", "VIX", "influencer_sentiment"] or ticker.startswith("Sector_"):
        return None
        
    print(f"[{ticker}] 処理中...")
    
    df = load_csv_safe(file_path)
    if df is None:
        return None
    
    # 必要な列があるか確認
    if "Close" not in df.columns:
        return None

    # -----------------------------------------------------
    # 1. 特徴量の作成
    # -----------------------------------------------------
    # --- 追加 1: 対市場 (N225) 感応度と相関 ---
    # n225_close は日付インデックスを持っている前提で結合
    # 一時的に結合して計算
    df["N225"] = n225_close
        # --- 新規追加: モメンタム (ROC) ---
    df["Momentum_10"] = df["Close"].pct_change(10, fill_method=None)
    df["Momentum_20"] = df["Close"].pct_change(20, fill_method=None)
       # --- 新規追加: ストキャスティクス ---
    # %K (14日), %D (3日), Slow%D (3日)
    low_min = df["Low"].rolling(window=14).min()
    high_max = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    df["Stoch_SlowD"] = df["Stoch_D"].rolling(window=3).mean()
    

    # ボラティリティ (20日の対数リターンの標準偏差)
    log_return = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_20"] = log_return.rolling(20).std()
    
    # 出来高変化率 (当日 / 過去5日平均)
    df["Volume_Ratio_5"] = df["Volume"] / df["Volume"].rolling(5).mean()
    
    # --- 新規追加: リバーサル指標 (長期リターン) ---
    df["Return_60d"] = df["Close"].pct_change(60,fill_method=None)   # 60営業日リターン
    df["Return_120d"] = df["Close"].pct_change(120,fill_method=None) # 120営業日リターン
    
    # --- 新規追加: リスクの質 (歪度・尖度) ---
    # 60日間の日次リターンの分布形状
    daily_returns = df["Close"].pct_change(fill_method=None)
    df["Skewness_60d"] = daily_returns.rolling(60).apply(lambda x: skew(x.dropna()), raw=False)
    df["Kurtosis_60d"] = daily_returns.rolling(60).apply(lambda x: kurtosis(x.dropna()), raw=False)
    # 60日間の相関とベータ
    # Beta = Cov(Stock, Market) / Var(Market)
    # 計算高速化のため、リターン同士で計算します
    stock_ret = df["Close"].pct_change(fill_method=None)
    market_ret = df["N225"].pct_change(fill_method=None)
    
    # 60日相関
    df["Corr_N225_60"] = stock_ret.rolling(60).corr(market_ret)
    
    # 60日ベータ (相関 * (個別ボラ / 市場ボラ))
    stock_vol = stock_ret.rolling(60).std()
    market_vol = market_ret.rolling(60).std()
    df["Beta_N225_60"] = df["Corr_N225_60"] * (stock_vol / market_vol)
    
    # 不要な列を削除
    df = df.drop(columns=["N225"],axis=1)
    # --- 追加 2: 長期的な価格位置 (アンカリング) ---
    # 52週 (約250営業日) の最高値・最安値との距離
    window_year = 250
    high_52w = df["High"].rolling(window=window_year).max()
    low_52w = df["Low"].rolling(window=window_year).min()
    
    # 現在価格が52週レンジのどこにいるか (0.0:最安値, 1.0:最高値)
    df["Price_Rank_52W"] = (df["Close"] - low_52w) / (high_52w - low_52w)
    
    # 最高値からの下落率 (ドローダウン)
    df["Drawdown_52W"] = (df["Close"] / high_52w) - 1.0
    # --- 追加 3: 効率性指標 (簡易シャープレシオ) ---
    # 20日リターン / 20日ボラティリティ
    # ゼロ除算を防ぐため、分母に極小値を加えるなどのケアも本来は有効
    # --- 追加 4: 日付アノマリー ---
    # 月末フラグ (翌日が月替わり)
    # df.index.shift(1) は freq が必要なので、Seriesに変換してshiftする
    dates = df.index.to_series()
    df["Is_Month_End"] = (dates.dt.month != dates.shift(-1).dt.month).astype(int)
    # 月初フラグ
    df["Is_Month_Start"] = (df.index.day <= 3).astype(int)
    # 曜日 (0:月, 4:金) -> 金曜日は手仕舞い売りが出やすい等
    df["Day_of_Week"] = df.index.dayofweek
    # 現在のコード
    # 移動平均乖離率 (5日, 25日, 75日)
    df["MA_Divergence_5"] = (df["Close"] - df["Close"].rolling(5).mean()) / df["Close"].rolling(5).mean()
    df["MA_Divergence_25"] = (df["Close"] - df["Close"].rolling(25).mean()) / df["Close"].rolling(25).mean()
    df["MA_Divergence_75"] = (df["Close"] - df["Close"].rolling(75).mean()) / df["Close"].rolling(75).mean()
    
    # RSI (14日)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # --- 新規追加: MACD ---
    # EMA12, EMA26
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # --- 新規追加: ボリンジャーバンド ---
    # 20日移動平均と標準偏差
    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = ma20 + (std20 * 2)
    df["BB_Lower"] = ma20 - (std20 * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / ma20
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
 
    # 市場データの結合 (Dateをキーに結合)
    df = df.join(market_features, how="left")
    
    # -----------------------------------------------------
    # 2. Targetの作成 (Alpha over Nikkei 225)
    # -----------------------------------------------------
    
    # 銘柄の将来リターン
    stock_future_return = df["Close"].shift(-TARGET_HORIZON) / df["Close"] - 1
    
    # 日経平均の将来リターン (同じ日付インデックスで計算)
    # n225_close は Series なので、df の index に合わせて結合または参照する必要がある
    # ここでは join を使って一時的に結合して計算する
    df["N225_Close"] = n225_close
    n225_future_return = df["N225_Close"].shift(-TARGET_HORIZON) / df["N225_Close"] - 1
    
    # Target = 銘柄リターン - 日経平均リターン
    raw_target = stock_future_return - n225_future_return
    # -10% ~ +10% の範囲に収める (例)
    df["Target"] = raw_target.clip(lower=-0.1, upper=0.1)
    
    # 不要な列の削除
    df = df.drop(columns=["N225_Close"])
    
    # 銘柄コード列の追加
    df["Code"] = ticker
    df["Sharpe_20"] = df["Momentum_20"] / (df["Volatility_20"] + 1e-6)
    
    # インデックスを列に戻す
    df = df.reset_index()

    return df

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

def main():
    # 保存ディレクトリ作成
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    # 市場データの準備
    n225_close, market_features = load_market_data()
    
    # 全銘柄のデータをリストに格納
    all_data = []
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    for file in files:
        processed_df = process_stock_data(file, n225_close, market_features)
        if processed_df is not None:
            all_data.append(processed_df)
    
    # 結合
    if not all_data:
        print("エラー: データが見つかりませんでした。")
        return
    train_data = pd.concat(all_data, axis=0)
    
    print(f"\n--- データ結合後 ---")
    print(f"データ形状: {train_data.shape}")
    
    # 高相関特徴量の削除
    train_data = remove_correlated_features(train_data, threshold=0.95)
    # 2. 【重要】「全滅列」の検出と削除
    # 欠損率が90%を超える列を探す（これが犯人の可能性が高い）
    nan_ratio = train_data.isna().mean()
    bad_cols = nan_ratio[nan_ratio > 0.9].index.tolist()
    
    # TargetはNaNがあっても（直近分なので）許す。それ以外で9割NaNなら列ごと消す
    bad_cols = [c for c in bad_cols if c != 'Target']
    
    if bad_cols:
        print(f"\n!!! 警告: 以下の列はほぼ全ての行がNaNのため削除します（全滅回避） !!!")
        print(f"削除列: {bad_cols}")
        train_data = train_data.drop(columns=bad_cols,axis=1)
    print(f"\n--- 相関フィルタリング後 ---")
    print(f"データ形状: {train_data.shape}")
    
    # NaN除去
    print(f"\n--- 処理完了 ---")
    print(f"元データ行数: {len(train_data)}")
    print(f"NaN含有行数: {train_data.isna().any(axis=1).sum()}")
    
    # train_data = train_data.dropna()
    print(f"削除後行数: {len(train_data)} (NaNを含む行を削除)")
    
    # 保存
    save_path = os.path.join(PROCESSED_DATA_DIR, "train_data.csv")
    train_data.to_csv(save_path, index=False)
    print(f"保存しました: {save_path}")
    
    # 確認表示
    print("\n--- データセット先頭5行 ---")
    print(train_data.head())
    print("\n--- データセット形状 ---")
    print(train_data.shape)

if __name__ == "__main__":
    main()
