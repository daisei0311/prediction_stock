import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import os
import glob
import datetime
from src.create_features import load_market_data, process_stock_data
from src.collect_data import CORE30_TICKERS, MY_WATCHLIST
from src.settings import OUTPUT_DIR, RAW_DATA_DIR

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

# OUTPUT_DIR = "./data/output"
MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 201,
    "min_child_samples": 27,
    "max_depth": 5,
    "learning_rate": 0.00850014159305548,
    "feature_fraction": 0.8055290535035726,
    "bagging_fraction": 0.6995096529497417,
    "bagging_freq": 4,
    "reg_alpha": 0.011807293120381207,
    "reg_lambda": 0.044717796346357114,
    "verbose": -1,
    "seed": 42
}

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def prepare_dataset():
    """
    Rawデータを読み込み、特徴量を作成して結合する
    create_features.py の関数を再利用
    """
    print("--- データの準備開始 ---")
    
    # 市場データの準備
    n225_close, market_features = load_market_data()
    
    # 全銘柄のデータをリストに格納
    all_data = []
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    for file in files:
        # create_features.py の関数を使用
        processed_df = process_stock_data(file, n225_close, market_features)
        if processed_df is not None:
            all_data.append(processed_df)
            
    if not all_data:
        raise ValueError("データが見つかりませんでした。")

    # 結合
    full_df = pd.concat(all_data, axis=0)
    full_df = full_df.sort_values(["Date", "Code"])
    
    return full_df

def train_and_predict(df):
    """
    モデル学習と予測を行う (時系列交差検証付き)
    """
    # 特徴量カラムの特定 (Date, Code, Target 以外)
    # ==========================================
    # 【ここに追加】カラム名のクリーニング処理
    # ==========================================
    print("--- カラム名のクリーニングを実行 ---")
    new_columns = []
    for col in df.columns:
        # カラム名を文字列に変換
        col_str = str(col)
        # LightGBMで禁止されている文字(JSON特殊文字)をアンダースコアに置換
        # 禁止文字: [ ] { } " , :
        clean_col = re.sub(r'[",\[\]\{\}:]', '_', col_str)
        new_columns.append(clean_col)
    
    # DataFrameのカラム名を更新
    df.columns = new_columns
    features = [c for c in df.columns if c not in ["Date", "Code", "Target", "Type", "Close"]]
    print(f"使用する特徴量: {features}")
    
    # 1. データの分割
    # Train Set: Targetが存在する (NaNでない)
    train_df = df.dropna(subset=["Target"]).copy().sort_values("Date").reset_index(drop=True)
    
    # Prediction Set: TargetがNaN (直近データ)
    latest_date = df["Date"].max()
    predict_df = df[df["Date"] == latest_date].copy()
    
    print(f"\n--- データ分割 ---")
    print(f"学習用データ期間: {train_df['Date'].min().date()} 〜 {train_df['Date'].max().date()}")
    print(f"学習データ数: {len(train_df)}")
    print(f"予測基準日: {latest_date.date()}")
    print(f"予測対象銘柄数: {len(predict_df)}")
    
    # デバッグ: CloseがNaNの銘柄を確認
    nan_close = predict_df[predict_df["Close"].isna()]
    if not nan_close.empty:
        print(f"\n!!! 警告: CloseがNaNの銘柄があります ({len(nan_close)}件) !!!")
        print(nan_close[["Code", "Date"]].head())
    else:
        print(f"\n確認: 全銘柄のCloseは正常です")
        
    print(f"Close列のサンプル: {predict_df['Close'].head().tolist()}")
    
    if len(predict_df) == 0:
        raise ValueError("予測用データがありません。")
        
    print(f"\n--- 予測用データ確認 ---")
    print(predict_df[["Code", "Close", "RSI_14"]].head())
    print(f"Close列のNaN数: {predict_df['Close'].isna().sum()}")

    # 2. 時系列交差検証 (TSCV)
    print(f"\n--- 時系列交差検証 (5 Folds) ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    
    fold = 1
    for train_index, val_index in tscv.split(train_df):
        # データ分割
        X_train_cv = train_df.iloc[train_index][features]
        y_train_cv = train_df.iloc[train_index]["Target"]
        X_val_cv = train_df.iloc[val_index][features]
        y_val_cv = train_df.iloc[val_index]["Target"]
        
        # データセット作成
        lgb_train = lgb.Dataset(X_train_cv, label=y_train_cv)
        lgb_val = lgb.Dataset(X_val_cv, label=y_val_cv, reference=lgb_train)
        
        # 学習
        model_cv = lgb.train(
            MODEL_PARAMS,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # 予測と評価
        preds_val = model_cv.predict(X_val_cv)
        
        # Rank IC (Spearman相関)
        rank_ic, _ = spearmanr(preds_val, y_val_cv)
        
        # RMSE
        rmse = np.sqrt(np.mean((preds_val - y_val_cv) ** 2))
        
        print(f"Fold {fold}: Rank IC = {rank_ic:.4f}, RMSE = {rmse:.4f}")
        
        cv_results.append({
            "Fold": fold,
            "Rank_IC": rank_ic,
            "RMSE": rmse,
            "Train_Period": f"{train_df.iloc[train_index]['Date'].min().date()} ~ {train_df.iloc[train_index]['Date'].max().date()}",
            "Val_Period": f"{train_df.iloc[val_index]['Date'].min().date()} ~ {train_df.iloc[val_index]['Date'].max().date()}"
        })
        fold += 1
        
    # CV結果の集計
    mean_ic = np.mean([r["Rank_IC"] for r in cv_results])
    std_ic = np.std([r["Rank_IC"] for r in cv_results])
    print(f"\n=== CV結果集計 ===")
    print(f"Mean Rank IC: {mean_ic:.4f}")
    print(f"Std Dev IC: {std_ic:.4f}")
    
    # CV結果の保存用テキスト作成
    cv_text = f"CV Date: {datetime.datetime.now()}\n"
    cv_text += f"Mean Rank IC: {mean_ic:.4f}\n"
    cv_text += f"Std Dev IC: {std_ic:.4f}\n\n"
    for r in cv_results:
        cv_text += f"Fold {r['Fold']}: IC={r['Rank_IC']:.4f}, RMSE={r['RMSE']:.4f} ({r['Val_Period']})\n"
    
    # 3. 全データでの最終学習
    print(f"\n--- 最終モデル学習 (全データ) ---")
    # 検証セットとして直近20%を使用 (Early Stopping用)
    n_train_final = int(len(train_df) * 0.8)
    X_train_final = train_df.iloc[:n_train_final][features]
    y_train_final = train_df.iloc[:n_train_final]["Target"]
    X_val_final = train_df.iloc[n_train_final:][features]
    y_val_final = train_df.iloc[n_train_final:]["Target"]
    
    lgb_train_final = lgb.Dataset(X_train_final, label=y_train_final)
    lgb_val_final = lgb.Dataset(X_val_final, label=y_val_final, reference=lgb_train_final)
    
    final_model = lgb.train(
        MODEL_PARAMS,
        lgb_train_final,
        valid_sets=[lgb_train_final, lgb_val_final],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # 4. 予測実行
    print(f"\n--- 予測実行 ---")
    preds = final_model.predict(predict_df[features])
    predict_df["Score"] = preds
    
    return predict_df, final_model, features, cv_text

def save_results(predict_df, model, feature_names, cv_text):
    """
    結果の保存と可視化
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # 1. ランキングCSVの保存
    # スコア順にソート
    ranking_df = predict_df.sort_values("Score", ascending=False).reset_index(drop=True)
    ranking_df["Rank"] = ranking_df.index + 1
    
    # Type列の追加
    def get_type(code):
        if code in CORE30_TICKERS:
            return "Core30"
        elif code in MY_WATCHLIST:
            return "MyList"
        else:
            return "Other"
            
    ranking_df["Type"] = ranking_df["Code"].apply(get_type)
    
    # 出力カラムの選定
    output_cols = ["Rank", "Code", "Type", "Close", "Score", "RSI_14", "MA_Divergence_25"]
    # 存在しないカラムがあれば除外 (念のため)
    output_cols = [c for c in output_cols if c in ranking_df.columns]
    
    save_path_csv = os.path.join(OUTPUT_DIR, f"forecast_{today_str}.csv")
    ranking_df[output_cols].to_csv(save_path_csv, index=False)
    
    # CV結果の保存
    save_path_cv = os.path.join(OUTPUT_DIR, f"cv_results_{today_str}.txt")
    with open(save_path_cv, "w") as f:
        f.write(cv_text)
    
    print(f"\nランキングを保存しました: {save_path_csv}")
    
    # TOP 10 表示
    print("\n=== 推奨銘柄 TOP 10 ===")
    print(ranking_df[output_cols].head(10))
    
    # 2. 特徴量重要度のプロット
    # TOP20まで表示
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False)
    imp_df = imp_df.head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df)
    plt.title(f"Feature Importance (Gain) - {today_str}")
    plt.tight_layout()
    
    save_path_img = os.path.join(OUTPUT_DIR, f"feature_importance_{today_str}.png")
    plt.savefig(save_path_img)
    print(f"特徴量重要度を保存しました: {save_path_img}")

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

def main():
    try:
        # データ準備
        full_df = prepare_dataset()
        
        # 学習と予測
        predict_df, model, features, cv_text = train_and_predict(full_df)
        
        # 結果保存
        save_results(predict_df, model, features, cv_text)
        
        print("\n=== 全処理完了 ===")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
