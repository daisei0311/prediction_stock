#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTubeチャンネルから動画を取得し、DeepSeek LLMで市場センチメントを分析するスクリプト
"""

import os
import yaml
import pandas as pd
import datetime
from googleapiclient.discovery import build
from openai import OpenAI
from src.settings import CONFIG_FILE, RAW_DATA_DIR, INFLUENCER_SENTIMENT_FILE

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
# CONFIG_FILE = "config.yaml"
# OUTPUT_DIR = "./data/raw"
# OUTPUT_FILE = "influencer_sentiment.csv"

# ---------------------------------------------------------
# 設定ファイルの読み込み
# ---------------------------------------------------------
def load_config():
    """config.yamlを読み込む"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ---------------------------------------------------------
# 既存データの最新日付取得
# ---------------------------------------------------------
def get_latest_date_from_csv():
    """
    既存のCSVファイルから最新の日付を取得する
    Returns:
        latest_date_str: 最新日付の文字列 (YYYY-MM-DD) または None
    """
    output_path = INFLUENCER_SENTIMENT_FILE
    if not os.path.exists(output_path):
        return None
    
    try:
        df = pd.read_csv(output_path)
        if 'Date' not in df.columns or df.empty:
            return None
        
        # Date列をdatetime型に変換して最大値を取得
        df['Date'] = pd.to_datetime(df['Date'])
        latest_date = df['Date'].max()
        
        return latest_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"警告: 既存データの読み込み失敗 - {e}")
        return None

# ---------------------------------------------------------
# YouTubeチャンネルIDの取得
# ---------------------------------------------------------
def get_channel_id_from_handle(youtube, handle):
    """
    ハンドル名(@SHO1112)からチャンネルIDを取得
    Args:
        youtube: YouTube APIクライアント
        handle: ハンドル名 (例: @SHO1112)
    Returns:
        channel_id: チャンネルID
    """
    # ハンドル名から@を削除
    if handle.startswith('@'):
        handle = handle[1:]
    
    try:
        # チャンネル検索
        request = youtube.search().list(
            part="snippet",
            q=handle,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if response['items']:
            channel_id = response['items'][0]['snippet']['channelId']
            print(f"チャンネルID取得成功: {channel_id}")
            return channel_id
        else:
            print(f"警告: ハンドル名 {handle} のチャンネルが見つかりませんでした")
            return None
            
    except Exception as e:
        print(f"エラー: チャンネルID取得失敗 - {e}")
        return None

# ---------------------------------------------------------
# YouTube動画の取得（Uploads Playlist方式）
# ---------------------------------------------------------
def get_uploads_playlist_id(youtube, channel_id):
    """
    チャンネルのアップロードプレイリストIDを取得
    Args:
        youtube: YouTube APIクライアント
        channel_id: チャンネルID
    Returns:
        uploads_playlist_id: アップロードプレイリストID
    """
    try:
        request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if response['items']:
            uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            print(f"アップロードプレイリストID取得成功: {uploads_playlist_id}")
            return uploads_playlist_id
        else:
            print(f"エラー: チャンネルIDが見つかりません")
            return None
            
    except Exception as e:
        print(f"エラー: プレイリストID取得失敗 - {e}")
        return None

def fetch_videos_from_playlist(youtube, playlist_id, cutoff_date='2023-01-01'):
    """
    プレイリストから全動画を取得（ページネーション対応）
    Args:
        youtube: YouTube APIクライアント
        playlist_id: プレイリストID
        cutoff_date: この日付より古い動画は取得しない
    Returns:
        videos_df: 動画データのDataFrame
    """
    import pytz
    # cutoff_dateがNoneの場合はデフォルト値を設定
    if cutoff_date is None:
        cutoff_date = '2023-01-01'
        
    cutoff = datetime.datetime.fromisoformat(cutoff_date).replace(tzinfo=pytz.UTC)
    videos = []
    next_page_token = None
    page_count = 0
    
    print(f"動画取得開始 ({cutoff_date} 以降)")
    
    try:
        while True:
            page_count += 1
            print(f"ページ {page_count} を取得中...")
            
            # プレイリストアイテムを取得
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            # 動画データを抽出
            for item in response['items']:
                published_at = item['snippet']['publishedAt']
                published_date = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                
                # カットオフ日付より古い場合は終了
                if published_date < cutoff:
                    print(f"カットオフ日付到達: {published_date.date()}")
                    break
                
                video_data = {
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': published_at
                }
                videos.append(video_data)
            
            # カットオフ日付に到達したら終了
            if videos and datetime.datetime.fromisoformat(videos[-1]['published_at'].replace('Z', '+00:00')) < cutoff:
                break
            
            # 次のページがあるか確認
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                print("全ページ取得完了")
                break
        
        print(f"取得した動画数: {len(videos)}")
        
        if not videos:
            print("動画が見つかりませんでした")
            return None
        
        # DataFrameに変換
        videos_df = pd.DataFrame(videos)
        videos_df['published_at'] = pd.to_datetime(videos_df['published_at'])
        videos_df['Date'] = videos_df['published_at'].dt.date
        
        return videos_df
        
    except Exception as e:
        print(f"エラー: 動画取得失敗 - {e}")
        return None

# ---------------------------------------------------------
# DeepSeekによるセンチメント分析
# ---------------------------------------------------------
def analyze_sentiment_with_deepseek(client, title, description):
    """
    DeepSeek APIで市場センチメントを分析
    Args:
        client: OpenAI互換クライアント
        title: 動画タイトル
        description: 動画説明文
    Returns:
        sentiment_score: -1.0 〜 +1.0 のセンチメントスコア
    """
    prompt = f"""以下のYouTube動画のタイトルと説明文から、株式市場に対するセンチメント（感情・見通し）を分析してください。

タイトル: {title}
説明文: {description[:500]}

分析結果として、-1.0（超弱気・暴落警戒）から+1.0（超強気・買い推奨）までの数値のみを返してください。
数値以外は一切出力しないでください。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "あなたは金融市場の専門家です。YouTubeの投資系動画から市場センチメントを数値化します。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # レスポンスから数値を抽出
        result = response.choices[0].message.content.strip()
        sentiment_score = float(result)
        
        # 範囲チェック
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return sentiment_score
        
    except Exception as e:
        print(f"警告: センチメント分析失敗 - {e}")
        return 0.0  # エラー時は中立

# ---------------------------------------------------------
# センチメントデータの保存
# ---------------------------------------------------------
def save_sentiment_data(videos_df, client):
    """
    動画データからセンチメントを分析し、日次データとして保存
    Args:
        videos_df: 動画データのDataFrame
        client: DeepSeek APIクライアント
    """
    output_path = INFLUENCER_SENTIMENT_FILE
    
    # 各動画のセンチメントを分析
    print("\n--- センチメント分析開始 ---")
    sentiments = []
    
    from tqdm import tqdm
    
    for idx, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="センチメント分析"):
        sentiment = analyze_sentiment_with_deepseek(
            client,
            row['title'],
            row['description']
        )
        sentiments.append({
            'Date': row['Date'],
            'Sentiment': sentiment,
            'Title': row['title'][:100]  # デバッグ用にタイトルも保存
        })
    
    # DataFrameに変換
    sentiment_df = pd.DataFrame(sentiments)
    
    # 日次で平均化（1日に複数動画がある場合）
    daily_sentiment = sentiment_df.groupby('Date').agg({
        'Sentiment': 'mean',
        'Title': 'first'  # 代表として最初のタイトルを保存
    }).reset_index()
    daily_sentiment.columns = ['Date', 'Influencer_Sentiment', 'Sample_Title']
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    # 既存データと結合（差分更新）
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, parse_dates=['Date'])
        combined_df = pd.concat([existing_df, daily_sentiment], ignore_index=True)
        # 重複削除（最新を優先）
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        combined_df = combined_df.sort_values('Date')
    else:
        combined_df = daily_sentiment
    
    # CSV保存
    combined_df.to_csv(output_path, index=False)
    print(f"\n保存完了: {output_path}")
    print(f"データ件数: {len(combined_df)}")
    print(f"期間: {combined_df['Date'].min().date()} 〜 {combined_df['Date'].max().date()}")
    print(f"\n--- センチメント統計 ---")
    print(combined_df['Influencer_Sentiment'].describe())

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    print("=== YouTubeセンチメント分析開始 ===\n")
    
    # 設定読み込み
    config = load_config()
    youtube_api_key = config.get('youtube_api_key')
    deepseek_api_key = config.get('deepseek_api_key')
    channel_handle = config.get('youtube_channel_id', {}).get(1)
    
    if not youtube_api_key or not deepseek_api_key or not channel_handle:
        print("エラー: config.yamlの設定が不足しています")
        return
    
    # YouTube APIクライアント
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    # DeepSeek APIクライアント
    deepseek_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    
    # チャンネルID取得
    channel_id = get_channel_id_from_handle(youtube, channel_handle)
    if not channel_id:
        return
    
    # アップロードプレイリストID取得
    uploads_playlist_id = get_uploads_playlist_id(youtube, channel_id)
    if not uploads_playlist_id:
        return
    
    # 最新データの取得（増分更新）
    latest_date = get_latest_date_from_csv()
    if latest_date:
        print(f"既存データを確認: 最新日付 {latest_date}")
        cutoff_date = latest_date
    else:
        print("既存データなし: 全期間取得")
        cutoff_date = '2023-01-01'
    
    # 動画取得
    videos_df = fetch_videos_from_playlist(youtube, uploads_playlist_id, cutoff_date=cutoff_date)
    if videos_df is None or len(videos_df) == 0:
        print("処理終了: 動画が取得できませんでした")
        return
    
    # センチメント分析と保存
    save_sentiment_data(videos_df, deepseek_client)
    
    print("\n=== 処理完了 ===")

if __name__ == "__main__":
    main()
