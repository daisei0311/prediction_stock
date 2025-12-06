#!/bin/bash

# 設定
PROJECT_DIR="/home/dk/getyoutubedata_poc"
LOG_FILE="${PROJECT_DIR}/logs/daily_run.log"
DATE=$(date "+%Y-%m-%d %H:%M:%S")

# ログ開始
echo "=== Daily Run Started at ${DATE} ===" >> "${LOG_FILE}"

# ディレクトリ移動
cd "${PROJECT_DIR}" || {
    echo "Error: Failed to change directory to ${PROJECT_DIR}" >> "${LOG_FILE}"
    exit 1
}

# 仮想環境のアクティベート
# .venvが存在するか確認
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at ${PROJECT_DIR}/.venv" >> "${LOG_FILE}"
    exit 1
fi

# PYTHONPATHの設定 (srcディレクトリを認識させるため)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# エラーハンドリング関数
run_script() {
    script_name=$1
    echo "Starting ${script_name}..." >> "${LOG_FILE}"
    
    # src/ プレフィックスを付けて実行
    if python "src/${script_name}" >> "${LOG_FILE}" 2>&1; then
        echo "Finished ${script_name} successfully." >> "${LOG_FILE}"
    else
        echo "Error: ${script_name} failed. Aborting pipeline." >> "${LOG_FILE}"
        exit 1
    fi
}

# パイプライン実行
run_script "collect_data.py"
run_script "collect_sentiment.py"
run_script "create_features.py"
run_script "predict_ranking.py"
run_script "generate_reports.py"
run_script "forecast_to_db.py"

# 完了
END_DATE=$(date "+%Y-%m-%d %H:%M:%S")
echo "=== Daily Run Completed at ${END_DATE} ===" >> "${LOG_FILE}"
echo "----------------------------------------" >> "${LOG_FILE}"

# ---------------------------------------------------------
# Git Push Automation
# ---------------------------------------------------------
echo "Starting Git Push Automation..." >> "${LOG_FILE}"

# Gitユーザー設定 (必要に応じて)
git config user.name "daisei0311" || true
git config user.email "d.k-guitar@outlook.jp" || true

# 変更をステージング
git add . >> "${LOG_FILE}" 2>&1

# 変更ファイルのリストを取得
CHANGES=$(git status --short)

# コミット (変更がない場合はエラーになるが、パイプラインを止めない)
if [ -n "$CHANGES" ]; then
    git commit -m "Daily update: ${DATE}" -m "Changed files:" -m "$CHANGES" >> "${LOG_FILE}" 2>&1
else
    echo "No changes to commit" >> "${LOG_FILE}"
fi

# リモートへプッシュ
if git push origin main >> "${LOG_FILE}" 2>&1; then
    echo "Git push successful." >> "${LOG_FILE}"
    echo "----------------------------------------" >> "${LOG_FILE}"
else
    echo "Error: Git push failed." >> "${LOG_FILE}"
    echo "----------------------------------------" >> "${LOG_FILE}"
    # git pushの失敗は致命的エラーとせず、処理を継続する (あるいは exit 1 にするかは要件次第)
fi
