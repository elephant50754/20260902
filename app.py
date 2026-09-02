import os
import re
import json
import requests
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

app = Flask(__name__)

LINE_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

# 讀取並自動清理網址（過濾 markdown 產生的中括號與小括號）
RAW_CACHE_URL = os.environ.get(
    "GITHUB_CACHE_URL",
    "https://raw.githubusercontent.com/elephant50754/20260902/main/iv_cache.json"
)
url_match = re.search(r'https?://[^\s\)\]]+', RAW_CACHE_URL)
GITHUB_CACHE_URL = url_match.group(0) if url_match else RAW_CACHE_URL.strip()

configuration = Configuration(access_token=LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def load_cache():
    # 1. 優先嘗試從 GitHub 遠端抓取最新快取
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(GITHUB_CACHE_URL, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        print(f"[Cache Warning] 遠端抓取失敗，HTTP 狀態碼: {resp.status_code}")
    except Exception as e:
        print(f"[Cache Warning] 遠端請求異常: {e}")

    # 2. 備援方案：若遠端網路延遲或讀取失敗，直接讀取專案本機檔案
    if os.path.exists("iv_cache.json"):
        try:
            with open("iv_cache.json", "r", encoding="utf-8") as f:
                print("[Cache Info] 成功載入本機 iv_cache.json 備援快取")
                return json.load(f)
        except Exception as e:
            print(f"[Cache Error] 本機讀取失敗: {e}")

    return None

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_msg = event.message.text.strip().upper()
    cache = load_cache()
    
    if not cache or "data" not in cache:
        reply = "⚠️ 暫時無法載入波動率快取資料，請稍後再試或檢查伺服器 Logs。"
    else:
        data_dict = cache["data"]
        if user_msg in data_dict:
            data = data_dict[user_msg]
            
            # 百分比轉換與格式化
            iv_val = float(data.get('iv', 0)) * 100
            hv_val = float(data.get('hv', 0)) * 100
            hv_high = float(data.get('hv_52w_high', 0)) * 100
            hv_low = float(data.get('hv_52w_low', 0)) * 100
            
            strike = data.get('strike', 'N/A')
            premium = data.get('premium', 'N/A')
            exp_info = data.get('exp_info', f"{data.get('exp_date', '')} ({data.get('dte', '')}天)")
            
            reply = f"📊【{user_msg} 波動率決策診斷】\n"
            reply += f"----------------------\n"
            reply += f"• 現貨股價：${data.get('spot', 0)}\n"
            reply += f"• 目前 IV：{iv_val:.2f}%\n"
            reply += f"• 30天 HV：{hv_val:.2f}%\n"
            reply += f"• 52週 HV 區間：{hv_low:.2f}% ~ {hv_high:.2f}%\n"
            reply += f"• IV/HV 比值：{data.get('iv_hv_ratio', 'N/A')}\n"
            reply += f"• 價平履約價：${strike}\n"
            reply += f"• 價平權利金：${premium}\n"
            reply += f"• 期權到期日：{exp_info}\n"
            reply += f"----------------------\n"
            reply += f"💡 建議策略：\n{data.get('strategy', '')}\n"
            reply += f"📅 更新日期：{data.get('updated_date', '')}"
        else:
            reply = f"找不到標的「{user_msg}」，請確認是否為 S&P 500 成分股或 2 倍槓桿 ETF。"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
