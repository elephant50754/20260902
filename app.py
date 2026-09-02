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
GITHUB_CACHE_URL = os.environ.get(
    "GITHUB_CACHE_URL",
    "https://raw.githubusercontent.com/elephant50754/20260902/main/iv_cache.json"
)

configuration = Configuration(access_token=LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def load_cache():
    if os.path.exists("iv_cache.json"):
        with open("iv_cache.json", "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = requests.get(GITHUB_CACHE_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"data": {}, "updated_at_utc": "未知"}

def process_query(msg: str) -> str:
    cache = load_cache()
    stock_data = cache.get("data", {})
    if not stock_data:
        return "⚠️ 目前無快取資料，請先在 GitHub Actions 執行手動更新。"

    text = msg.strip().upper()

    # 1. 查詢單一個股診斷卡片
    if text in stock_data:
        d = stock_data[text]
        return (
            f"📊 【{text} 波動率決策診斷】\n"
            f"━━━━━━━━━━━━━━━\n"
            f"• 現貨股價：${d['spot']}\n"
            f"• 目前 IV：{d['iv']}%\n"
            f"• 30天 HV：{d['hv']}%\n"
            f"• 52週 HV 區間：{d['hv_52w_low']}% ~ {d['hv_52w_high']}%\n"
            f"• IV/HV 比值：{d['iv_hv_ratio']} (市場預期/近期實際)\n"
            f"• 30D ATM 履約價：${d['strike']} ({d['dte']}天)\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💡 建議策略：\n{d['strategy']}\n"
            f"📅 更新日期：{d['updated_date']}"
        )

    # 2. 依表格策略篩選 (Sell Put / Buy Call)
    if text in ["SELL PUT", "SELLPUT", "賣方"]:
        matched = [d for d in stock_data.values() if d["strategy_tag"] == "SELL_PUT"]
        matched.sort(key=lambda x: x["iv"], reverse=True)
        return format_strategy_list(matched, "🔥 適合 Sell Put 策略（IV偏高、權利金貴）")

    if text in ["BUY CALL", "BUYCALL", "買方"]:
        matched = [d for d in stock_data.values() if d["strategy_tag"] == "BUY_CALL"]
        matched.sort(key=lambda x: x["iv"])
        return format_strategy_list(matched, "❄️ 適合 Buy Call 策略（IV偏低、權利金便宜）")

    # 3. 數值區間篩選 (IV 40-60)
    range_m = re.match(r"^IV\s*(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)$", text)
    if range_m:
        min_v, max_v = float(range_m.group(1)), float(range_m.group(2))
        matched = [d for d in stock_data.values() if min_v <= d['iv'] <= max_v]
        matched.sort(key=lambda x: x['iv'], reverse=True)
        return format_strategy_list(matched, f"🎯 IV 落在 {min_v}% ~ {max_v}% 標的")

    # 4. 門檻篩選 (IV > 50 / IV < 30)
    gt_m = re.match(r"^IV\s*>\s*=?\s*(\d+(?:\.\d+)?)$", text)
    if gt_m:
        val = float(gt_m.group(1))
        matched = [d for d in stock_data.values() if d['iv'] >= val]
        matched.sort(key=lambda x: x['iv'], reverse=True)
        return format_strategy_list(matched, f"🔥 IV 大於 {val}% 標的")

    return (
        "💡 【IV 波動率追蹤機器人指令】\n\n"
        "1. 個股診斷：輸入代號如「NVDA」、「AAPL」\n"
        "2. 策略推薦：輸入「Sell Put」或「Buy Call」\n"
        "3. 區間篩選：輸入「IV 40-60」或「IV > 50」"
    )

def format_strategy_list(items: list, title: str) -> str:
    if not items:
        return f"{title}\n查無符合條件之標的。"
    lines = [f"{title}\n符合標的共 {len(items)} 檔：\n"]
    for d in items:
        lines.append(f"• {d['symbol']:<5} | IV: {d['iv']}% | IV/HV: {d['iv_hv_ratio']} (${d['spot']})")
    return "\n".join(lines)

@app.route("/callback", methods=['POST'])
def callback():
    sig = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_msg(event):
    reply = process_query(event.message.text)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
