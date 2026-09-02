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

# 讀取並自動清理網址
RAW_CACHE_URL = os.environ.get(
    "GITHUB_CACHE_URL",
    "https://raw.githubusercontent.com/elephant50754/20260902/main/iv_cache.json"
)
url_match = re.search(r'https?://[^\s\)\]]+', RAW_CACHE_URL)
GITHUB_CACHE_URL = url_match.group(0) if url_match else RAW_CACHE_URL.strip()

configuration = Configuration(access_token=LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def load_cache():
    """優先自 GitHub 遠端抓取，失敗時讀取本機 iv_cache.json 備援"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(GITHUB_CACHE_URL, headers=headers, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[Cache Warning] 遠端抓取失敗: {e}")

    if os.path.exists("iv_cache.json"):
        try:
            with open("iv_cache.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache Error] 本機讀取失敗: {e}")
    return None

def to_pct(val):
    """安全轉換為百分比數值"""
    try:
        f = float(val)
        return f * 100 if f <= 1.5 else f
    except (ValueError, TypeError):
        return 0.0

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
    raw_text = event.message.text.strip()
    cmd = raw_text.upper()
    cache = load_cache()

    if not cache or "data" not in cache:
        reply = "⚠️ 暫時無法讀取快取資料，請確認 GitHub Actions 排程是否已產出 iv_cache.json。"
    else:
        data_dict = cache["data"]

        # -------------------------------------------------------------
        # 指令 1：Sell Put / 賣方（列出高 IV、適合做賣方收租的標的）
        # -------------------------------------------------------------
        if cmd in ["SELL PUT", "SELLPUT", "賣方", "SELL"]:
            candidates = []
            for sym, item in data_dict.items():
                tag = item.get("strategy_tag", "")
                ratio = float(item.get("iv_hv_ratio") or 0)
                if tag == "SELL_PUT" or ratio >= 1.25:
                    candidates.append(item)

            candidates.sort(key=lambda x: float(x.get("iv_hv_ratio") or 0), reverse=True)
            top_list = candidates[:15]

            if not top_list:
                reply = "目前市場無顯著偏高 IV 的 Sell Put 標的。"
            else:
                reply = f"🎯【Sell Put 賣方精選清單】(共 {len(candidates)} 檔，取前 15 檔)\n"
                reply += "----------------------\n"
                for idx, c in enumerate(top_list, 1):
                    iv = to_pct(c.get("iv", 0))
                    prem = c.get("premium", "")
                    spot_val = float(c.get("spot", 0) or 0)
                    prem_str = ""
                    if prem != "":
                        try:
                            p_pct = (float(prem) / spot_val * 100) if spot_val > 0 else 0
                            prem_str = f" | Put:${prem} ({p_pct:.2f}%)"
                        except (ValueError, TypeError):
                            prem_str = f" | Put:${prem}"
                    reply += f"{idx}. {c['symbol']} (${c['spot']})\n"
                    reply += f"   • IV: {iv:.1f}%{prem_str}\n"
                reply += "----------------------\n"
                reply += "💡 賣方要訣：鎖定高 IV 且權利金%豐厚之標的，收取純時間價值。"

        # -------------------------------------------------------------
        # 指令 2：Buy Call / 買方（列出低 IV、適合買方進場的標的）
        # -------------------------------------------------------------
        elif cmd in ["BUY CALL", "BUYCALL", "買方", "BUY"]:
            candidates = []
            for sym, item in data_dict.items():
                tag = item.get("strategy_tag", "")
                ratio = float(item.get("iv_hv_ratio") or 999)
                if tag == "BUY_CALL" or ratio <= 0.80:
                    candidates.append(item)

            candidates.sort(key=lambda x: to_pct(x.get("iv", 999)))
            top_list = candidates[:15]

            if not top_list:
                reply = "目前市場無顯著偏低 IV 的 Buy Call 標的。"
            else:
                reply = f"🚀【Buy Call 買方精選清單】(共 {len(candidates)} 檔，取前 15 檔)\n"
                reply += "----------------------\n"
                for idx, c in enumerate(top_list, 1):
                    iv = to_pct(c.get("iv", 0))
                    reply += f"{idx}. {c['symbol']} (${c['spot']})\n"
                    reply += f"   • IV: {iv:.1f}%\n"
                reply += "----------------------\n"
                reply += "💡 買方要訣：IV 處於歷史低檔，合約極度便宜，適合做方向性佈局。"

        # -------------------------------------------------------------
        # 指令 3：IV 區間篩選（例如輸入「IV 40-60」）
        # -------------------------------------------------------------
        elif re.match(r'^IV\s*(\d+(?:\.\d+)?)\s*[-~至到\s]\s*(\d+(?:\.\d+)?)$', cmd):
            match = re.match(r'^IV\s*(\d+(?:\.\d+)?)\s*[-~至到\s]\s*(\d+(?:\.\d+)?)$', cmd)
            low_bound = float(match.group(1))
            high_bound = float(match.group(2))
            if low_bound > high_bound:
                low_bound, high_bound = high_bound, low_bound

            filtered = []
            for sym, item in data_dict.items():
                iv = to_pct(item.get("iv", 0))
                if low_bound <= iv <= high_bound:
                    filtered.append((iv, item))

            filtered.sort(key=lambda x: x[0], reverse=True)
            top_list = filtered[:15]

            if not top_list:
                reply = f"🔍 在 IV {low_bound}% ~ {high_bound}% 區間內未找到符合標的。"
            else:
                reply = f"🔍【IV 介於 {low_bound}% ~ {high_bound}% 標的】(共 {len(filtered)} 檔，取前 15 檔)\n"
                reply += "----------------------\n"
                for idx, (iv, c) in enumerate(top_list, 1):
                    strat_short = "Sell Put" if "Sell" in c.get("strategy", "") else ("Buy Call" if "Buy" in c.get("strategy", "") else "中性")
                    reply += f"{idx}. {c['symbol']} (${c['spot']}) -> IV: {iv:.1f}% | 建議: {strat_short}\n"
                reply += "----------------------\n"
                reply += f"💡 輸入個股代號（如 {top_list[0][1]['symbol']}）可看完整期權診斷。"

        # -------------------------------------------------------------
        # 指令 4：單一標的診斷卡片（例如輸入「NVDA」、「AAPL」）
        # -------------------------------------------------------------
        elif cmd in data_dict:
            data = data_dict[cmd]
            spot_val = float(data.get('spot', 0) or 0)
            iv_val = to_pct(data.get('iv', 0))

            strike = data.get('strike', '')
            premium = data.get('premium', '')
            exp_info = data.get('exp_info', f"{data.get('exp_date', '')} ({data.get('dte', '')}天)")

            reply = f"📊【{cmd} 波動率決策診斷】\n"
            reply += "----------------------\n"
            reply += f"• 現貨股價：${data.get('spot', 0)}\n"
            reply += f"• 目前 IV：{iv_val:.2f}%\n"

            # 僅在有權利金時顯示履約價、權利金金額與權利金%
            if premium != "":
                try:
                    prem_num = float(premium)
                    prem_pct = (prem_num / spot_val * 100) if spot_val > 0 else 0.0
                    prem_pct_str = f"{prem_pct:.2f}%"
                except (ValueError, TypeError):
                    prem_pct_str = "N/A"

                reply += f"• 價外 Put 履約價：${strike}\n"
                reply += f"• 價外 Put 權利金：${premium}\n"
                reply += f"• 權利金%：{prem_pct_str}\n"

            reply += f"• 參考到期日：{exp_info}\n"
            reply += "----------------------\n"
            reply += f"💡 建議策略：\n{data.get('strategy', '')}\n"
            reply += f"📅 更新日期：{data.get('updated_date', '')}"

        # -------------------------------------------------------------
        # 提示選單（無效指令或幫助）
        # -------------------------------------------------------------
        else:
            reply = (
                "🤖【美股期權決策機器人 指令說明】\n"
                "----------------------\n"
                "1. 個股健檢：輸入代號，如「NVDA」或「TSLA」\n"
                "2. 賣方策略：輸入「Sell Put」或「賣方」\n"
                "3. 買方策略：輸入「Buy Call」或「買方」\n"
                "4. 區間篩選：輸入「IV 40-60」或「IV 30~50」"
            )

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
