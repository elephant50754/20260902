import os
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# 追蹤清單（可依需求擴充或替換為 S&P 100/500）
WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL", "META",
    "PLTR", "COIN", "NFLX", "AVGO", "SPY", "QQQ", "IWM"
]

def fetch_volatility_metrics(symbol: str):
    """抓取現價、計算 30D HV、52週 HV 高低、30D ATM IV 及 IV/HV 比值"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None

        spot = ticker.fast_info.get("lastPrice", hist['Close'].iloc[-1])
        spot = round(float(spot), 2)

        # 1. 計算 30 天滾動年化歷史波動率 (HV)
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        rolling_hv = log_ret.rolling(window=30).std() * np.sqrt(252)
        valid_hv = rolling_hv.dropna()
        if valid_hv.empty:
            return None

        current_hv = float(valid_hv.iloc[-1])
        hv_52w_high = float(valid_hv.max())
        hv_52w_low = float(valid_hv.min())

        # 2. 抓取 30 天平值 (ATM) 隱含波動率 (IV)
        expirations = ticker.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in expirations]
        target_date = min(exp_dates, key=lambda d: abs((d - today).days - 30))
        target_date_str = target_date.strftime("%Y-%m-%d")

        calls = ticker.option_chain(target_date_str).calls.copy()
        if calls.empty:
            return None

        calls['diff'] = (calls['strike'] - spot).abs()
        atm_call = calls.sort_values('diff').iloc[0]
        current_iv = atm_call.get('impliedVolatility')

        if current_iv is None or pd.isna(current_iv) or current_iv <= 0:
            return None

        current_iv = float(current_iv)
        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else None

        # 3. 策略判斷（完全對照 Excel 決策框架）
        # IV >= 50% 屬高檔區間 -> Sell Put
        # IV <= 25% 屬低檔區間 -> Buy Call
        if current_iv >= 0.50:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
        elif current_iv <= 0.25:
            strategy = "Buy Call（IV偏低，適合買方進場）"
            strategy_tag = "BUY_CALL"
        else:
            strategy = "觀望 / 中性（無明顯優勢）"
            strategy_tag = "NEUTRAL"

        return {
            "symbol": symbol,
            "spot": spot,
            "iv": round(current_iv * 100, 2),             # %
            "hv": round(current_hv * 100, 2),             # %
            "hv_52w_high": round(hv_52w_high * 100, 2),   # %
            "hv_52w_low": round(hv_52w_low * 100, 2),     # %
            "iv_hv_ratio": iv_hv_ratio,
            "strategy": strategy,
            "strategy_tag": strategy_tag,
            "exp_date": target_date_str,
            "dte": (target_date - today).days,
            "strike": float(atm_call['strike']),
            "updated_date": datetime.now().strftime("%Y-%m-%d")
        }
    except Exception as e:
        print(f"[{symbol}] 抓取異常: {e}")
        return None

def update_google_sheets(results: list):
    """將結果寫入 Google 試算表（對齊 IV追蹤表 的欄位佈局）"""
    creds_json_str = os.environ.get("GOOGLE_CREDS_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not creds_json_str or not sheet_id:
        print("未設定 Google Sheets 憑證，略過寫入試算表。")
        return

    try:
        creds_dict = json.loads(creds_json_str)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(sheet_id).worksheet("IV追蹤表")

        # 整理要寫入的列（從 Row 5 開始填入資料）
        rows_to_insert = []
        for r in results:
            row = [
                r["symbol"],                           # A: 股票代號
                r["spot"],                             # B: 股價
                r["iv"] / 100,                         # C: 目前IV (以小數寫入，表內格式化為百分比)
                "",                                    # D: 52週IV高
                "",                                    # E: 52週IV低
                "",                                    # F: IV Percentile
                "",                                    # G: IV Rank
                r["hv"] / 100,                         # H: 目前HV
                r["hv_52w_high"] / 100,                # I: 52週HV高
                r["hv_52w_low"] / 100,                 # J: 52週HV低
                "",                                    # K: HV Percentile
                r["iv_hv_ratio"],                      # L: IV/HV 比值
                r["strategy"],                         # M: 建議策略
                "",                                    # N: 選擇權權利金 (選填)
                "",                                    # O: 權利金% (公式自動算)
                "GitHub Actions 自動更新",             # P: 資料來源 / 備註
                r["updated_date"]                      # Q: 更新日期
            ]
            rows_to_insert.append(row)

        # 清除舊有資料（保留前 4 列標題與說明）並寫入新資料
        sheet.batch_clear(["A5:Q100"])
        sheet.update("A5", rows_to_insert, value_input_option="USER_ENTERED")
        print("成功將資料寫入 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    print(f"開始執行美股 IV/HV 與策略計算 (共 {len(WATCHLIST)} 檔)...")
    results = []

    for sym in WATCHLIST:
        data = fetch_volatility_metrics(sym)
        if data:
            results.append(data)
            print(f"  {sym:<5} | 現價: ${data['spot']} | IV: {data['iv']}% | HV: {data['hv']}% | 比值: {data['iv_hv_ratio']} | {data['strategy_tag']}")
        time.sleep(0.3)

    # 1. 寫入 Google Sheets
    update_google_sheets(results)

    # 2. 存為本機快取 JSON (供 LINE Webhook 使用)
    cache_payload = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "data": {item["symbol"]: item for item in results}
    }
    with open("iv_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    print("已成功產出 iv_cache.json 快取檔案。")

if __name__ == "__main__":
    main()
