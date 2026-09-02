import os
import io
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# 僅保留正 2 倍 (2x)、反向 2 倍 (-2x) 與核心基準 ETF (已完全剔除 3 倍標的)
LEVERAGED_AND_BENCHMARK_ETFS = [
    # --- 指數正 2 倍 (2x) ---
    "QLD",   # ProShares Ultra QQQ (那斯達克100 正2)
    "SSO",   # ProShares Ultra S&P500 (標普500 正2)
    "UWM",   # ProShares Ultra Russell2000 (羅素2000 正2)
    
    # --- 行業/板塊正 2 倍 (2x) ---
    "USD",   # ProShares Ultra Semiconductors (半導體 正2)
    "ROM",   # ProShares Ultra Technology (科技 正2)
    "UYG",   # ProShares Ultra Financials (金融 正2)
    
    # --- 熱門個股正 2 倍槓桿 ETF ---
    "NVDL",  # GraniteShares 2x Long NVDA (輝達 正2)
    "TSLL",  # Direxion Daily TSLA Bull 2X (特斯拉 正2)
    "MSTU",  # T-Rex 2X Long MSTR (微策略 正2)
    "MSTX",  # Defiance Daily Target 2X Long MSTR (微策略 正2)
    "CONL",  # GraniteShares 2x Long COIN (Coinbase 正2)
    
    # --- 反向 2 倍避險型 (-2x) ---
    "QID",   # ProShares UltraShort QQQ (那斯達克100 反2)
    "SDS",   # ProShares UltraShort S&P500 (標普500 反2)
    
    # --- 核心基準指數 ETF ---
    "SPY", "QQQ", "IWM", "SMH", "DIA"
]

def get_tracking_tickers():
    """動態取得 S&P 500 成分股，並合併 2 倍槓桿與基準 ETF"""
    print("正在取得 S&P 500 最新成分股名單...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        tables = pd.read_html(io.StringIO(resp.text))
        sp500_table = tables[0]
        sp500_tickers = [str(t).strip().replace(".", "-") for t in sp500_table["Symbol"].tolist()]
    except Exception as e:
        print(f"取得 S&P 500 清單失敗: {e}，使用核心代表性標的...")
        sp500_tickers = [
            "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
            "JPM", "V", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "AMD", "NFLX"
        ]

    # 合併清單（去重複並排序）
    combined_tickers = sorted(list(set(sp500_tickers + LEVERAGED_AND_BENCHMARK_ETFS)))
    print(f"清單整理完成！總計追蹤 {len(combined_tickers)} 檔標的 (S&P 500 + 2倍槓桿/基準 ETF)。")
    return combined_tickers

def fetch_volatility_metrics(symbol: str):
    """計算單一標的的現價、HV、IV、價平權利金與建議策略"""
    try:
        ticker = yf.Ticker(symbol)

        # 1. 抓取歷史股價計算 30 天滾動年化歷史波動率 (HV)
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None

        spot = ticker.fast_info.get("lastPrice", hist["Close"].iloc[-1])
        spot = round(float(spot), 2)
        if spot <= 0:
            return None

        log_ret = np.log(hist["Close"] / hist["Close"].shift(1))
        rolling_hv = log_ret.rolling(window=30).std() * np.sqrt(252)
        valid_hv = rolling_hv.dropna()
        if valid_hv.empty:
            return None

        current_hv = float(valid_hv.iloc[-1])
        hv_52w_high = float(valid_hv.max())
        hv_52w_low = float(valid_hv.min())

        # 2. 抓取最接近 30 天到期的選擇權鏈
        expirations = ticker.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in expirations]
        target_date = min(exp_dates, key=lambda d: abs((d - today).days - 30))
        target_date_str = target_date.strftime("%Y-%m-%d")

        chain = ticker.option_chain(target_date_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return None

        # 3. 尋找價平 (ATM) 履約價
        calls["diff"] = (calls["strike"] - spot).abs()
        atm_call = calls.sort_values("diff").iloc[0] if not calls.empty else None

        puts["diff"] = (puts["strike"] - spot).abs()
        atm_put = puts.sort_values("diff").iloc[0] if not puts.empty else None

        # 4. 提取 IV
        call_iv = float(atm_call.get("impliedVolatility", 0)) if atm_call is not None else 0
        put_iv = float(atm_put.get("impliedVolatility", 0)) if atm_put is not None else 0

        valid_ivs = [v for v in [call_iv, put_iv] if v > 0.05]
        if not valid_ivs:
            current_iv = call_iv if call_iv > 0 else put_iv
        else:
            current_iv = sum(valid_ivs) / len(valid_ivs)

        if current_iv <= 0:
            return None

        # 5. 計算價平合約權利金 Mid-Price = (Bid + Ask) / 2
        def get_mid_price(opt_row):
            if opt_row is None:
                return 0.0
            bid = float(opt_row.get("bid", 0))
            ask = float(opt_row.get("ask", 0))
            last = float(opt_row.get("lastPrice", 0))
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 2)
            return round(last, 2)

        call_mid = get_mid_price(atm_call)
        put_mid = get_mid_price(atm_put)

        # 6. 建議策略判斷
        if current_iv >= 0.50:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
            premium = put_mid if put_mid > 0 else call_mid
        elif current_iv <= 0.25:
            strategy = "Buy Call（IV偏低，適合買方進場）"
            strategy_tag = "BUY_CALL"
            premium = call_mid if call_mid > 0 else put_mid
        else:
            strategy = "觀望 / 中性（無明顯優勢）"
            strategy_tag = "NEUTRAL"
            premium = call_mid if call_mid > 0 else put_mid

        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else None

        return {
            "symbol": symbol,
            "spot": spot,
            "iv": current_iv,                              # 小數格式
            "hv": current_hv,                              # 小數格式
            "hv_52w_high": hv_52w_high,
            "hv_52w_low": hv_52w_low,
            "iv_hv_ratio": iv_hv_ratio,
            "strategy": strategy,
            "strategy_tag": strategy_tag,
            "premium": premium,                            # N 欄位：價平權利金
            "exp_date": target_date_str,
            "dte": (target_date - today).days,
            "updated_date": datetime.now().strftime("%Y-%m-%d")
        }
    except Exception:
        return None

def update_google_sheets(results: list):
    """將結果批次寫入 Google 試算表"""
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

        rows_to_insert = []
        for idx, r in enumerate(results, start=5):
            # 數值格式校正 (確保以小數寫入，讓 Google 試算表正確顯示為 %)
            iv_val = r["iv"] / 100 if r["iv"] > 1.5 else r["iv"]
            hv_val = r["hv"] / 100 if r["hv"] > 1.5 else r["hv"]
            hv_h_val = r["hv_52w_high"] / 100 if r["hv_52w_high"] > 1.5 else r["hv_52w_high"]
            hv_l_val = r["hv_52w_low"] / 100 if r["hv_52w_low"] > 1.5 else r["hv_52w_low"]

            # O 欄位自動填入公式：權利金 ÷ 現價
            formula_prem_pct = f'=IF(OR($A{idx}="",$N{idx}="",$B{idx}="",$B{idx}=0),"",$N{idx}/$B{idx})'

            note = "2倍槓桿/基準 ETF" if r["symbol"] in LEVERAGED_AND_BENCHMARK_ETFS else "S&P 500 成分股"

            row = [
                r["symbol"],           # A: 股票代號
                r["spot"],             # B: 股價
                round(iv_val, 4),      # C: 目前IV
                "",                    # D: 52週IV高
                "",                    # E: 52週IV低
                "",                    # F: IV Percentile
                "",                    # G: IV Rank
                round(hv_val, 4),      # H: 目前HV
                round(hv_h_val, 4),    # I: 52週HV高
                round(hv_l_val, 4),    # J: 52週HV低
                "",                    # K: HV Percentile
                r["iv_hv_ratio"],      # L: IV/HV 比值
                r["strategy"],         # M: 建議策略
                r["premium"],          # N: 選擇權權利金 (價平合約 Mid-Price)
                formula_prem_pct,      # O: 權利金% (公式自動計算)
                note,                  # P: 資料來源 / 備註
                r["updated_date"]      # Q: 更新日期
            ]
            rows_to_insert.append(row)

        print(f"準備寫入 Google Sheets (共 {len(rows_to_insert)} 筆)...")
        # 清除第 5 列以下舊資料並寫入新資料
        sheet.batch_clear(["A5:Q"])
        sheet.update("A5", rows_to_insert, value_input_option="USER_ENTERED")
        print("成功將 S&P 500 與 2 倍槓桿 ETF 數據寫入 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始掃描選擇權指標與價平權利金 (共 {len(tickers)} 檔)...")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(fetch_volatility_metrics, sym): sym for sym in tickers}
        completed_count = 0
        total = len(tickers)

        for future in as_completed(future_to_symbol):
            completed_count += 1
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 25 == 0:
                    print(f"進度 [{completed_count}/{total}] | 已處理 {len(results)} 檔標的...")

    results.sort(key=lambda x: x["symbol"])
    print(f"\n掃描結束！成功取得 {len(results)} 檔標的數據。")

    # 1. 批次更新 Google Sheets
    update_google_sheets(results)

    # 2. 存入 JSON 快取供 LINE 機器人即時過濾
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
