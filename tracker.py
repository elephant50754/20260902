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

# 僅保留正 2 倍 (2x)、反向 2 倍 (-2x) 與核心基準 ETF
LEVERAGED_AND_BENCHMARK_ETFS = [
    # 指數正 2 倍
    "QLD", "SSO", "UWM",
    # 板塊正 2 倍
    "USD", "ROM", "UYG",
    # 熱門個股正 2 倍
    "NVDL", "TSLL", "MSTU", "MSTX", "CONL",
    # 反向 2 倍避險
    "QID", "SDS",
    # 基準指數 ETF
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
        sp500_tickers = [str(t).strip().replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    except Exception as e:
        print(f"取得 S&P 500 清單失敗: {e}，使用核心代表性標的...")
        sp500_tickers = [
            "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
            "JPM", "V", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "AMD", "NFLX"
        ]

    combined_tickers = sorted(list(set(sp500_tickers + LEVERAGED_AND_BENCHMARK_ETFS)))
    print(f"清單整理完成！總計追蹤 {len(combined_tickers)} 檔標的。")
    return combined_tickers

def select_target_monthly_expiration(expirations, today):
    """
    篩選標準月選擇權 (每個月第三個星期五，排除週期權)。
    若當月結算日剩餘天數小於 25 天，自動順延至「結算日後 25~30 天」的下個月標準結算日。
    """
    if not expirations:
        return None, None

    exp_dates = []
    for d_str in expirations:
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            exp_dates.append((d, d_str))
        except Exception:
            continue

    if not exp_dates:
        return None, None

    # 美股標準月選結算日：週五 (weekday == 4) 且日期落在 15~21 號之間
    monthly_exps = [
        (d, d_str) for d, d_str in exp_dates
        if d.weekday() == 4 and 15 <= d.day <= 21 and d >= today
    ]
    monthly_exps.sort(key=lambda x: x[0])

    if monthly_exps:
        nearest_date, nearest_str = monthly_exps[0]
        dte = (nearest_date - today).days

        # 若當月結算日小於 25 天，順延至下個月標準結算日 (相距約 28 天)
        if dte < 25 and len(monthly_exps) > 1:
            target_date, target_str = monthly_exps[1]
        else:
            target_date, target_str = nearest_date, nearest_str
    else:
        future_dates = [(d, d_str) for d, d_str in exp_dates if d >= today]
        if not future_dates:
            return None, None
        target_date, target_str = min(future_dates, key=lambda x: abs((x[0] - today).days - 35))

    target_dte = (target_date - today).days
    return target_str, target_dte

def fetch_volatility_metrics(symbol: str):
    """計算單一標的現價、HV、月選 IV、策略及連動權利金"""
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

        # 2. 抓取標準月選擇權合約
        expirations = ticker.options
        if not expirations:
            return None

        today = datetime.now().date()
        target_date_str, target_dte = select_target_monthly_expiration(expirations, today)
        if not target_date_str:
            return None

        chain = ticker.option_chain(target_date_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return None

        # 3. 基礎 IV 計算 (取最接近現價合約之均值)
        calls["diff"] = (calls["strike"] - spot).abs()
        atm_call_base = calls.sort_values("diff").iloc[0] if not calls.empty else None

        puts["diff"] = (puts["strike"] - spot).abs()
        atm_put_base = puts.sort_values("diff").iloc[0] if not puts.empty else None

        call_iv = float(atm_call_base.get("impliedVolatility", 0)) if atm_call_base is not None else 0
        put_iv = float(atm_put_base.get("impliedVolatility", 0)) if atm_put_base is not None else 0

        valid_ivs = [v for v in [call_iv, put_iv] if v > 0.05]
        if not valid_ivs:
            current_iv = call_iv if call_iv > 0 else put_iv
        else:
            current_iv = sum(valid_ivs) / len(valid_ivs)

        if current_iv <= 0:
            return None

        # 4. 決策策略判斷
        if current_iv >= 0.50:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
        elif current_iv <= 0.25:
            strategy = "Buy Call（IV偏低，適合買方進場）"
            strategy_tag = "BUY_CALL"
        else:
            strategy = "觀望 / 中性（無明顯優勢）"
            strategy_tag = "NEUTRAL"

        # 5. 權利金計算工具函式
        def calc_mid_price(row):
            if row is None:
                return 0.0
            b = float(row.get("bid", 0))
            a = float(row.get("ask", 0))
            l = float(row.get("lastPrice", 0))
            return round((b + a) / 2, 2) if b > 0 and a > 0 else round(l, 2)

        # 6. 策略連動邏輯：
        # - 僅有 Sell Put 時抓取 Strike <= 現價 的價外 Put 權利金
        # - Buy Call 或 觀望/中性 均不顯示權利金 (填入空字串 "")
        if strategy_tag == "SELL_PUT" and not puts.empty:
            otm_puts = puts[puts["strike"] <= spot]
            target_contract = otm_puts.sort_values("strike", ascending=False).iloc[0] if not otm_puts.empty else atm_put_base
            chosen_strike = float(target_contract["strike"]) if target_contract is not None else spot
            chosen_premium = calc_mid_price(target_contract)
        else:
            chosen_strike = ""
            chosen_premium = ""

        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else None

        return {
            "symbol": symbol,
            "spot": spot,
            "iv": current_iv,
            "hv": current_hv,
            "hv_52w_high": hv_52w_high,
            "hv_52w_low": hv_52w_low,
            "iv_hv_ratio": iv_hv_ratio,
            "strategy": strategy,
            "strategy_tag": strategy_tag,
            "premium": chosen_premium,                         # N 欄位：僅 Sell Put 有值，Buy Call 為 ""
            "strike": chosen_strike,                           # 履約價
            "exp_date": target_date_str,
            "dte": target_dte,
            "exp_info": f"{target_date_str} ({target_dte}天)",
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

        # 更新第 4 列標題
        sheet.update("P4:R4", [["期權到期日\n【自動填入】", "資料來源 / 備註", "更新日期"]])

        rows_to_insert = []
        for idx, r in enumerate(results, start=5):
            iv_val = r["iv"] / 100 if r["iv"] > 1.5 else r["iv"]
            hv_val = r["hv"] / 100 if r["hv"] > 1.5 else r["hv"]
            hv_h_val = r["hv_52w_high"] / 100 if r["hv_52w_high"] > 1.5 else r["hv_52w_high"]
            hv_l_val = r["hv_52w_low"] / 100 if r["hv_52w_low"] > 1.5 else r["hv_52w_low"]

            # O 欄位公式：若 N 欄為空，O 欄自動保持空白
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
                r["premium"],          # N: 選擇權權利金 (僅 Sell Put 顯示)
                formula_prem_pct,      # O: 權利金% (若 N 為空則顯示為空白)
                r["exp_info"],         # P: 期權到期日
                note,                  # Q: 資料來源 / 備註
                r["updated_date"]      # R: 更新日期
            ]
            rows_to_insert.append(row)

        print(f"準備寫入 Google Sheets (共 {len(rows_to_insert)} 筆)...")
        sheet.batch_clear(["A5:R"])
        sheet.update("A5", rows_to_insert, value_input_option="USER_ENTERED")
        print("成功將數據與 Sell Put 權利金寫入 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始掃描選擇權指標與 Sell Put 權利金 (共 {len(tickers)} 檔)...")

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

    # 1. 寫入 Google Sheets
    update_google_sheets(results)

    # 2. 存入 JSON 快取供 LINE 機器人即時查詢
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
