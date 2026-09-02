import os
import io
import json
import time
import math
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# 2 倍槓桿與基準指數 ETF
LEVERAGED_AND_BENCHMARK_ETFS = [
    "QLD", "SSO", "UWM", "USD", "ROM", "UYG",
    "NVDL", "TSLL", "MSTU", "MSTX", "CONL",
    "QID", "SDS",
    "SPY", "QQQ", "IWM", "SMH", "DIA"
]

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(is_call: bool, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """標準 Black-Scholes 期權定價公式"""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def implied_volatility_solver(is_call: bool, S: float, K: float, T: float, market_price: float, r: float = 0.045) -> float:
    """透過二分搜尋法從期權真實市價反推隱含波動率 (IV)"""
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if market_price <= intrinsic:
        return 0.0

    low = 0.01
    high = 4.0
    for _ in range(35):
        mid = (low + high) / 2.0
        p = bs_price(is_call, S, K, T, r, mid)
        if abs(p - market_price) < 1e-4:
            return mid
        if p < market_price:
            low = mid
        else:
            high = mid
    return mid

def get_tracking_tickers():
    """動態取得 S&P 500 成分股，並合併 2 倍槓桿與基準 ETF"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        tables = pd.read_html(io.StringIO(resp.text))
        sp500_tickers = [str(t).strip().replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    except Exception:
        sp500_tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V"]

    return sorted(list(set(sp500_tickers + LEVERAGED_AND_BENCHMARK_ETFS)))

def select_target_monthly_expiration(expirations, today):
    """篩選結算日後 25~30 天之標準月期權"""
    if not expirations:
        return None, None

    exp_dates = []
    for d_str in expirations:
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            exp_dates.append((d, d_str))
        except Exception:
            continue

    monthly_exps = [
        (d, d_str) for d, d_str in exp_dates
        if d.weekday() == 4 and 15 <= d.day <= 21 and d >= today
    ]
    monthly_exps.sort(key=lambda x: x[0])

    if monthly_exps:
        nearest_date, nearest_str = monthly_exps[0]
        dte = (nearest_date - today).days
        if dte < 25 and len(monthly_exps) > 1:
            target_date, target_str = monthly_exps[1]
        else:
            target_date, target_str = nearest_date, nearest_str
    else:
        future_dates = [(d, d_str) for d, d_str in exp_dates if d >= today]
        if not future_dates:
            return None, None
        target_date, target_str = min(future_dates, key=lambda x: abs((x[0] - today).days - 35))

    return target_str, (target_date - today).days

def fetch_volatility_metrics(symbol: str):
    """取得現價、精確計算 HV、反推真實 IV、連動 Sell Put 權利金"""
    try:
        ticker = yf.Ticker(symbol)

        # 1. 抓取歷史數據（2 年歷史以精準計算 52 週滾動區間）
        hist = ticker.history(period="2y")
        if hist.empty or len(hist) < 42:
            return None

        spot = float(hist["Close"].iloc[-1])
        spot = round(spot, 2)
        if spot <= 0:
            return None

        # 2. 計算 21 個交易日（對齊 thinkorswim 30 天日曆日）的滾動年化 HV
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1))
        rolling_hv = (log_ret.rolling(window=21).std() * np.sqrt(252)).dropna()
        if rolling_hv.empty:
            return None

        current_hv = float(rolling_hv.iloc[-1])
        # 取過去 252 個交易日（52 週）的 HV 極值
        hv_past_year = rolling_hv.iloc[-252:] if len(rolling_hv) >= 252 else rolling_hv
        hv_52w_high = float(hv_past_year.max())
        hv_52w_low = float(hv_past_year.min())

        # 3. 取得標準月期權鏈
        expirations = ticker.options
        if not expirations:
            return None

        today = datetime.now().date()
        target_date_str, target_dte = select_target_monthly_expiration(expirations, today)
        if not target_date_str or target_dte <= 0:
            return None

        chain = ticker.option_chain(target_date_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return None

        # 4. 尋找價平與價外合約並提取市價
        def get_mid_price(row):
            if row is None:
                return 0.0
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            last = float(row.get("lastPrice", 0))
            return round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else round(last, 2)

        calls["diff"] = (calls["strike"] - spot).abs()
        atm_call = calls.sort_values("diff").iloc[0] if not calls.empty else None
        call_mid = get_mid_price(atm_call)

        puts["diff"] = (puts["strike"] - spot).abs()
        atm_put = puts.sort_values("diff").iloc[0] if not puts.empty else None
        put_mid = get_mid_price(atm_put)

        # 5. 反推真實 IV（修正 Yahoo Finance 官方傳回 0.00001 之問題）
        T = target_dte / 365.0
        call_iv = implied_volatility_solver(True, spot, float(atm_call["strike"]), T, call_mid) if atm_call is not None else 0.0
        put_iv = implied_volatility_solver(False, spot, float(atm_put["strike"]), T, put_mid) if atm_put is not None else 0.0

        # 若反推成功則優先採用，否則備用原回傳值
        valid_ivs = [v for v in [call_iv, put_iv] if v > 0.08]
        if valid_ivs:
            current_iv = sum(valid_ivs) / len(valid_ivs)
        else:
            raw_c_iv = float(atm_call.get("impliedVolatility", 0)) if atm_call is not None else 0
            raw_p_iv = float(atm_put.get("impliedVolatility", 0)) if atm_put is not None else 0
            raw_valid = [v for v in [raw_c_iv, raw_p_iv] if v > 0.08]
            current_iv = sum(raw_valid) / len(raw_valid) if raw_valid else 0.20

        # 6. 建議策略決策框架（依據 IV/HV 比值與絕對 IV 水平）
        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else 1.0

        if iv_hv_ratio >= 1.25 or current_iv >= 0.50:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
            # 僅在 Sell Put 時抓取 Strike <= 現價 的價外 Put 權利金
            otm_puts = puts[puts["strike"] <= spot]
            target_put = otm_puts.sort_values("strike", ascending=False).iloc[0] if not otm_puts.empty else atm_put
            chosen_premium = get_mid_price(target_put)
            chosen_strike = float(target_put["strike"]) if target_put is not None else spot
        elif iv_hv_ratio <= 0.80 or current_iv <= 0.18:
            strategy = "Buy Call（IV偏低，適合買方進場）"
            strategy_tag = "BUY_CALL"
            chosen_premium = ""  # Buy Call 不顯示權利金
            chosen_strike = ""
        else:
            strategy = "觀望 / 中性（無明顯優勢）"
            strategy_tag = "NEUTRAL"
            chosen_premium = ""
            chosen_strike = ""

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
            "premium": chosen_premium,
            "strike": chosen_strike,
            "exp_date": target_date_str,
            "dte": target_dte,
            "exp_info": f"{target_date_str} ({target_dte}天)",
            "updated_date": datetime.now().strftime("%Y-%m-%d")
        }
    except Exception:
        return None

def update_google_sheets(results: list):
    """將修正後的正確數據與動態公式批次寫入 Google 試算表"""
    creds_json_str = os.environ.get("GOOGLE_CREDS_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not creds_json_str or not sheet_id:
        print("未設定 Google Sheets 憑證，略過寫入。")
        return

    try:
        creds_dict = json.loads(creds_json_str)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).worksheet("IV追蹤表")

        sheet.update("P4:R4", [["期權到期日\n【自動填入】", "資料來源 / 備註", "更新日期"]])

        rows_to_insert = []
        for idx, r in enumerate(results, start=5):
            # L 欄公式：自動計算 IV / HV 比值
            formula_ratio = f'=IF(OR($A{idx}="",$H{idx}="",$H{idx}=0),"",$C{idx}/$H{idx})'
            # O 欄公式：若 N 欄為空則自動顯示空白
            formula_prem_pct = f'=IF(OR($A{idx}="",$N{idx}="",$B{idx}="",$B{idx}=0),"",$N{idx}/$B{idx})'
            # G 欄公式：保留原生 IV Rank 公式結構
            formula_iv_rank = f'=IF(OR($A{idx}="",$D{idx}="",$E{idx}="",$D{idx}=$E{idx}),"",($C{idx}-$E{idx})/($D{idx}-$E{idx}))'

            note = "2倍槓桿/基準 ETF" if r["symbol"] in LEVERAGED_AND_BENCHMARK_ETFS else "S&P 500 成分股"

            row = [
                r["symbol"],               # A: 股票代號
                r["spot"],                 # B: 股價
                round(r["iv"], 4),         # C: 目前IV (真實 20%~45% 小數)
                "",                        # D: 52週IV高
                "",                        # E: 52週IV低
                "",                        # F: IV Percentile
                formula_iv_rank,           # G: IV Rank 公式
                round(r["hv"], 4),         # H: 目前 21D HV (對齊 ToS)
                round(r["hv_52w_high"], 4),# I: 52週HV高
                round(r["hv_52w_low"], 4), # J: 52週HV低
                "",                        # K: HV Percentile
                formula_ratio,             # L: IV/HV 比值公式
                r["strategy"],             # M: 建議策略
                r["premium"],              # N: 選擇權權利金 (僅 Sell Put 顯示)
                formula_prem_pct,          # O: 權利金% 公式
                r["exp_info"],             # P: 期權到期日
                note,                      # Q: 資料來源 / 備註
                r["updated_date"]          # R: 更新日期
            ]
            rows_to_insert.append(row)

        print(f"準備寫入 Google Sheets (共 {len(rows_to_insert)} 筆)...")
        sheet.batch_clear(["A5:R"])
        sheet.update("A5", rows_to_insert, value_input_option="USER_ENTERED")
        print("成功將精確波動率數據與公式寫入 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始多線程掃描波動率指標 (共 {len(tickers)} 檔)...")

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
    print(f"\n掃描結束！共計成功處理 {len(results)} 檔標的數據。")

    update_google_sheets(results)

    cache_payload = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "data": {item["symbol"]: item for item in results}
    }
    with open("iv_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    print("已成功更新 iv_cache.json 快取檔案。")

if __name__ == "__main__":
    main()
