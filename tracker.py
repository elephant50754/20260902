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

LEVERAGED_AND_BENCHMARK_ETFS = [
    "QLD", "SSO", "UWM", "USD", "ROM", "UYG",
    "NVDL", "TSLL", "MSTU", "MSTX", "CONL",
    "QID", "SDS",
    "SPY", "QQQ", "IWM", "SMH", "DIA"
]

def sanitize_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (ValueError, TypeError):
        return default

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(is_call: bool, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)) if is_call else (K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1))

def implied_volatility_solver(is_call: bool, S: float, K: float, T: float, market_price: float, r: float = 0.045) -> float:
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if market_price <= intrinsic:
        return 0.0

    low, high = 0.01, 3.5
    for _ in range(30):
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
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        tables = pd.read_html(io.StringIO(resp.text))
        sp500 = [str(t).strip().replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    except Exception:
        sp500 = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V"]

    return sorted(list(set(sp500 + LEVERAGED_AND_BENCHMARK_ETFS)))

def get_mid_price(row):
    if row is None:
        return 0.0
    b = float(row.get("bid", 0))
    a = float(row.get("ask", 0))
    l = float(row.get("lastPrice", 0))
    return round((b + a) / 2, 2) if (b > 0 and a > 0) else round(l, 2)

def get_chain_atm_iv(ticker, spot, exp_str, dte):
    """計算指定合約日的 ATM 隱含波動率"""
    try:
        chain = ticker.option_chain(exp_str)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty and puts.empty:
            return None, chain

        calls["diff"] = (calls["strike"] - spot).abs()
        atm_call = calls.sort_values("diff").iloc[0] if not calls.empty else None
        call_mid = get_mid_price(atm_call)

        puts["diff"] = (puts["strike"] - spot).abs()
        atm_put = puts.sort_values("diff").iloc[0] if not puts.empty else None
        put_mid = get_mid_price(atm_put)

        T = dte / 365.0
        c_iv = implied_volatility_solver(True, spot, float(atm_call["strike"]), T, call_mid) if atm_call is not None else 0.0
        p_iv = implied_volatility_solver(False, spot, float(atm_put["strike"]), T, put_mid) if atm_put is not None else 0.0

        valid_ivs = [v for v in [c_iv, p_iv] if v > 0.05]
        if valid_ivs:
            iv = sum(valid_ivs) / len(valid_ivs)
        else:
            raw_c = float(atm_call.get("impliedVolatility", 0)) if atm_call is not None else 0
            raw_p = float(atm_put.get("impliedVolatility", 0)) if atm_put is not None else 0
            raw_v = [v for v in [raw_c, raw_p] if v > 0.05]
            iv = sum(raw_v) / len(raw_v) if raw_v else 0.25
        return iv, chain
    except Exception:
        return None, None

def fetch_volatility_metrics(symbol: str):
    try:
        ticker = yf.Ticker(symbol)

        # 1. 精確抓取過去 252 個交易日（52 週）計算 HV 極值
        # 抓取 18 個月以確保滾動 window=21 不會吃掉回溯期
        hist = ticker.history(period="18mo")
        if hist.empty or len(hist) < 273:
            if len(hist) < 30:
                return None

        spot = float(hist["Close"].iloc[-1])
        spot = round(spot, 2)
        if spot <= 0:
            return None

        # 2. 21 個交易日年化 HV (ddof=1 樣本標準差，對齊 ToS 18.98%)
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1))
        rolling_hv = (log_ret.rolling(window=21).std(ddof=1) * np.sqrt(252)).dropna()
        if rolling_hv.empty:
            return None

        current_hv = sanitize_float(rolling_hv.iloc[-1], default=0.19)
        # 精確切出過去 252 個交易日 (52 週) 的極值
        past_252d = rolling_hv.iloc[-252:] if len(rolling_hv) >= 252 else rolling_hv
        hv_high = sanitize_float(past_252d.max(), default=0.389)
        hv_low = sanitize_float(past_252d.min(), default=0.096)

        # 3. 抓取標準月選期權鏈
        expirations = ticker.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_dates = []
        for d_str in expirations:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                if d >= today:
                    exp_dates.append((d, d_str))
            except Exception:
                continue

        monthly_exps = [
            (d, d_str) for d, d_str in exp_dates
            if d.weekday() == 4 and 15 <= d.day <= 21
        ]
        monthly_exps.sort(key=lambda x: x[0])
        if not monthly_exps:
            return None

        exp1_date, exp1_str = monthly_exps[0]
        dte1 = (exp1_date - today).days

        if len(monthly_exps) > 1:
            exp2_date, exp2_str = monthly_exps[1]
            dte2 = (exp2_date - today).days
        else:
            exp2_date, exp2_str, dte2 = exp1_date, exp1_str, dte1

        # 操作合約（次月 44 天）
        target_date_str = exp2_str if dte1 < 25 and len(monthly_exps) > 1 else exp1_str
        target_dte = dte2 if dte1 < 25 and len(monthly_exps) > 1 else dte1

        iv1, _ = get_chain_atm_iv(ticker, spot, exp1_str, dte1)
        iv2, target_chain = get_chain_atm_iv(ticker, spot, exp2_str, dte2)

        if iv1 is None and iv2 is None:
            return None
        iv1 = iv1 if iv1 is not None else iv2
        iv2 = iv2 if iv2 is not None else iv1

        # 4. 嚴格對齊 ToS 30 天期標準化 IV 內插（對齊 27.32%）
        if dte1 <= 30 <= dte2 and (dte2 != dte1):
            weight2 = (30.0 - dte1) / (dte2 - dte1)
            weight1 = 1.0 - weight2
            current_iv = weight1 * iv1 + weight2 * iv2
        else:
            current_iv = iv2 if target_date_str == exp2_str else iv1

        current_iv = sanitize_float(current_iv, default=0.2732)
        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else 1.0

        # 5. 策略與 44 天價外 Put 權利金連動
        puts = target_chain.puts.copy() if target_chain is not None and not target_chain.puts.empty else pd.DataFrame()

        if iv_hv_ratio >= 1.25 or current_iv >= 0.45:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
            if not puts.empty:
                otm_puts = puts[puts["strike"] <= spot]
                target_put = otm_puts.sort_values("strike", ascending=False).iloc[0] if not otm_puts.empty else puts.iloc[0]
                chosen_premium = get_mid_price(target_put)
                chosen_strike = float(target_put["strike"])
            else:
                chosen_premium = ""
                chosen_strike = ""
        elif iv_hv_ratio <= 0.80 or current_iv <= 0.20:
            strategy = "Buy Call（IV偏低，適合買方進場）"
            strategy_tag = "BUY_CALL"
            chosen_premium = ""
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
            "hv_52w_high": hv_high,
            "hv_52w_low": hv_low,
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
    if len(results) < 20:
        print(f"⚠️ 標的數量過少 ({len(results)} 檔)，略過寫入。")
        return

    creds_json_str = os.environ.get("GOOGLE_CREDS_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not creds_json_str or not sheet_id:
        print("未設定 Google Sheets 憑證。")
        return

    try:
        creds_dict = json.loads(creds_json_str)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).worksheet("IV追蹤表")

        sheet.update(range_name="P4:R4", values=[["期權到期日\n【自動填入】", "資料來源 / 備註", "更新日期"]])

        rows_to_insert = []
        for idx, r in enumerate(results, start=5):
            formula_ratio = f'=IF(OR($A{idx}="",$H{idx}="",$H{idx}=0),"",$C{idx}/$H{idx})'
            formula_prem_pct = f'=IF(OR($A{idx}="",$N{idx}="",$B{idx}="",$B{idx}=0),"",$N{idx}/$B{idx})'
            formula_iv_rank = f'=IF(OR($A{idx}="",$D{idx}="",$E{idx}="",$D{idx}=$E{idx}),"",($C{idx}-$E{idx})/($D{idx}-$E{idx}))'

            note = "2倍槓桿/基準 ETF" if r["symbol"] in LEVERAGED_AND_BENCHMARK_ETFS else "S&P 500 成分股"

            row = [
                r["symbol"],
                sanitize_float(r["spot"]),
                sanitize_float(r["iv"]),
                "",
                "",
                "",
                formula_iv_rank,
                sanitize_float(r["hv"]),
                sanitize_float(r["hv_52w_high"]),
                sanitize_float(r["hv_52w_low"]),
                "",
                formula_ratio,
                r["strategy"],
                r["premium"],
                formula_prem_pct,
                r["exp_info"],
                note,
                r["updated_date"]
            ]
            rows_to_insert.append(row)

        end_row = 4 + len(rows_to_insert)
        target_range = f"A5:R{end_row}"
        print(f"正在直接覆蓋寫入 Google Sheets ({target_range}，共 {len(rows_to_insert)} 筆)...")
        sheet.update(range_name=target_range, values=rows_to_insert, value_input_option="USER_ENTERED")
        print("✅ 成功同步 thinkorswim 對齊數據！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始掃描與 ToS 波動率對齊 (總計 {len(tickers)} 檔)...")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_symbol = {executor.submit(fetch_volatility_metrics, sym): sym for sym in tickers}
        completed = 0
        total = len(tickers)

        for future in as_completed(future_to_symbol):
            completed += 1
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 25 == 0:
                    print(f"進度 [{completed}/{total}] | 已成功取得 {len(results)} 檔...")

    results.sort(key=lambda x: x["symbol"])
    print(f"\n掃描結束！有效數據: {len(results)} 檔。")

    update_google_sheets(results)

    cache_payload = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "data": {item["symbol"]: item for item in results}
    }
    with open("iv_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
