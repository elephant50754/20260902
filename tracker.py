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

# 僅收錄美股市場所有主流「正向 2 倍槓桿 (2x Bull)」ETF
LEVERAGED_2X_BULL_ETFS = [
    # --- 指數型 正向 2 倍 ---
    "SSO",   # ProShares Ultra S&P500 (標普500 正向 2x)
    "QLD",   # ProShares Ultra QQQ (那斯達克100 正向 2x)
    "UWM",   # ProShares Ultra Russell2000 (羅素2000 正向 2x)
    "DDM",   # ProShares Ultra Dow30 (道瓊 正向 2x)

    # --- 產業板塊型 正向 2 倍 ---
    "USD",   # ProShares Ultra Semiconductors (半導體 正向 2x)
    "ROM",   # ProShares Ultra Technology (科技板塊 正向 2x)
    "UYG",   # ProShares Ultra Financials (金融板塊 正向 2x)
    "CURE",  # Direxion Daily Healthcare Bull 2X (醫療健康 正向 2x)
    "ERX",   # Direxion Daily Energy Bull 2X (能源板塊 正向 2x)
    "UXI",   # ProShares Ultra Industrials (工業板塊 正向 2x)
    "UCC",   # ProShares Ultra Consumer Services (非必需消費 正向 2x)

    # --- 熱門個股 / 科技主題 正向 2 倍 ---
    "NVDL",  # GraniteShares 2x Long NVDA Daily ETF (輝達 正向 2x)
    "TSLL",  # Direxion Daily TSLA Bull 2X Shares (特斯拉 正向 2x)
    "MSTU",  # T-Rex 2X Long MSTR Daily Target ETF (微策略 正向 2x)
    "MSTX",  # Defiance Daily Target 2X Long MSTR ETF (微策略 正向 2x)
    "CONL",  # GraniteShares 2x Long COIN Daily ETF (Coinbase 正向 2x)
    "AAPU",  # Direxion Daily AAPL Bull 2X Shares (蘋果 正向 2x)
    "MSFU",  # Direxion Daily MSFT Bull 2X Shares (微軟 正向 2x)
    "AMZU",  # Direxion Daily AMZN Bull 2X Shares (亞馬遜 正向 2x)
    "GGLL",  # Direxion Daily GOOGL Bull 2X Shares (Alphabet 正向 2x)
    "FBL",   # GraniteShares 2x Long META Daily ETF (Meta 正向 2x)
    "AMDL"   # GraniteShares 2x Long AMD Daily ETF (AMD 正向 2x)
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

    combined = sorted(list(set(sp500 + LEVERAGED_2X_BULL_ETFS)))
    print(f"篩選完成！共追蹤 {len(combined)} 檔標的（S&P 500 + 正向 2 倍槓桿 ETF）。")
    return combined

def get_mid_price(row):
    if row is None:
        return 0.0
    b = float(row.get("bid", 0))
    a = float(row.get("ask", 0))
    l = float(row.get("lastPrice", 0))
    return round((b + a) / 2, 2) if (b > 0 and a > 0) else round(l, 2)

def select_target_monthly_expiration(expirations, today):
    if not expirations:
        return None, None

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
        future_dates = [(d, d_str) for d, d_str in exp_dates if d >= today]
        if not future_dates:
            return None, None
        target_date, target_str = min(future_dates, key=lambda x: abs((x[0] - today).days - 35))
        return target_str, (target_date - today).days

    exp1_date, exp1_str = monthly_exps[0]
    dte1 = (exp1_date - today).days

    if len(monthly_exps) > 1:
        exp2_date, exp2_str = monthly_exps[1]
        dte2 = (exp2_date - today).days
    else:
        exp2_date, exp2_str, dte2 = exp1_date, exp1_str, dte1

    target_date_str = exp2_str if dte1 < 25 and len(monthly_exps) > 1 else exp1_str
    target_dte = dte2 if dte1 < 25 and len(monthly_exps) > 1 else dte1
    return target_date_str, target_dte

def get_chain_robust_iv(ticker, spot, exp_str, dte):
    try:
        chain = ticker.option_chain(exp_str)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty and puts.empty:
            return None, chain

        T = dte / 365.0
        iv_pool = []

        if not calls.empty:
            calls["diff"] = (calls["strike"] - spot).abs()
            top_calls = calls.sort_values("diff").head(3)
            for _, c_row in top_calls.iterrows():
                mid = get_mid_price(c_row)
                k = float(c_row["strike"])
                iv_s = implied_volatility_solver(True, spot, k, T, mid)
                iv_raw = float(c_row.get("impliedVolatility", 0))
                for v in [iv_s, iv_raw]:
                    if 0.08 <= v <= 2.5:
                        iv_pool.append(v)

        if not puts.empty:
            puts["diff"] = (puts["strike"] - spot).abs()
            top_puts = puts.sort_values("diff").head(3)
            for _, p_row in top_puts.iterrows():
                mid = get_mid_price(p_row)
                k = float(p_row["strike"])
                iv_s = implied_volatility_solver(False, spot, k, T, mid)
                iv_raw = float(p_row.get("impliedVolatility", 0))
                for v in [iv_s, iv_raw]:
                    if 0.08 <= v <= 2.5:
                        iv_pool.append(v)

        if iv_pool:
            iv_pool.sort()
            trim_len = max(1, int(len(iv_pool) * 0.15))
            valid_subset = iv_pool[trim_len:-trim_len] if len(iv_pool) > 4 else iv_pool
            robust_iv = sum(valid_subset) / len(valid_subset)
        else:
            robust_iv = 0.25

        return robust_iv, chain
    except Exception:
        return None, None

def fetch_volatility_metrics(symbol: str):
    for attempt in range(2):
        try:
            ticker = yf.Ticker(symbol)

            # 1. 抓取過去 18 個月完整日收盤價序列以回推 252 交易日歷史序列
            hist = ticker.history(period="18mo")
            if hist.empty or len(hist) < 273:
                if len(hist) < 35:
                    return None

            spot = float(hist["Close"].iloc[-1])
            spot = round(spot, 2)
            if spot <= 0:
                return None

            # 2. 計算全歷史過去 252 個交易日的 21 日滾動 HV 序列
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1))
            rolling_hv = (log_ret.rolling(window=21).std(ddof=1) * np.sqrt(252)).dropna()
            if rolling_hv.empty:
                return None

            # 過去一年（252 個交易日）的完整 HV 序列
            past_252d_hv = rolling_hv.iloc[-252:] if len(rolling_hv) >= 252 else rolling_hv
            current_hv = float(past_252d_hv.iloc[-1])

            hv_high = float(past_252d_hv.max())
            hv_low = float(past_252d_hv.min())

            # 3. 取得期權鏈並算出當前絕對 IV
            expirations = ticker.options
            if not expirations:
                return None

            today = datetime.now().date()
            target_date_str, target_dte = select_target_monthly_expiration(expirations, today)
            if not target_date_str or target_dte <= 0:
                return None

            current_iv_raw, target_chain = get_chain_robust_iv(ticker, spot, target_date_str, target_dte)
            if current_iv_raw is None or current_iv_raw <= 0.01:
                current_iv_raw = current_hv * 1.35

            current_iv_abs = sanitize_float(current_iv_raw, default=0.25)

            # 4. 【核心演算法：天數比例 IV Percentile 嚴格統計回推】
            # 用過去 252 個交易日中，回推每日 IV_t <= 目前 IV 的天數，除以總交易天數
            vrp_ratio = current_iv_abs / current_hv if current_hv > 0 else 1.25
            daily_iv_series = past_252d_hv * vrp_ratio

            # 統計小於或等於當前 IV 的天數比例
            days_below = (daily_iv_series <= current_iv_abs).sum()
            total_days = len(daily_iv_series)
            iv_percentile = sanitize_float(days_below / total_days, default=0.50)

            # 52週 IV 高低點取回推序列的歷史最大與最小值
            iv_52w_high = sanitize_float(daily_iv_series.max(), default=current_iv_abs * 1.2)
            iv_52w_low = sanitize_float(daily_iv_series.min(), default=current_iv_abs * 0.8)

            iv_hv_ratio = round(current_iv_abs / current_hv, 2) if current_hv > 0 else 1.0

            # 5. 策略判定與 44 天價外 Put 權利金連動
            puts = target_chain.puts.copy() if target_chain is not None and not target_chain.puts.empty else pd.DataFrame()

            if iv_percentile >= 0.48 or iv_hv_ratio >= 1.25:
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
            elif iv_percentile <= 0.25 or iv_hv_ratio <= 0.80:
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
                "iv": iv_percentile,                               # C 欄直接呈現天數統計之 IV Percentile (0~100%)
                "iv_abs": current_iv_abs,                          # 絕對 IV
                "iv_pct": iv_percentile,                           # 天數百分比
                "iv_52w_high": iv_52w_high,                        # 52週 IV 高
                "iv_52w_low": iv_52w_low,                          # 52週 IV 低
                "hv": current_hv,                                  # 目前 HV
                "hv_52w_high": hv_high,                            # 52週 HV 高
                "hv_52w_low": hv_low,                              # 52週 HV 低
                "iv_hv_ratio": iv_hv_ratio,                        # 比值
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
            time.sleep(0.5)
            continue
    return None

def update_google_sheets(results: list):
    if len(results) < 350:
        print(f"⚠️ 標的數量不足 ({len(results)} 檔)，略過寫入以防異常。")
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
            # K 欄公式：HV Percentile 自動計算
            formula_hv_pct = f'=IF(OR($A{idx}="",$I{idx}="",$J{idx}="",$I{idx}=$J{idx}),"",($H{idx}-$J{idx})/($I{idx}-$J{idx}))'
            # L 欄公式：IV/HV 比值自動計算
            formula_ratio = f'{r["iv_hv_ratio"]}x'
            # O 欄公式：權利金佔比
            formula_prem_pct = f'=IF(OR($A{idx}="",$N{idx}="",$B{idx}="",$B{idx}=0),"",$N{idx}/$B{idx})'

            note = "正向 2 倍槓桿 ETF (Yahoo天數統計)" if r["symbol"] in LEVERAGED_2X_BULL_ETFS else "S&P 500 成分股 (Yahoo天數統計)"

            row = [
                r["symbol"],                           # A: 股票代號
                sanitize_float(r["spot"]),             # B: 股價
                sanitize_float(r["iv_pct"]),           # C: 目前 IV 直接填入天數佔比之 IV Percentile
                sanitize_float(r["iv_52w_high"]),      # D: 52週IV高
                sanitize_float(r["iv_52w_low"]),       # E: 52週IV低
                sanitize_float(r["iv_pct"]),           # F: IV Percentile (天數百分比)
                sanitize_float(r["iv_pct"]),           # G: IV Rank
                sanitize_float(r["hv"]),               # H: 目前HV
                sanitize_float(r["hv_52w_high"]),      # I: 52週HV高
                sanitize_float(r["hv_52w_low"]),       # J: 52週HV低
                formula_hv_pct,                        # K: HV Percentile 公式
                formula_ratio,                         # L: IV/HV 比值
                r["strategy"],                         # M: 建議策略
                r["premium"],                          # N: 選擇權權利金 (僅 Sell Put)
                formula_prem_pct,                      # O: 權利金% 公式
                r["exp_info"],                         # P: 期權到期日
                note,                                  # Q: 資料來源 / 備註
                r["updated_date"]                      # R: 更新日期
            ]
            rows_to_insert.append(row)

        end_row = 4 + len(rows_to_insert)
        target_range = f"A5:R{end_row}"
        print(f"正在直接覆蓋寫入 Google Sheets ({target_range}，共 {len(rows_to_insert)} 筆)...")
        sheet.batch_clear(["A5:R"])
        sheet.update(range_name=target_range, values=rows_to_insert, value_input_option="USER_ENTERED")
        print("✅ 天數佔比型 IV Percentile 已成功同步至 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始透過 Yahoo Finance 執行 252 交易日天數佔比 IV Percentile 運算 (總計 {len(tickers)} 檔)...")

    results = []
    # 限制 2 線程並間隔 0.35 秒，保護 IP 防止 Yahoo 429 阻斷
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_symbol = {executor.submit(fetch_volatility_metrics, sym): sym for sym in tickers}
        completed = 0
        total = len(tickers)

        for future in as_completed(future_to_symbol):
            completed += 1
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 25 == 0:
                    print(f"進度 [{completed}/{total}] | 已成功處理 {len(results)} 檔標的...")
            time.sleep(0.35)

    results.sort(key=lambda x: x["symbol"])
    print(f"\n處理結束！有效數據: {len(results)} 檔。")

    update_google_sheets(results)

    cache_payload = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "data": {item["symbol"]: item for item in results}
    }
    with open("iv_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    print("iv_cache.json 已成功同步。")

if __name__ == "__main__":
    main()
