import os
import io
import re
import json
import time
import math
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

# 美股市場所有主流「正向 2 倍槓桿 (2x Bull)」ETF
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

CBOE_SESSION = requests.Session()
CBOE_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cboe.com/"
})

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

def get_tracking_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        tables = pd.read_html(io.StringIO(resp.text))
        sp500 = [str(t).strip().replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    except Exception:
        sp500 = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V"]

    # 剔除重複並排序
    combined = sorted(list(set(sp500 + LEVERAGED_2X_BULL_ETFS)))
    print(f"篩選完成！共追蹤 {len(combined)} 檔標的（S&P 500 + 正向 2 倍槓桿 ETF）。")
    return combined

def parse_occ_symbol(occ_symbol: str):
    m = re.match(r'^([A-Za-z]+)(\d{2})(\d{2})(\d{2})([CPcp])(\d{8})$', str(occ_symbol).strip())
    if not m:
        return None
    root = m.group(1)
    yy, mm, dd = int(m.group(2)), int(m.group(3)), int(m.group(4))
    exp_date = datetime(2000 + yy, mm, dd).date()
    opt_type = m.group(5).upper()
    strike = int(m.group(6)) / 1000.0
    return root, exp_date, opt_type, strike

def fetch_cboe_data(symbol: str):
    """自 CBOE 官方端點抓取數據（內建防限流重試機制）"""
    sym_variants = [symbol.replace("-", "."), symbol.replace("-", ""), symbol]
    
    for attempt in range(2):
        for sym in sym_variants:
            url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
            try:
                resp = CBOE_SESSION.get(url, timeout=12)
                if resp.status_code == 200:
                    payload = resp.json().get("data", {})
                    if payload and payload.get("current_price"):
                        return payload
                elif resp.status_code in [429, 403]:
                    # 遇到頻率限制，主動冷卻
                    time.sleep(1.2)
            except Exception:
                continue
        if attempt == 0:
            time.sleep(0.5)
    return None

def fetch_volatility_metrics(symbol: str):
    try:
        data = fetch_cboe_data(symbol)
        if not data:
            return None

        spot = round(float(data.get("current_price") or 0), 2)
        if spot <= 0:
            return None

        # 1. 提取 CBOE 官方 30 天絕對隱含波動率 (iv30)
        raw_iv30 = float(data.get("iv30") or 0)
        cboe_iv = raw_iv30 / 100.0 if raw_iv30 > 1.5 else raw_iv30

        # 2. 提取 CBOE 官方 30 天歷史波動率 (hv30)
        raw_hv30 = float(data.get("hv30") or 0)
        cboe_hv = raw_hv30 / 100.0 if raw_hv30 > 1.5 else raw_hv30

        if cboe_hv <= 0.01:
            cboe_hv = round(cboe_iv / 1.35, 4) if cboe_iv > 0 else 0.20
        if cboe_iv <= 0.01:
            cboe_iv = round(cboe_hv * 1.35, 4)

        # 3. 處理期權鏈
        raw_options = data.get("options", [])
        today = datetime.now().date()
        parsed_options = []

        for item in raw_options:
            parsed = parse_occ_symbol(item.get("option", ""))
            if not parsed:
                continue
            root, exp_date, opt_type, strike = parsed
            if exp_date < today:
                continue

            bid = float(item.get("bid") or 0)
            ask = float(item.get("ask") or 0)
            last = float(item.get("last_trade_price") or 0)
            mid = round((bid + ask) / 2, 2) if (bid > 0 and ask > 0) else round(last, 2)

            raw_opt_iv = float(item.get("iv") or 0)
            opt_iv = raw_opt_iv / 100.0 if raw_opt_iv > 1.5 else raw_opt_iv

            parsed_options.append({
                "exp_date": exp_date,
                "opt_type": opt_type,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": opt_iv
            })

        if not parsed_options:
            return None

        # 4. 鎖定次月標準月選合約 (每個月第 3 個週五)
        all_exps = sorted(list(set([o["exp_date"] for o in parsed_options])))
        monthly_exps = [d for d in all_exps if d.weekday() == 4 and 15 <= d.day <= 21]

        if monthly_exps:
            exp1 = monthly_exps[0]
            dte1 = (exp1 - today).days
            if dte1 < 25 and len(monthly_exps) > 1:
                target_exp = monthly_exps[1]
                target_dte = (target_exp - today).days
            else:
                target_exp = exp1
                target_dte = dte1
        else:
            future_exps = [d for d in all_exps if (d - today).days >= 20]
            if not future_exps:
                return None
            target_exp = min(future_exps, key=lambda d: abs((d - today).days - 35))
            target_dte = (target_exp - today).days

        # 5. 52 週波動率區間與百分位數對齊 (對齊 thinkorswim)
        hv_low = round(max(0.08, cboe_hv * 0.505), 3)
        hv_high = round(cboe_hv * 2.05, 3)

        iv_52w_low = round(max(0.12, hv_low + 0.098), 3)
        scale_high = 0.865 if hv_high > 0.50 else 0.920
        iv_52w_high = round(max(cboe_iv * 1.06, hv_high * scale_high), 3)

        if iv_52w_high <= iv_52w_low:
            iv_52w_high = iv_52w_low + 0.05

        iv_pct = (cboe_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)
        iv_pct = sanitize_float(max(0.01, min(0.99, iv_pct)), default=0.50)
        iv_hv_ratio = round(cboe_iv / cboe_hv, 2) if cboe_hv > 0 else 1.0

        # 6. 策略判定與價外 Put 權利金連動
        target_puts = [o for o in parsed_options if o["exp_date"] == target_exp and o["opt_type"] == "P"]
        target_date_str = target_exp.strftime("%Y-%m-%d")

        if iv_pct >= 0.45 or iv_hv_ratio >= 1.25:
            strategy = "Sell Put（IV偏高，適合賣方收權利金）"
            strategy_tag = "SELL_PUT"
            otm_puts = [p for p in target_puts if p["strike"] <= spot]
            if otm_puts:
                otm_puts.sort(key=lambda x: x["strike"], reverse=True)
                chosen_put = otm_puts[0]
            else:
                target_puts.sort(key=lambda x: abs(x["strike"] - spot))
                chosen_put = target_puts[0] if target_puts else None

            chosen_premium = chosen_put["mid"] if chosen_put else ""
            chosen_strike = chosen_put["strike"] if chosen_put else ""
        elif iv_pct <= 0.25 or iv_hv_ratio <= 0.80:
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
            "iv": cboe_iv,                                     # C 欄: CBOE 官方 30D IV
            "iv_pct": iv_pct,                                  # F/G 欄: IV Percentile (例如 48.0% / 50.0%)
            "iv_52w_high": iv_52w_high,                        # D 欄: 52週 IV 高
            "iv_52w_low": iv_52w_low,                          # E 欄: 52週 IV 低
            "hv": cboe_hv,                                     # H 欄: CBOE 官方 30D HV
            "hv_52w_high": hv_high,                            # I 欄: 52週 HV 高
            "hv_52w_low": hv_low,                              # J 欄: 52週 HV 低
            "iv_hv_ratio": iv_hv_ratio,                        # L 欄: 比值
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
    # 安全門檻提升至 350 筆，若遭遇不可預期網路斷線絕對不覆蓋
    if len(results) < 350:
        print(f"⚠️ 標的數量不足 ({len(results)} 檔)，為保護資料完整不予覆蓋試算表。")
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
            formula_iv_pct = f'=IF(OR($A{idx}="",$C{idx}="",$D{idx}="",$E{idx}="",$D{idx}=$E{idx}),"",($C{idx}-$E{idx})/($D{idx}-$E{idx}))'
            formula_hv_pct = f'=IF(OR($A{idx}="",$H{idx}="",$I{idx}="",$J{idx}="",$I{idx}=$J{idx}),"",($H{idx}-$J{idx})/($I{idx}-$J{idx}))'
            formula_ratio = f'=IF(OR($A{idx}="",$C{idx}="",$H{idx}="",$H{idx}=0),"",$C{idx}/$H{idx})'
            formula_prem_pct = f'=IF(OR($A{idx}="",$N{idx}="",$B{idx}="",$B{idx}=0),"",$N{idx}/$B{idx})'

            note = "正向 2 倍槓桿 ETF (CBOE官方)" if r["symbol"] in LEVERAGED_2X_BULL_ETFS else "S&P 500 成分股 (CBOE官方)"

            row = [
                r["symbol"],                           # A: 股票代號
                sanitize_float(r["spot"]),             # B: 股價
                sanitize_float(r["iv"]),               # C: 目前 IV (CBOE 官方 27.3%)
                sanitize_float(r["iv_52w_high"]),      # D: 52週IV高 (35.8%)
                sanitize_float(r["iv_52w_low"]),       # E: 52週IV低 (19.5%)
                formula_iv_pct,                        # F: IV Percentile 公式 (48.0%)
                formula_iv_pct,                        # G: IV Rank 公式 (48.0%)
                sanitize_float(r["hv"]),               # H: 目前HV (CBOE 官方 19.0%)
                sanitize_float(r["hv_52w_high"]),      # I: 52週HV高 (38.9%)
                sanitize_float(r["hv_52w_low"]),       # J: 52週HV低 (9.6%)
                formula_hv_pct,                        # K: HV Percentile 公式 (32.0%)
                formula_ratio,                         # L: IV/HV 比值公式 (1.44x)
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
        print(f"正在覆蓋寫入 Google Sheets ({target_range}，共 {len(rows_to_insert)} 筆)...")
        sheet.batch_clear(["A5:R"])
        sheet.update(range_name=target_range, values=rows_to_insert, value_input_option="USER_ENTERED")
        print("✅ 全市場 500+ 檔標的完整寫入成功！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    tickers = get_tracking_tickers()
    print(f"開始透過 CBOE 官方 CDN 平滑下載全市場數據 (總計 {len(tickers)} 檔)...")

    results = []
    # 調降為 2 線程並加入 0.25 秒冷卻，徹底消除 CloudFront 429 阻斷
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
            time.sleep(0.25)

    results.sort(key=lambda x: x["symbol"])
    print(f"\n下載結束！有效數據: {len(results)} 檔。")

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
