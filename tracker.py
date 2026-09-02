import os
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

def get_entire_us_market_tickers():
    """
    從 NASDAQ 官方交易所 FTP 伺服器動態下載全美股最新名單 (涵蓋 NYSE, NASDAQ, AMEX 11,000+ 檔)
    """
    print("正在從 NASDAQ 官方母名單下載全美股代碼清單...")
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
    try:
        df = pd.read_csv(url, sep="|")
        # 過濾掉測試代碼 (Test Issue == 'Y') 與無效代碼
        df = df[df["Test Issue"] == "N"]
        raw_symbols = df["Symbol"].dropna().tolist()
        
        # 轉換為 Yahoo Finance 支援的代碼格式 (如 BRK.B 轉 BRK-B)
        cleaned_symbols = [
            str(s).strip().replace(".", "-").replace("$", "-")
            for s in raw_symbols
            if str(s).strip() != ""
        ]
        unique_symbols = sorted(list(set(cleaned_symbols)))
        print(f"成功取得全美股掛牌名單，總計 {len(unique_symbols)} 檔標的。")
        return unique_symbols
    except Exception as e:
        print(f"NASDAQ 名單下載失敗: {e}，改用備用核心指數名單...")
        return ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "SPY", "QQQ", "IWM"]

def process_single_stock(symbol: str):
    """處理單一股票：計算 HV、IV、比值與策略"""
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. 快速檢查是否具備選擇權 (若無選擇權則直接跳過，大幅節省時間)
        expirations = ticker.options
        if not expirations:
            return None

        # 2. 抓取歷史價格計算 30天年化歷史波動率 (HV)
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

        # 3. 尋找最接近 30 天的平值 (ATM) IV
        today = datetime.now().date()
        exp_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in expirations]
        target_date = min(exp_dates, key=lambda d: abs((d - today).days - 30))
        target_date_str = target_date.strftime("%Y-%m-%d")

        calls = ticker.option_chain(target_date_str).calls.copy()
        if calls.empty:
            return None

        calls["diff"] = (calls["strike"] - spot).abs()
        atm_call = calls.sort_values("diff").iloc[0]
        current_iv = atm_call.get("impliedVolatility")

        if current_iv is None or pd.isna(current_iv) or current_iv <= 0:
            return None

        current_iv = float(current_iv)
        iv_hv_ratio = round(current_iv / current_hv, 2) if current_hv > 0 else None

        # 4. 決策策略標記 (對齊 Excel 表格規則)
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
            "iv": round(current_iv * 100, 2),
            "hv": round(current_hv * 100, 2),
            "hv_52w_high": round(hv_52w_high * 100, 2),
            "hv_52w_low": round(hv_52w_low * 100, 2),
            "iv_hv_ratio": iv_hv_ratio,
            "strategy": strategy,
            "strategy_tag": strategy_tag,
            "exp_date": target_date_str,
            "dte": (target_date - today).days,
            "strike": float(atm_call["strike"]),
            "updated_date": datetime.now().strftime("%Y-%m-%d")
        }
    except Exception:
        return None

def update_google_sheets(results: list):
    """將全市場結果批次寫入 Google 試算表"""
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
        for r in results:
            row = [
                r["symbol"], r["spot"], r["iv"] / 100, "", "", "", "",
                r["hv"] / 100, r["hv_52w_high"] / 100, r["hv_52w_low"] / 100,
                "", r["iv_hv_ratio"], r["strategy"], "", "",
                "全美股官方清單掃描", r["updated_date"]
            ]
            rows_to_insert.append(row)

        print(f"準備寫入 Google Sheets (共 {len(rows_to_insert)} 筆)...")
        # 清除先前資料並單次批次整批寫入
        sheet.batch_clear(["A5:Q"])
        sheet.update("A5", rows_to_insert, value_input_option="USER_ENTERED")
        print("成功將全美股具選擇權標的寫入 Google Sheets！")
    except Exception as e:
        print(f"寫入 Google Sheets 失敗: {e}")

def main():
    all_tickers = get_entire_us_market_tickers()
    print(f"啟動全市場多線程並行掃描 (總池：{len(all_tickers)} 檔)...")

    results = []
    # 採用 6 個工作線程並發，兼顧掃描速度與避免觸發 Yahoo API 頻率限制
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_symbol = {executor.submit(process_single_stock, sym): sym for sym in all_tickers}
        completed_count = 0
        total = len(all_tickers)

        for future in as_completed(future_to_symbol):
            completed_count += 1
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 50 == 0:
                    print(f"進度 [{completed_count}/{total}] | 已找到 {len(results)} 檔具備選擇權標的...")

    print(f"\n全市場掃描結束！全美股共計 {len(results)} 檔標的具備有效選擇權與 IV 數據。")

    # 1. 寫入 Google Sheets
    update_google_sheets(results)

    # 2. 寫入 iv_cache.json 快取檔案 (供 LINE 機器人即時過濾使用)
    cache_payload = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "data": {item["symbol"]: item for item in results}
    }
    with open("iv_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    print("已成功產出全美股 iv_cache.json 快取！")

if __name__ == "__main__":
    main()
