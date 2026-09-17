#!/usr/bin/env python3
"""
KOSPI200 · KOSDAQ150 스윙 스크리너 (보유 1주 기준)

기준: 오닐(CANSLIM의 M·L·I), 와인스타인 단계분석, 미너비니 트렌드 템플릿,
      다바스 박스 돌파, 7% 손절 원칙을 1주 스윙에 맞게 조정.

사용법
  pip install pykrx pandas
  python swing_screener.py                 # 오늘(또는 직전 거래일) 기준
  python swing_screener.py --date 20260917 # 특정일 기준
  python swing_screener.py --no-flow       # 수급 조회 생략(빠름)

실행 시점
  - 가격·거래량: 정규장 마감 후 16:00 이후
  - 외국인·기관 수급 확정치: 18:00 이후 권장

결과
  output/swing_YYYYMMDD.html  (리포트)
  output/swing_YYYYMMDD.csv   (전 종목 점수)
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────── 설정 ───────────────────────────────
CFG = {
    "min_value_20d": 5e9,        # 20일 평균 거래대금 하한 (50억원)
    "vol_mult": 1.5,             # 돌파일 거래량 ≥ 50일 평균 × 1.5
    "pivot_lookback": 20,        # 피벗 = 직전 20거래일 최고가
    "max_extension": 0.05,       # 피벗 대비 +5% 넘으면 추격 금지
    "near_high_ratio": 0.75,     # 52주 고가의 75% 이상
    "above_low_ratio": 1.30,     # 52주 저가 대비 +30% 이상
    "pullback_band": 0.03,       # 눌림 인정 범위: 피벗·20일선 ±3%
    "breakout_recent_days": 10,  # 눌림 대상: 최근 10거래일 내 돌파 이력
    "watch_band": 0.05,          # 돌파 임박: 피벗 아래 5% 이내
    "stop_pct": 0.07,            # 최대 손절 -7%
    "target_pct": 0.10,          # 1차 목표 +10%
    "hold_days": 5,              # 시간 손절: 5거래일
    "request_pause": 0.15,       # KRX 요청 간격(초)
}

UNIVERSE = {"KOSPI200": "1028", "KOSDAQ150": "2203"}
MARKET_INDEX = {"코스피": ("1001", "KS11"), "코스닥": ("2001", "KQ11")}
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
        "거래량": "volume", "거래대금": "value", "등락률": "chg"}


# ─────────────────────────── 데이터 수집 ───────────────────────────
def _pykrx():
    try:
        from pykrx import stock
        return stock
    except ImportError:
        sys.exit("pykrx가 없습니다: pip install pykrx")


def nearest_business_day(date: str) -> str:
    stock = _pykrx()
    try:
        return stock.get_nearest_business_day_in_a_week(date=date, prev=True)
    except Exception:
        return date


def get_universe(date: str) -> pd.DataFrame:
    stock = _pykrx()
    rows = []
    for uni, code in UNIVERSE.items():
        tickers = []
        for call in (lambda: stock.get_index_portfolio_deposit_file(ticker=code, date=date),
                     lambda: stock.get_index_portfolio_deposit_file(code, date),
                     lambda: stock.get_index_portfolio_deposit_file(code)):
            try:
                tickers = list(call())
                if tickers:
                    break
            except Exception:
                continue
        if not tickers:
            print(f"[경고] {uni} 구성종목을 받지 못했습니다.")
        for t in tickers:
            try:
                name = stock.get_market_ticker_name(t)
            except Exception:
                name = t
            rows.append({"ticker": t, "name": name, "universe": uni})
    if not rows:
        sys.exit("구성종목 조회 실패. pykrx 업데이트(pip install -U pykrx) 후 다시 실행하세요.")
    return pd.DataFrame(rows).drop_duplicates("ticker")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLS)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    if "value" not in df.columns:
        df["value"] = df["close"] * df["volume"]
    df = df[["open", "high", "low", "close", "volume", "value"]].astype(float)
    return df[df["close"] > 0]


def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    stock = _pykrx()
    try:
        df = stock.get_market_ohlcv(start, end, ticker)
        if df is not None and len(df):
            return _normalize(df)
    except Exception:
        pass
    try:  # 대체 경로
        import FinanceDataReader as fdr
        df = fdr.DataReader(ticker, start, end)
        if len(df):
            return _normalize(df)
    except Exception:
        pass
    return None


def get_index(name: str, start: str, end: str) -> pd.DataFrame | None:
    krx_code, fdr_code = MARKET_INDEX[name]
    stock = _pykrx()
    try:
        df = stock.get_index_ohlcv(start, end, krx_code)
        if df is not None and len(df):
            return _normalize(df)
    except Exception:
        pass
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(fdr_code, start, end)
        if len(df):
            return _normalize(df)
    except Exception:
        pass
    return None


def get_flow(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """외국인·기관 순매수 대금(원)."""
    stock = _pykrx()
    try:
        df = stock.get_market_trading_value_by_date(start, end, ticker)
        out = pd.DataFrame(index=df.index)
        out["foreign"] = df.get("외국인합계", 0)
        out["inst"] = df.get("기관합계", 0)
        return out.astype(float)
    except Exception:
        return None


# ─────────────────────────── 분석 로직 ───────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 50, 60, 120, 150, 200):
        df[f"ma{n}"] = df["close"].rolling(n).mean()
    df["vol50"] = df["volume"].rolling(50).mean()
    df["value20"] = df["value"].rolling(20).mean()
    df["high52"] = df["high"].rolling(252, min_periods=120).max()
    df["low52"] = df["low"].rolling(252, min_periods=120).min()
    df["pivot"] = df["high"].shift(1).rolling(CFG["pivot_lookback"]).max()
    df["vol50_prev"] = df["volume"].shift(1).rolling(50).mean()
    df["is_breakout"] = (df["close"] > df["pivot"]) & (
        df["volume"] >= CFG["vol_mult"] * df["vol50_prev"])
    return df


def rs_raw(close: pd.Series) -> float:
    """IBD식 가중 수익률: 최근 3개월 2배 + 6·9·12개월."""
    c = close.values
    def ret(n):
        return c[-1] / c[-n - 1] - 1 if len(c) > n else np.nan
    parts = [ret(63), ret(63), ret(126), ret(189), ret(252)]
    parts = [p for p in parts if not np.isnan(p)]
    return float(np.mean(parts)) if len(parts) >= 3 else np.nan


def trend_checks(r: pd.Series, ma200_20ago: float) -> dict:
    return {
        "종가>50일선": r.close > r.ma50,
        "50>150일선": r.ma50 > r.ma150,
        "150>200일선": r.ma150 > r.ma200,
        "200일선 상승": r.ma200 > ma200_20ago,
        "52주고가 75%↑": r.close >= CFG["near_high_ratio"] * r.high52,
        "52주저가 +30%↑": r.close >= CFG["above_low_ratio"] * r.low52,
    }


def analyze(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 210:
        return None
    df = add_indicators(df)
    r = df.iloc[-1]
    if pd.isna(r.ma200) or pd.isna(r.pivot) or r.value20 < CFG["min_value_20d"]:
        return None

    checks = trend_checks(r, df["ma200"].iloc[-21])
    trend_n = int(sum(bool(v) for v in checks.values()))
    rng = r.high - r.low
    close_pos = (r.close - r.low) / rng if rng > 0 else 1.0
    vol_ratio = r.volume / r.vol50_prev if r.vol50_prev else np.nan
    ext = r.close / r.pivot - 1

    setup, note, ref = None, "", r.pivot

    # A. 돌파: 피벗 종가 돌파 + 거래량 + 과열 아님 + 고가권 마감
    if (r.is_breakout and ext <= CFG["max_extension"] and close_pos >= 0.5
            and r.close > r.ma50):
        setup, note = "돌파", f"피벗 {r.pivot:,.0f} 돌파, 거래량 {vol_ratio:.1f}배"
    elif r.is_breakout and ext > CFG["max_extension"]:
        note = f"피벗 대비 +{ext*100:.1f}% 과열, 추격 금지"

    # B. 눌림: 최근 돌파 후 피벗·20일선 부근으로 거래량 줄며 조정
    if setup is None:
        recent = df.iloc[-1 - CFG["breakout_recent_days"]:-1]
        hits = recent[recent["is_breakout"]]
        if len(hits):
            bp = hits["pivot"].iloc[-1]
            near_pivot = abs(r.close / bp - 1) <= CFG["pullback_band"]
            near_ma20 = abs(r.close / r.ma20 - 1) <= CFG["pullback_band"]
            if ((near_pivot or near_ma20) and r.close >= bp * 0.97
                    and r.volume < r.vol50_prev and r.close > r.ma50):
                setup, ref = "눌림", bp
                where = "피벗" if near_pivot else "20일선"
                note = f"{hits.index[-1]:%m/%d} 돌파 후 {where} 눌림, 거래량 {vol_ratio:.1f}배"

    # C. 관찰: 추세 양호 + 피벗 아래 5% 이내 (돌파 임박)
    if setup is None and trend_n >= 5 and -CFG["watch_band"] <= ext <= 0:
        setup = "관찰"
        note = f"피벗 {r.pivot:,.0f}까지 {abs(ext)*100:.1f}%"

    entry = r.close
    if setup == "돌파":
        stop = max(entry * (1 - CFG["stop_pct"]), r.pivot * 0.97)
    elif setup == "눌림":
        stop = max(entry * (1 - CFG["stop_pct"]), min(ref, r.ma20) * 0.97)
    elif setup == "관찰":
        entry = r.pivot * 1.001            # 돌파 시 진입가
        stop = max(entry * (1 - CFG["stop_pct"]), r.pivot * 0.97)
    else:
        stop = np.nan
    target = entry * (1 + CFG["target_pct"])
    rr = (target - entry) / (entry - stop) if setup and entry > stop else np.nan

    return {
        "close": r.close, "chg": (r.close / df["close"].iloc[-2] - 1) * 100,
        "pivot": r.pivot, "ext": ext * 100, "vol_ratio": vol_ratio,
        "close_pos": close_pos, "value20_eok": r.value20 / 1e8,
        "from_high": (r.close / r.high52 - 1) * 100,
        "trend_n": trend_n,
        "trend_fail": ", ".join(k for k, v in checks.items() if not v),
        "rs_raw": rs_raw(df["close"]),
        "setup": setup, "note": note,
        "entry": entry, "stop": stop, "target": target, "rr": rr,
        "stop_pct": (stop / entry - 1) * 100 if setup else np.nan,
    }


def market_regime(idx: pd.DataFrame | None) -> dict:
    """오닐 M: 지수 위치 + 최근 25거래일 분배일 수."""
    if idx is None or len(idx) < 200:
        return {"label": "판정 불가", "level": "unknown", "detail": "지수 데이터 부족"}
    c = idx["close"]
    ma50, ma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    last = c.iloc[-1]
    chg = c.pct_change()
    dist = ((chg <= -0.002) & (idx["volume"] > idx["volume"].shift(1))).iloc[-25:].sum()
    high = idx["high"].iloc[-252:].max()
    dd = (last / high - 1) * 100
    if last > ma50 > ma200 and dist <= 4:
        label, level, sizing = "상승 추세", "up", "정상 비중"
    elif last > ma200 or last > ma50:
        label, level, sizing = "상승 압박", "mid", "비중 절반"
    else:
        label, level, sizing = "조정", "down", "신규 매수 최소화"
    return {
        "label": label, "level": level, "sizing": sizing,
        "detail": (f"종가 {last:,.2f} / 50일선 {ma50:,.2f} / 200일선 {ma200:,.2f} / "
                   f"고점 대비 {dd:.1f}% / 분배일 {int(dist)}회(25일)"),
    }


def flow_score(flow: pd.DataFrame | None) -> tuple[float, str]:
    if flow is None or len(flow) < 5:
        return 50.0, "-"
    net = flow["foreign"] + flow["inst"]
    n5, n1 = net.iloc[-5:].sum(), net.iloc[-1]
    txt = f"5일 {n5/1e8:+,.0f}억 (외 {flow['foreign'].iloc[-5:].sum()/1e8:+,.0f} / 기 {flow['inst'].iloc[-5:].sum()/1e8:+,.0f})"
    if n5 > 0 and n1 > 0:
        return 100.0, txt
    if n5 > 0:
        return 65.0, txt
    if n1 > 0:
        return 35.0, txt
    return 0.0, txt


def composite(row: pd.Series) -> float:
    vol_part = min((row.vol_ratio or 0) / 3, 1) * 100
    if row.setup == "눌림":  # 눌림은 거래량 감소가 좋은 신호
        vol_part = max(0, min((1 - (row.vol_ratio or 1)) * 200, 100))
    return round(0.35 * row.rs_pct + 0.30 * row.trend_n / 6 * 100
                 + 0.15 * vol_part + 0.20 * row.flow_score, 1)


# ─────────────────────────── 리포트 ───────────────────────────
CSS = """
:root{--paper:#F2F4F6;--sheet:#FFFFFF;--ink:#17202A;--muted:#5A6572;--rule:#D6DCE3;
--up:#C62E2E;--down:#2A58A6;--amber:#9A620E;--wash-up:#FBEAEA;--wash-mid:#FBF1DF;--wash-down:#E7EEF8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12161B;--sheet:#1A2027;
--ink:#E6EAEE;--muted:#98A3AF;--rule:#2E3741;--up:#F06A6A;--down:#6E9BE8;--amber:#E0A344;
--wash-up:#3A1F22;--wash-mid:#3A2F1C;--wash-down:#1D2A3F}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:Pretendard,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;
font-size:15px;line-height:1.55;font-variant-numeric:tabular-nums}
main{max-width:1120px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:22px;margin:0 0 4px;font-weight:700}
.sub{color:var(--muted);margin:0 0 22px}
.regime{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:28px}
.mkt{background:var(--sheet);border-left:6px solid var(--muted);padding:14px 18px}
.mkt.up{border-color:var(--up)}.mkt.mid{border-color:var(--amber)}.mkt.down{border-color:var(--down)}
.mkt .name{color:var(--muted);font-size:14px}
.mkt .state{font-size:40px;font-weight:800;letter-spacing:-.02em;line-height:1.15}
.mkt.up .state{color:var(--up)}.mkt.mid .state{color:var(--amber)}.mkt.down .state{color:var(--down)}
.mkt .size{font-weight:600}.mkt .det{color:var(--muted);font-size:13px;margin-top:4px}
section{margin-top:30px}
h2{font-size:18px;margin:0 0 4px}
.desc{color:var(--muted);margin:0 0 10px;font-size:14px}
.wrap{overflow-x:auto;background:var(--sheet);border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;min-width:900px}
th,td{padding:8px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{font-size:13px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--sheet)}
td.l,th.l{text-align:left}
td.name b{display:block}td.name span{color:var(--muted);font-size:12px}
.pos{color:var(--up)}.neg{color:var(--down)}
.note{white-space:normal;min-width:150px;text-align:left;font-size:13px}
.why{min-width:230px}
.score{font-weight:700}
.empty{padding:18px;color:var(--muted);background:var(--sheet);border:1px dashed var(--rule)}
.rules{margin-top:34px;color:var(--muted);font-size:13px;max-width:78ch}
"""


def _fmt(v, f="{:,.0f}"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _table(df: pd.DataFrame, kind: str) -> str:
    if df.empty:
        msg = {"돌파": "오늘 조건을 모두 충족한 돌파 종목이 없습니다. 무리하게 진입하지 않는 것도 전략입니다.",
               "눌림": "최근 돌파 후 건강하게 눌린 종목이 없습니다.",
               "관찰": "돌파 임박 종목이 없습니다."}[kind]
        return f'<p class="empty">{msg}</p>'
    entry_h = "돌파 시 진입" if kind == "관찰" else "진입(종가)"
    head = (f'<tr><th class="l">종목</th><th>점수</th><th>종가</th><th>등락</th><th>RS</th>'
            f'<th>추세</th><th>고점대비</th><th>{entry_h}</th><th>손절</th><th>목표</th>'
            f'<th>손익비</th><th class="l">수급</th><th class="l">근거</th></tr>')
    body = []
    for _, r in df.iterrows():
        cls = "pos" if r.chg > 0 else "neg" if r.chg < 0 else ""
        body.append(
            "<tr>"
            f'<td class="l name"><b>{html.escape(str(r["name"]))}</b><span>{r.ticker} {r.universe}</span></td>'
            f'<td class="score">{r.score:.0f}</td>'
            f"<td>{_fmt(r.close)}</td>"
            f'<td class="{cls}">{r.chg:+.2f}%</td>'
            f"<td>{r.rs_pct:.0f}</td>"
            f'<td title="{html.escape(r.trend_fail or "모두 충족")}">{r.trend_n}/6</td>'
            f"<td>{round(r.from_high) + 0:.0f}%</td>"
            f"<td>{_fmt(r.entry)}</td>"
            f'<td class="neg">{_fmt(r.stop)}<br><small>{r.stop_pct:.1f}%</small></td>'
            f'<td class="pos">{_fmt(r.target)}</td>'
            f'<td>{_fmt(r.rr, "{:.1f}")}</td>'
            f'<td class="note">{html.escape(r.flow_txt)}</td>'
            f'<td class="note why">{html.escape(r.note)}</td>'
            "</tr>")
    return f'<div class="wrap"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def build_html(date: str, regimes: dict, res: pd.DataFrame, n_total: int, n_ok: int) -> str:
    d = dt.datetime.strptime(date, "%Y%m%d")
    mk = "".join(
        f'<div class="mkt {g["level"]}"><div class="name">{n}</div>'
        f'<div class="state">{g["label"]}</div>'
        f'<div class="size">{g.get("sizing","")}</div><div class="det">{html.escape(g["detail"])}</div></div>'
        for n, g in regimes.items())
    secs = [
        ("돌파", "오늘 20일 고점을 거래량과 함께 종가로 넘은 종목. 피벗 +5% 이내만 표시."),
        ("눌림", "최근 10일 내 돌파한 뒤, 거래량이 줄며 피벗이나 20일선까지 되돌린 종목. 손익비가 가장 좋은 자리."),
        ("관찰", "추세 조건 5개 이상 충족, 피벗 아래 5% 이내. 내일 돌파하면 진입 후보."),
    ]
    body = "".join(
        f"<section><h2>{k} 후보</h2><p class=\"desc\">{t}</p>"
        f"{_table(res[res.setup == k].sort_values('score', ascending=False).head(15), k)}</section>"
        for k, t in secs)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>스윙 후보 {d:%Y-%m-%d}</title><style>{CSS}</style></head><body><main>
<h1>스윙 후보 {d:%Y년 %m월 %d일}</h1>
<p class="sub">코스피200과 코스닥150 {n_total}종목 중 거래대금 조건을 넘은 {n_ok}종목을 분석했습니다. 보유 기간은 최대 {CFG['hold_days']}거래일 기준입니다.</p>
<div class="regime">{mk}</div>
{body}
<div class="rules"><p>점수는 상대강도 35%, 추세 조건 30%, 거래량 15%, 외국인·기관 수급 20%로 계산합니다.
RS는 분석 대상 안에서의 백분위(99가 최상)입니다. 추세 칸에 마우스를 올리면 미충족 조건이 보입니다.</p>
<p>손절은 진입가 −7%와 피벗(또는 20일선) −3% 중 가까운 쪽이고, 목표는 +10%입니다.
{CFG['hold_days']}거래일 안에 목표에 닿지 않으면 정리하는 것을 원칙으로 합니다.
시장 판정이 조정이면 후보가 있어도 비중을 크게 줄이세요.</p>
<p>이 리포트는 공개 시세로 계산한 기계적 선별 결과이며, 투자 권유가 아닙니다. 실적, 공시, 뉴스는 따로 확인하세요.</p></div>
</main></body></html>"""


# ─────────────────────────── 실행 ───────────────────────────
def run(date: str, out_dir: Path, use_flow: bool = True) -> Path:
    now = dt.datetime.now()
    if date == now.strftime("%Y%m%d"):
        if now.hour < 16:
            print("[주의] 16시 이전입니다. 당일 시세가 확정되지 않았을 수 있습니다.")
        elif now.hour < 18 and use_flow:
            print("[주의] 18시 이전입니다. 외국인·기관 수급은 잠정치로 반영됩니다.")
    date = nearest_business_day(date)
    start = (dt.datetime.strptime(date, "%Y%m%d") - dt.timedelta(days=420)).strftime("%Y%m%d")
    print(f"기준일 {date} / 조회 시작 {start}")

    regimes = {n: market_regime(get_index(n, start, date)) for n in MARKET_INDEX}
    for n, g in regimes.items():
        print(f"  {n}: {g['label']}  ({g['detail']})")

    uni = get_universe(date)
    print(f"유니버스 {len(uni)}종목 분석 중...")
    rows = []
    for i, u in enumerate(uni.itertuples(), 1):
        a = analyze(get_ohlcv(u.ticker, start, date))
        if a:
            rows.append({"ticker": u.ticker, "name": u.name, "universe": u.universe, **a})
        if i % 50 == 0:
            print(f"  {i}/{len(uni)}")
        time.sleep(CFG["request_pause"])

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("분석 가능한 종목이 없습니다. 데이터 조회 상태를 확인하세요.")
    res["rs_pct"] = res["rs_raw"].rank(pct=True).fillna(0.5) * 99

    res["flow_score"], res["flow_txt"] = 50.0, "-"
    if use_flow:
        fstart = (dt.datetime.strptime(date, "%Y%m%d") - dt.timedelta(days=14)).strftime("%Y%m%d")
        for idx in res.index[res["setup"].notna()]:
            s, t = flow_score(get_flow(res.at[idx, "ticker"], fstart, date))
            res.at[idx, "flow_score"], res.at[idx, "flow_txt"] = s, t
            time.sleep(CFG["request_pause"])

    res["score"] = res.apply(composite, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"swing_{date}.csv"
    html_path = out_dir / f"swing_{date}.html"
    res.sort_values("score", ascending=False).drop(columns=["rs_raw"]).to_csv(
        csv_path, index=False, encoding="utf-8-sig", float_format="%.2f")
    html_path.write_text(build_html(date, regimes, res, len(uni), len(res)), encoding="utf-8")

    for k in ("돌파", "눌림", "관찰"):
        sub = res[res.setup == k].sort_values("score", ascending=False).head(5)
        names = ", ".join(f"{r['name']}({r.score:.0f})" for _, r in sub.iterrows()) or "없음"
        print(f"{k}: {names}")
    print(f"리포트: {html_path}\nCSV: {csv_path}")
    return html_path


def main():
    p = argparse.ArgumentParser(description="KOSPI200·KOSDAQ150 스윙 스크리너")
    p.add_argument("--date", default=dt.datetime.now().strftime("%Y%m%d"))
    p.add_argument("--out", default="output")
    p.add_argument("--no-flow", action="store_true", help="수급 조회 생략")
    a = p.parse_args()
    run(a.date, Path(a.out), use_flow=not a.no_flow)


if __name__ == "__main__":
    main()
