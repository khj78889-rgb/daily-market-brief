#!/usr/bin/env python3
"""
거래량 급증 스캐너 — 코스피·코스닥 전 종목

기준: 기준일 거래량 ÷ 직전 거래일 거래량 ≥ 배수(기본 3배)

사용법
  pip install pykrx pandas
  python volume_surge.py                    # 직전 거래일 기준
  python volume_surge.py --date 20260929    # 특정일 기준
  python volume_surge.py --ratio 5          # 5배 이상만
  python volume_surge.py --min-value 10     # 거래대금 10억원 이상만
  python volume_surge.py --universe index   # 코스피200·코스닥150만
  python volume_surge.py --lookback 10      # 최근 10거래일 급증분까지 추적(급증 후 눌림 표)

결과
  output/volume_surge_YYYYMMDD.csv / .html
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import sys
from pathlib import Path

import pandas as pd

CFG = {
    "lookback": 10,          # 급증 후 눌림: 며칠 전 급증까지 추적할지(거래일)
    "pullback_vol": 0.5,     # 급등일 거래량 대비 현재 거래량 비율 상한
    "pullback_gain": 5.0,    # 급등일 최소 상승률(%)
    "pullback_floor": -3.0,  # 급등일 종가 대비 현재가 허용 하락폭(%)
    "ratio": 3.0,        # 전일 대비 거래량 배수 하한
    "min_value_eok": 5,  # 기준일 거래대금 하한(억원). 잡음 제거용
    "min_price": 1000,   # 동전주 제외
}
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
        "거래량": "volume", "거래대금": "value", "등락률": "chg"}


def _pykrx():
    try:
        from pykrx import stock
        return stock
    except ImportError:
        sys.exit("pykrx가 없습니다: pip install pykrx")


def snapshot(date: str) -> pd.DataFrame:
    """해당일 전 종목 시세(코스피+코스닥)."""
    stock = _pykrx()
    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_market_ohlcv_by_ticker(date, market=market)
        except Exception as e:
            print(f"[경고] {market} {date} 조회 실패: {e}")
            continue
        if df is None or df.empty:
            continue
        df = df.rename(columns=COLS)
        df["market"] = market
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    if "value" not in out.columns:
        out["value"] = out["close"] * out["volume"]
    return out


def index_members(date: str) -> set[str]:
    """코스피200·코스닥150 구성종목."""
    stock = _pykrx()
    out: set[str] = set()
    for code in ("1028", "2203"):
        for call in (lambda: stock.get_index_portfolio_deposit_file(ticker=code, date=date),
                     lambda: stock.get_index_portfolio_deposit_file(code, date),
                     lambda: stock.get_index_portfolio_deposit_file(code)):
            try:
                got = list(call())
                if got:
                    out.update(got)
                    break
            except Exception:
                continue
    if not out:
        print("[경고] 지수 구성종목을 받지 못해 전 종목으로 진행합니다.")
    return out


def prev_business_day(date: str) -> str:
    stock = _pykrx()
    d = (dt.datetime.strptime(date, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")
    try:
        return stock.get_nearest_business_day_in_a_week(date=d, prev=True)
    except Exception:
        return d


def scan(date: str, ratio: float, min_value_eok: float, universe: str = "all") -> tuple[pd.DataFrame, str]:
    stock = _pykrx()
    try:
        date = stock.get_nearest_business_day_in_a_week(date=date, prev=True)
    except Exception:
        pass
    prev = prev_business_day(date)
    print(f"기준일 {date} / 비교일 {prev}")

    cur, old = snapshot(date), snapshot(prev)
    if cur.empty or old.empty:
        sys.exit("시세 조회 실패. KRX_ID·KRX_PW 설정과 pykrx 버전을 확인하세요.")

    if universe == "index":
        members = index_members(date)
        if members:
            cur = cur[cur.index.isin(members)]

    df = cur.join(old[["volume"]].rename(columns={"volume": "prev_volume"}), how="inner")
    df = df[(df["prev_volume"] > 0) & (df["close"] >= CFG["min_price"])
            & (df["value"] >= min_value_eok * 1e8)]
    df["ratio"] = df["volume"] / df["prev_volume"]
    df = df[df["ratio"] >= ratio].copy()

    names = []
    for t in df.index:
        try:
            names.append(stock.get_market_ticker_name(t))
        except Exception:
            names.append(t)
    df.insert(0, "name", names)
    df["value_eok"] = df["value"] / 1e8
    df = df.sort_values("ratio", ascending=False)
    return df, prev


def business_days(date: str, n: int) -> list[str]:
    """기준일 포함, 과거로 n개 거래일."""
    stock = _pykrx()
    out, d = [], dt.datetime.strptime(date, "%Y%m%d")
    while len(out) < n:
        try:
            b = stock.get_nearest_business_day_in_a_week(date=d.strftime("%Y%m%d"), prev=True)
        except Exception:
            b = d.strftime("%Y%m%d")
        if b not in out:
            out.append(b)
        d = dt.datetime.strptime(b, "%Y%m%d") - dt.timedelta(days=1)
    return list(reversed(out))


def pullback_scan(date: str, ratio: float, min_value_eok: float,
                  universe: str, lookback: int) -> pd.DataFrame:
    """최근 급증 뒤, 거래량이 마르며 가격은 버티는 종목."""
    days = business_days(date, lookback + 1)
    snaps = {}
    for d in days:
        df = snapshot(d)
        if not df.empty:
            snaps[d] = df
        print(f"  {d} 시세 {len(df)}종목")
    days = [d for d in days if d in snaps]
    if len(days) < 3:
        return pd.DataFrame()

    last = days[-1]
    cur = snaps[last]
    if universe == "index":
        members = index_members(last)
        if members:
            cur = cur[cur.index.isin(members)]

    rows = []
    for t in cur.index:
        c = cur.loc[t]
        if c["value"] < min_value_eok * 1e8 or c["close"] < CFG["min_price"]:
            continue
        best = None
        # 급등일 후보: 마지막 날과 직전 날은 제외(눌림 기간 2일 이상 확보)
        for i in range(1, len(days) - 2):
            d, pd_ = days[i], days[i - 1]
            if t not in snaps[d].index or t not in snaps[pd_].index:
                continue
            row, prev_row = snaps[d].loc[t], snaps[pd_].loc[t]
            if prev_row["volume"] <= 0:
                continue
            surge = row["volume"] / prev_row["volume"]
            if surge < ratio or row.get("chg", 0) < CFG["pullback_gain"]:
                continue
            if best is None or surge > best[1]:
                best = (d, surge, float(row["close"]), float(row["volume"]))
        if best is None:
            continue
        sd, surge, sclose, svol = best
        hold = (c["close"] / sclose - 1) * 100
        vol_cut = c["volume"] / svol
        if hold < CFG["pullback_floor"] or vol_cut > CFG["pullback_vol"]:
            continue
        rows.append({"ticker": t, "surge_date": sd, "surge_ratio": surge,
                     "surge_close": sclose, "close": float(c["close"]),
                     "hold": hold, "vol_cut": vol_cut, "chg": float(c.get("chg", 0)),
                     "value_eok": float(c["value"]) / 1e8, "market": c["market"]})
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("ticker")
    stock = _pykrx()
    names = []
    for t in out.index:
        try:
            names.append(stock.get_market_ticker_name(t))
        except Exception:
            names.append(t)
    out.insert(0, "name", names)
    return out.sort_values("vol_cut")


CSS = """
:root{--paper:#F2F4F6;--sheet:#FFF;--ink:#17202A;--muted:#5A6572;--rule:#D6DCE3;--up:#C62E2E;--down:#2A58A6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12161B;--sheet:#1A2027;--ink:#E6EAEE;
--muted:#98A3AF;--rule:#2E3741;--up:#F06A6A;--down:#6E9BE8}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:Pretendard,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;
font-size:15px;font-variant-numeric:tabular-nums}
main{max-width:900px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:21px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px;font-size:14px}
.wrap{overflow-x:auto;background:var(--sheet);border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%}th,td{padding:8px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{font-size:13px;color:var(--muted);position:sticky;top:0;background:var(--sheet)}
td.l,th.l{text-align:left}.pos{color:var(--up)}.neg{color:var(--down)}
.big{font-weight:700}
h2{font-size:18px;margin:28px 0 4px}
.cards{display:none}
.card{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:12px 14px;margin-bottom:8px}
.card .top{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.card .nm{font-weight:700;font-size:16px}
.card .tk{color:var(--muted);font-size:12px;margin-top:2px}
.card .mult{font-size:22px;font-weight:800;white-space:nowrap}
.card .px{margin-top:8px;font-size:15px}
.card .px b{font-size:17px}
.card dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 6px;margin:10px 0 0}
.card dl div{min-width:0}
.card dt{color:var(--muted);font-size:11px;margin-bottom:2px}
.card dd{margin:0;font-size:14px;font-weight:600;white-space:nowrap}
.sum{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:12px 14px;margin:0 0 16px;font-size:14px}
.sum b{font-size:18px}
@media (max-width:720px){
 main{padding:18px 12px 44px}
 h1{font-size:19px}
 .wrap{display:none}
 .cards{display:block}
}
"""


def _surge_cards(df: pd.DataFrame) -> str:
    cards = "".join(
        f'<article class="card"><div class="top"><div>'
        f'<div class="nm">{html.escape(str(r["name"]))}</div><div class="tk">{i} {r["market"]}</div></div>'
        f'<div class="mult">{r["ratio"]:.1f}배</div></div>'
        f'<div class="px"><b>{r["close"]:,.0f}원</b> '
        f'<span class="{"pos" if r["chg"] > 0 else "neg" if r["chg"] < 0 else ""}">{r["chg"]:+.2f}%</span></div>'
        f'<dl><div><dt>거래량</dt><dd>{r["volume"]/10000:,.0f}만주</dd></div>'
        f'<div><dt>전일</dt><dd>{r["prev_volume"]/10000:,.1f}만주</dd></div>'
        f'<div><dt>거래대금</dt><dd>{r["value_eok"]:,.0f}억</dd></div></dl></article>'
        for i, r in df.iterrows())
    return f'<div class="cards">{cards}</div>'


def _pullback_cards(df: pd.DataFrame) -> str:
    cards = "".join(
        f'<article class="card"><div class="top"><div>'
        f'<div class="nm">{html.escape(str(r["name"]))}</div><div class="tk">{i} {r["market"]}</div></div>'
        f'<div style="text-align:right"><div class="mult">{r["vol_cut"]*100:.0f}%</div>'
        f'<div class="tk">급등일 대비 거래량</div></div></div>'
        f'<div class="px"><b>{r["close"]:,.0f}원</b> '
        f'<span class="{"pos" if r["hold"] > 0 else "neg"}">급등일 대비 {r["hold"]:+.1f}%</span></div>'
        f'<dl><div><dt>급등일</dt><dd>{r["surge_date"][4:6]}/{r["surge_date"][6:]}</dd></div>'
        f'<div><dt>당시 배수</dt><dd>{r["surge_ratio"]:.1f}배</dd></div>'
        f'<div><dt>급등일 종가</dt><dd>{r["surge_close"]:,.0f}</dd></div></dl></article>'
        for i, r in df.iterrows())
    return f'<div class="cards">{cards}</div>'


def build_html(df: pd.DataFrame, date: str, prev: str, ratio: float, universe: str = "all",
               pull: pd.DataFrame | None = None, lookback: int = 0) -> str:
    d = dt.datetime.strptime(date, "%Y%m%d")
    rows = "".join(
        f'<tr><td class="l">{html.escape(str(r["name"]))}<br>'
        f'<small style="color:var(--muted)">{i} {r["market"]}</small></td>'
        f'<td class="big">{r["ratio"]:.1f}배</td>'
        f'<td>{r["close"]:,.0f}</td>'
        f'<td class="{"pos" if r["chg"] > 0 else "neg" if r["chg"] < 0 else ""}">{r["chg"]:+.2f}%</td>'
        f'<td>{r["volume"]:,.0f}</td><td>{r["prev_volume"]:,.0f}</td>'
        f'<td>{r["value_eok"]:,.0f}억</td></tr>'
        for i, r in df.iterrows())
    pull_block = pullback_html(pull, lookback) if lookback else ""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>거래량 급증 {d:%Y-%m-%d}</title><style>{CSS}</style></head><body><main>
<h1>거래량 급증 {d:%Y년 %m월 %d일}</h1>
<p class="sub">생성 {dt.datetime.now():%Y-%m-%d %H:%M} · {"장중 데이터(미확정)" if dt.datetime.now().strftime("%Y%m%d") == date and dt.datetime.now().hour < 16 else "마감 확정"}</p>
<p class="sub">직전 거래일({prev[:4]}-{prev[4:6]}-{prev[6:]}) 대비 거래량 {ratio:.0f}배 이상, 거래대금 {CFG['min_value_eok']}억원 이상인 종목 {len(df)}개입니다.
{"코스피200·코스닥150 구성종목" if universe == "index" else "코스피·코스닥 보통주"} 기준이며 ETF는 제외됩니다.</p>
<div class="sum"><b>{len(df)}종목</b> 급증 · 상승 {int((df["chg"] > 0).sum())} / 하락 {int((df["chg"] < 0).sum())} ·
10배 이상 {int((df["ratio"] >= 10).sum())}종목</div>
<h2>급증 종목</h2>
<div class="wrap"><table><thead><tr><th class="l">종목</th><th>배수</th><th>종가</th><th>등락</th>
<th>거래량</th><th>전일 거래량</th><th>거래대금</th></tr></thead><tbody>{rows}</tbody></table></div>
{_surge_cards(df)}
{pull_block}
</main></body></html>"""


def pullback_html(df: pd.DataFrame, lookback: int) -> str:
    head = (f'<h2>급증 후 눌림</h2><p class="sub">최근 {lookback}거래일 안에 거래량 급증과 함께 '
            f'{CFG["pullback_gain"]:.0f}% 이상 오른 뒤, 지금은 거래량이 급등일의 '
            f'{CFG["pullback_vol"]*100:.0f}% 이하로 줄고 가격은 급등일 종가 부근을 지키는 종목입니다.</p>')
    if df is None or df.empty:
        return head + '<p class="sub">해당 종목이 없습니다.</p>'
    rows = "".join(
        f'<tr><td class="l">{html.escape(str(r["name"]))}<br>'
        f'<small style="color:var(--muted)">{i} {r["market"]}</small></td>'
        f'<td>{r["surge_date"][4:6]}/{r["surge_date"][6:]}</td>'
        f'<td>{r["surge_ratio"]:.1f}배</td>'
        f'<td>{r["surge_close"]:,.0f}</td>'
        f'<td class="big">{r["close"]:,.0f}</td>'
        f'<td class="{"pos" if r["hold"] > 0 else "neg"}">{r["hold"]:+.1f}%</td>'
        f'<td>{r["vol_cut"]*100:.0f}%</td><td>{r["value_eok"]:,.0f}억</td></tr>'
        for i, r in df.iterrows())
    return (head + '<div class="wrap"><table><thead><tr><th class="l">종목</th><th>급등일</th>'
            '<th>당시 배수</th><th>급등일 종가</th><th>현재가</th><th>유지율</th>'
            f'<th>거래량 비율</th><th>거래대금</th></tr></thead><tbody>{rows}</tbody></table></div>'
            + _pullback_cards(df))


def main():
    p = argparse.ArgumentParser(description="거래량 급증 스캐너")
    p.add_argument("--date", default=dt.datetime.now().strftime("%Y%m%d"))
    p.add_argument("--ratio", type=float, default=CFG["ratio"])
    p.add_argument("--min-value", type=float, default=CFG["min_value_eok"], help="거래대금 하한(억원)")
    p.add_argument("--universe", choices=["all", "index"], default="all",
                   help="all: 코스피·코스닥 전 종목, index: 코스피200·코스닥150")
    p.add_argument("--lookback", type=int, default=CFG["lookback"],
                   help="급증 후 눌림 추적 기간(거래일). 0이면 생략")
    p.add_argument("--out", default="output")
    a = p.parse_args()

    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        print("[주의] KRX_ID·KRX_PW가 없습니다. 조회가 실패할 수 있습니다.")

    df, prev = scan(a.date, a.ratio, a.min_value, a.universe)
    pull = pd.DataFrame()
    if a.lookback:
        print(f"급증 후 눌림 추적 (최근 {a.lookback}거래일)")
        pull = pullback_scan(a.date, a.ratio, a.min_value, a.universe, a.lookback)
    date = df.attrs.get("date", a.date)
    try:
        date = _pykrx().get_nearest_business_day_in_a_week(date=a.date, prev=True)
    except Exception:
        pass

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_index" if a.universe == "index" else ""
    csv_path = out / f"volume_surge{tag}_{date}.csv"
    html_path = out / f"volume_surge{tag}_{date}.html"
    df.reset_index().rename(columns={"index": "ticker", "티커": "ticker"}).to_csv(
        csv_path, index=False, encoding="utf-8-sig", float_format="%.2f")
    page = build_html(df, date, prev, a.ratio, a.universe, pull, a.lookback)
    html_path.write_text(page, encoding="utf-8")
    (out / "index.html").write_text(page, encoding="utf-8")   # 항상 최신 결과로 덮어씀

    if a.lookback and not pull.empty:
        pull.reset_index().to_csv(out / f"volume_pullback{tag}_{date}.csv",
                                  index=False, encoding="utf-8-sig", float_format="%.2f")
    print(f"{a.ratio:.0f}배 이상: {len(df)}종목 / 급증 후 눌림: {len(pull)}종목")
    for i, r in df.head(15).iterrows():
        print(f"  {r['name']}({i}) {r['ratio']:.1f}배 {r['chg']:+.2f}% 거래대금 {r['value_eok']:,.0f}억")
    print(f"리포트: {html_path}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
