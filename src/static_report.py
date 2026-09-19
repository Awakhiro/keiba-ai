"""
JavaScript を使わない静的な予想ページを作る。

Google Drive アプリのプレビューやメールの添付では JavaScript が動かないため、
対話式のビューアでは中身が表示されない。
このモジュールは全レース・全買い目を最初から HTML に書き出すので、
どんな環境で開いても読める。

絞り込みや切り替えはできないかわりに、確実に表示される。
"""

import html as _html

FRAME_COLORS = {1: ("#FFFFFF", "#1A1A1A"), 2: ("#1A1A1A", "#FFFFFF"),
                3: ("#D0342C", "#FFFFFF"), 4: ("#2565C7", "#FFFFFF"),
                5: ("#EFC126", "#1A1A1A"), 6: ("#1D8A50", "#FFFFFF"),
                7: ("#EA7A22", "#FFFFFF"), 8: ("#F2A0B5", "#1A1A1A")}

CSS = """
:root{--paper:#E9E7DF;--card:#F4F2EB;--ink:#22252A;--muted:#71767D;
--rule:#C9C6BB;--shu:#B5301E;--ai:#1F4E79;--ok:#1D6E45}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:"Hiragino Sans","Yu Gothic","Noto Sans JP",system-ui,sans-serif;
font-size:15px;line-height:1.55;font-feature-settings:"palt" 1;
font-variant-numeric:tabular-nums}
.wrap{max-width:620px;margin:0 auto;padding:0 14px 60px}
header{padding:22px 0 12px;border-bottom:2px solid var(--ink)}
h1{margin:0;font-size:28px;font-weight:800;letter-spacing:.16em}
.sub{margin-top:6px;font-size:12px;color:var(--muted)}
.race{border-bottom:1px solid var(--rule);padding:18px 0}
.rhead{display:flex;align-items:flex-start;gap:11px}
.stamp{flex:none;width:36px;height:36px;border:2px solid var(--shu);color:var(--shu);
display:flex;align-items:center;justify-content:center;font-size:18px;
font-weight:800;border-radius:50%}
.stamp.g-A{border-color:var(--ink);color:var(--ink)}
.stamp.g-B{border-color:var(--muted);color:var(--muted);border-width:1px}
.stamp.g-C{border-color:var(--rule);color:var(--muted);border-width:1px}
.rtitle{font-size:18px;font-weight:700}
.rmeta{font-size:12px;color:var(--muted);margin-top:2px}
.rec{border:2px solid var(--ink);padding:11px 12px;margin:12px 0}
.rec.b{border-color:var(--ai)}
.rec.skip{border-style:dashed;border-color:var(--rule);color:var(--muted)}
.rec-h{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.rec-mode{font-size:11px;color:var(--muted);letter-spacing:.08em}
.rec-type{font-size:18px;font-weight:800}
.rec-shape{font-size:12px;color:var(--muted);margin-left:auto}
.buy{margin:9px 0 8px;padding:0;list-style:none}
.buy li{display:flex;align-items:center;gap:4px;padding:4px 0;font-weight:700;
border-bottom:1px solid rgba(201,198,187,.45)}
.buy .o{margin-left:auto;font-weight:400;font-size:12px;color:var(--muted)}
.sep{font-weight:400;color:var(--muted);margin:0 2px}
.stats{display:flex;gap:16px;font-size:11px;color:var(--muted);
border-top:1px solid var(--rule);padding-top:7px}
.stats b{display:block;font-size:15px;color:var(--ink);line-height:1.3}
.fr{display:inline-flex;align-items:center;justify-content:center;
min-width:21px;height:21px;padding:0 3px;font-size:12px;font-weight:700;
border:1px solid rgba(0,0,0,.3)}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}
th{font-size:10px;color:var(--muted);font-weight:400;text-align:right;
padding-bottom:4px;border-bottom:1px solid var(--rule)}
th:nth-child(-n+2){text-align:left}
td{padding:5px 0;border-bottom:1px solid rgba(201,198,187,.45)}
td:not(:first-child):not(:nth-child(2)){text-align:right}
.mark{width:22px;font-size:16px;font-weight:700;color:var(--shu)}
.dim{color:var(--muted);font-size:12px}
.pos{color:var(--ok);font-weight:700}
footer{padding:24px 0;font-size:11px;color:var(--muted);line-height:1.8}
"""


def _chip(no, frame):
    bg, fg = FRAME_COLORS.get(int(frame or 1), ("#FFF", "#1A1A1A"))
    return f'<span class="fr" style="background:{bg};color:{fg}">{int(no)}</span>'


def _rec_block(rec, race, mode_label, cls):
    if not rec:
        return ""
    if rec.get("skip"):
        return (f'<div class="rec skip"><div class="rec-h">'
                f'<span class="rec-mode">{mode_label}</span>'
                f'<span class="rec-type">見送り</span></div>'
                f'<div style="font-size:12px;margin-top:5px">'
                f'{_html.escape(rec.get("reason", ""))}</div></div>')

    frames = {h["no"]: h["frame"] for h in race["horses"]}
    sep = '<span class="sep">→</span>' if "単" in rec["type"] else '<span class="sep">−</span>'
    lines = []
    for i, combo in enumerate(rec["combos"]):
        nums = sep.join(_chip(n, frames.get(n, 1)) for n in combo)
        odds = rec["detail"][i]["odds"] if i < len(rec["detail"]) else None
        lines.append(f'<li>{nums}<span class="o">{odds:.1f}倍</span></li>'
                     if odds else f"<li>{nums}</li>")
    hit = ""
    if rec.get("hit") is True:
        hit = '<span class="pos">的中</span>'
    elif rec.get("hit") is False:
        hit = '<span class="dim">不的中</span>'
    note = "" if rec.get("real_odds") else '<div class="dim">配当は推定です</div>'
    return f"""<div class="rec {cls}">
<div class="rec-h"><span class="rec-mode">{mode_label}</span>
<span class="rec-type">{rec['type']}</span>
<span class="dim">{rec['points']}点</span>{hit}
<span class="rec-shape">{_html.escape(rec['shape'])}</span></div>
<ul class="buy">{''.join(lines)}</ul>
<div class="stats">
<div><b>{rec['p']*100:.0f}%</b>的中率</div>
<div><b>{rec['odds']:.1f}倍</b>合成オッズ</div>
<div><b>{rec['ret']*100:.0f}%</b>期待回収率</div></div>{note}</div>"""


def _horse_table(race):
    rows = []
    show_fin = any(h.get("fin") for h in race["horses"])
    for h in race["horses"]:
        fin = f'<td>{h["fin"]}</td>' if show_fin else ""
        edge = h.get("edge")
        edge_s = f'{edge:.2f}' if edge is not None else "-"
        cls = "pos" if (edge and edge >= 1.25 and (h.get("gap") or 0) >= 1) else "dim"
        rows.append(
            f'<tr><td class="mark">{h["markA"]}</td>'
            f'<td>{_chip(h["no"], h["frame"])} {_html.escape(str(h["name"])[:12])}</td>'
            f'<td>{h["pA"]*100:.0f}%</td>'
            f'<td class="dim">{h["odds"]:.1f}</td>'
            f'<td class="{cls}">{edge_s}</td>{fin}</tr>')
    fin_h = "<th>着</th>" if show_fin else ""
    return (f'<table><tr><th>印</th><th>馬</th><th>3着内</th>'
            f'<th>単勝</th><th>妙味</th>{fin_h}</tr>{"".join(rows)}</table>')


def render(payload, title="予想") -> str:
    meta = payload.get("meta", {})
    blocks = []
    for r in payload["races"]:
        rec = r.get("recommend") or {}
        blocks.append(f"""<section class="race">
<div class="rhead"><span class="stamp g-{r['grade']}">{r['grade']}</span>
<span><span class="rtitle">{r['venue']}{r.get('race_no') or ''}R</span>
<div class="rmeta">{r.get('post_time','')} {r['surface']}{r['distance']}m　
{r['field_size']}頭　自信度 {r['conf']}　
{'両モデル一致' if r.get('agree') else 'モデル割れ'}</div></span></div>
{_rec_block(rec.get('A'), r, '当てにいく', '')}
{_rec_block(rec.get('B'), r, '妙味を狙う', 'b')}
{_horse_table(r)}</section>""")

    sub = "　".join(str(x) for x in (meta.get("date"), meta.get("venues")) if x)
    return f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>{_html.escape(title)}</h1>
<div class="sub">{_html.escape(sub)}　{len(payload['races'])}レース　更新 {payload.get('generated_at','')}</div>
</header>{''.join(blocks)}
<footer>期待回収率が100%を超えていても利益が出るとは限りません。
控除率は馬連・馬単22.5%、3連複25%、3連単27.5%。<br>
分析ツールであり投資助言ではありません。</footer></div></body></html>"""


def write(payload, path, title="予想"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(payload, title))
    return path
