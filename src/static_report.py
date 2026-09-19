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
.modes{position:sticky;top:0;z-index:5;background:var(--paper);
padding:10px 0 9px;border-bottom:1px solid var(--rule);margin-bottom:4px}
.modes .row{display:flex;border:1px solid var(--ink);border-radius:2px;overflow:hidden}
.modes button{flex:1;font:inherit;font-size:13px;padding:8px 4px;background:none;
border:none;color:var(--ink);cursor:pointer}
.modes button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.modes button:focus-visible{outline:2px solid var(--shu);outline-offset:-2px}
.hint{font-size:11px;color:var(--muted);margin-top:6px}
.idx{margin:14px 0 4px;border-top:1px solid var(--rule)}
.idx a{display:flex;align-items:center;gap:9px;padding:9px 2px;text-decoration:none;
color:var(--ink);border-bottom:1px solid rgba(201,198,187,.55)}
.idx .g{flex:none;width:22px;height:22px;border:1.5px solid var(--shu);color:var(--shu);
border-radius:50%;display:flex;align-items:center;justify-content:center;
font-size:12px;font-weight:700}
.idx .g.g-A{border-color:var(--ink);color:var(--ink)}
.idx .g.g-B,.idx .g.g-C{border-color:var(--rule);color:var(--muted);border-width:1px}
.idx .nm{font-weight:700;font-size:14px;min-width:74px}
.idx .bt{font-size:12px;color:var(--muted)}
.idx .hp{margin-left:auto;font-size:12px;color:var(--muted)}
.sec-h{font-size:11px;color:var(--muted);letter-spacing:.1em;margin:22px 0 0}
.none{padding:20px 2px;font-size:13px;color:var(--muted)}
.warn{margin-top:7px;padding:7px 9px;border:1px solid var(--shu);
background:rgba(181,48,30,.06);font-size:11px;line-height:1.6;color:var(--shu)}
.warn b{font-weight:700}
.banner{border:2px solid var(--shu);background:rgba(181,48,30,.07);color:var(--shu);
padding:10px 12px;margin:12px 0;font-size:12px;line-height:1.7}
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
    if rec.get("real_odds"):
        note = ""
    else:
        note = ('<div class="warn">この券種はまだ発売前のため、配当は単勝オッズからの'
                '<b>推定値</b>です。実際のオッズとは大きく異なることがあります。'
                '発売後に再実行すると実配当に変わります。</div>')
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


def _sort_key(r):
    """発走時刻の早い順。時刻が無ければ場とレース番号で。"""
    t = str(r.get("post_time") or "")
    if ":" in t:
        try:
            h, m = t.split(":")
            return (0, int(h) * 60 + int(m), str(r.get("venue")), r.get("race_no") or 0)
        except ValueError:
            pass
    return (1, 0, str(r.get("venue")), r.get("race_no") or 0)


def _index_row(r):
    rec = (r.get("recommend") or {}).get("A") or {}
    if rec.get("skip"):
        bet, hit = "見送り", ""
    elif rec:
        bet = f"{rec['type']} {rec['points']}点"
        hit = f"的中 {rec['p']*100:.0f}%"
    else:
        bet, hit = "", ""
    label = f"{r['venue']}{r.get('race_no') or ''}R"
    return (f'<a href="#r{r["race_id"]}" data-grade="{r["grade"]}" data-conf="{r["conf"]}">'
            f'<span class="g g-{r["grade"]}">{r["grade"]}</span>'
            f'<span class="nm">{label}</span>'
            f'<span class="bt">{bet}</span>'
            f'<span class="hp">{hit}</span></a>')


MODE_JS = """
(function(){
  var btns=document.querySelectorAll('.modes button');
  var idx=document.getElementById('idx');
  var wrapSec=document.getElementById('secs');
  var rows=Array.prototype.slice.call(idx.children);
  var secs=Array.prototype.slice.call(wrapSec.children);
  var order={S:3,A:2,B:1,C:0};
  function apply(mode){
    var pick=[];
    rows.forEach(function(a,i){
      var g=a.getAttribute('data-grade');
      var on=(mode==='all')||order[g]>=2;
      a.style.display=on?'':'none';
      secs[i].style.display=on?'':'none';
      if(on)pick.push(i);
    });
    if(mode==='pick'){
      pick.sort(function(x,y){
        return parseFloat(rows[y].getAttribute('data-conf'))-
               parseFloat(rows[x].getAttribute('data-conf'));});
    }
    pick.forEach(function(i){idx.appendChild(rows[i]);wrapSec.appendChild(secs[i]);});
    var empty=document.getElementById('empty');
    if(empty)empty.style.display=pick.length?'none':'';
    var hint=document.getElementById('hint');
    if(hint)hint.textContent=(mode==='all')
      ?'発走の早い順に全'+rows.length+'レース':'A級以上を自信度の高い順に'+pick.length+'レース';
  }
  btns.forEach(function(b){
    b.addEventListener('click',function(){
      btns.forEach(function(x){x.setAttribute('aria-pressed',String(x===b));});
      apply(b.getAttribute('data-mode'));
    });
  });
  apply('all');
})();
"""


def render(payload, title="予想") -> str:
    meta = payload.get("meta", {})
    races = sorted(payload["races"], key=_sort_key)

    est = 0
    for r in races:
        for m in ("A", "B"):
            rec = (r.get("recommend") or {}).get(m)
            if rec and not rec.get("skip") and not rec.get("real_odds"):
                est += 1
    banner = ""
    if est:
        banner = (f'<div class="banner">この時点では馬連などが未発売のため、'
                  f'{est}件の買い目で<b>配当が推定値</b>になっています。'
                  f'単勝オッズから逆算した参考値で、実際のオッズとはずれます。'
                  f'発売後（開催当日の朝以降）に再実行すると実配当に変わります。</div>')

    blocks = []
    for r in races:
        rec = r.get("recommend") or {}
        blocks.append(f"""<section class="race" id="r{r['race_id']}"
data-grade="{r['grade']}" data-conf="{r['conf']}">
<div class="rhead"><span class="stamp g-{r['grade']}">{r['grade']}</span>
<span><span class="rtitle">{r['venue']}{r.get('race_no') or ''}R</span>
<div class="rmeta">{r.get('post_time','')} {r['surface']}{r['distance']}m　
{r['field_size']}頭　自信度 {r['conf']}　
{'両モデル一致' if r.get('agree') else 'モデル割れ'}</div></span></div>
{_rec_block(rec.get('A'), r, '当てにいく', '')}
{_rec_block(rec.get('B'), r, '妙味を狙う', 'b')}
{_horse_table(r)}</section>""")

    n_pick = sum(1 for r in races if r["grade"] in ("S", "A"))
    sub = "　".join(str(x) for x in (meta.get("date"), meta.get("venues")) if x)
    updated = _html.escape(str(meta.get("updated") or payload.get("generated_at", "")))

    return f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>{_html.escape(title)}</h1>
<div class="sub">{_html.escape(sub)}　更新 {updated}</div></header>
<div class="modes"><div class="row">
<button type="button" data-mode="all" aria-pressed="true">全レース</button>
<button type="button" data-mode="pick" aria-pressed="false">勝負レース（A級以上）</button>
</div><div class="hint" id="hint">発走の早い順に全{len(races)}レース　
自信度の高いレースだけ見るには右のボタン</div></div>
{banner}
<div class="idx" id="idx">{''.join(_index_row(r) for r in races)}</div>
<div class="none" id="empty" style="display:none">A級以上のレースがありません。</div>
<div class="sec-h">各レースの予想</div>
<div id="secs">{''.join(blocks)}</div>
<footer>期待回収率が100%を超えていても利益が出るとは限りません。
控除率は馬連・馬単22.5%、3連複25%、3連単27.5%。<br>
分析ツールであり投資助言ではありません。</footer></div>
<script>{MODE_JS}</script></body></html>"""


def write(payload, path, title="予想"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(payload, title))
    return path
