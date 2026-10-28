"""Animated hero banner: CatBoost trees growing one after another (boosting), an employee dot
falling down each tree, and the risk score filling up. Pure HTML/CSS/SVG - no extra libraries."""

CYCLE = 13  # seconds per loop


def _tree(cx, label, cls, leaf_colors):
    mid = [(cx - 36, 100), (cx + 36, 100)]
    leaves = [cx - 54, cx - 18, cx + 18, cx + 54]
    out = [f'<g class="tree {cls}">',
           f'<text x="{cx}" y="26" class="lab" text-anchor="middle">{label}</text>']
    for mx, my in mid:
        out.append(f'<line x1="{cx}" y1="48" x2="{mx}" y2="{my}" class="br"/>')
    for i, (mx, my) in enumerate([mid[0], mid[0], mid[1], mid[1]]):
        out.append(f'<line x1="{mx}" y1="{my}" x2="{leaves[i]}" y2="152" class="br"/>')
    out.append(f'<circle cx="{cx}" cy="48" r="9" class="nd"/>')
    for mx, my in mid:
        out.append(f'<circle cx="{mx}" cy="{my}" r="8" class="nd"/>')
    for lx, col in zip(leaves, leaf_colors):
        out.append(f'<rect x="{lx - 14}" y="145" width="28" height="15" rx="5" fill="{col}" class="leaf"/>')
    out.append("</g>")
    return "\n".join(out)


def hero_html():
    cxs = [80, 220, 360]
    starts = [0, 26, 52]                 # when each tree appears (% of the loop)
    css = []
    for k, (cx, a) in enumerate(zip(cxs, starts), 1):
        first = "0%" if a == 0 else f"0%,{a}%"
        css.append(f"@keyframes t{k}{{{first}{{opacity:0;transform:scale(.7)}}{a + 6}%{{opacity:1;transform:scale(1)}}"
                   f"93%{{opacity:1;transform:scale(1)}}98%,100%{{opacity:0;transform:scale(1)}}}}")
        b = a + 8
        r, m, l = (cx, 48), (cx - 36, 100), (cx - 54, 152)
        tr = lambda p: f"transform:translate({p[0]}px,{p[1]}px)"
        css.append(f"@keyframes b{k}{{0%,{b}%{{opacity:0;{tr(r)}}}{b + 2}%{{opacity:1;{tr(r)}}}"
                   f"{b + 6}%{{opacity:1;{tr(m)}}}{b + 11}%{{opacity:1;{tr(l)}}}{b + 14}%{{opacity:1;{tr(l)}}}"
                   f"{b + 16}%,100%{{opacity:0;{tr(l)}}}}}")
        css.append(f".t{k}{{animation:t{k} {CYCLE}s ease-out infinite}} .ball{k}{{animation:b{k} {CYCLE}s ease-in-out infinite}}")
    css.append(f"@keyframes pl1{{0%,20%{{opacity:0}}24%,93%{{opacity:1}}98%,100%{{opacity:0}}}}"
               f"@keyframes pl2{{0%,46%{{opacity:0}}50%,93%{{opacity:1}}98%,100%{{opacity:0}}}}"
               f".plus1{{animation:pl1 {CYCLE}s infinite}} .plus2{{animation:pl2 {CYCLE}s infinite}}")
    css.append(f"@keyframes fill{{0%,76%{{opacity:1;transform:scaleX(0)}}90%,93%{{opacity:1;transform:scaleX(.78)}}"
               f"98%,100%{{opacity:0;transform:scaleX(.78)}}}} .meter{{animation:fill {CYCLE}s ease-out infinite}}")
    cols = ["#008F87", "#FF686B", "#008F87", "#FF686B"]
    trees = "\n".join(_tree(cx, f"TREE {k}", f"t{k}", cols[k % 2:] + cols[:k % 2])
                      for k, cx in enumerate(cxs, 1))
    balls = "\n".join(f'<circle r="6" cx="0" cy="0" class="ball ball{k}" fill="#292929" stroke="#fff" stroke-width="2"/>'
                      for k in (1, 2, 3))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@800&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
html,body{{margin:0;background:transparent;font-family:'DM Sans',Inter,system-ui,sans-serif;color:#292929}}
.hero{{position:relative;overflow:hidden;background:#FFD43B;border-radius:28px;padding:34px 40px 64px;display:flex;
      flex-wrap:wrap;gap:24px;align-items:center;justify-content:space-between;box-sizing:border-box;min-height:300px}}
.copy{{flex:1 1 360px;max-width:560px;position:relative;z-index:2}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;display:flex;align-items:center;gap:8px}}
.dot{{width:9px;height:9px;border-radius:50%;background:#008F87;display:inline-block}}
h1{{font-family:'Barlow Condensed',Oswald,Impact,'Arial Narrow',sans-serif;font-weight:800;text-transform:uppercase;
   font-size:clamp(2.4rem,5.2vw,4rem);line-height:.92;margin:14px 0 14px;letter-spacing:.005em}}
h1 span{{color:#008F87}}
p{{font-size:16px;line-height:1.5;margin:0;max-width:480px;opacity:.88}}
.steps{{margin-top:18px;font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}}
.steps b{{color:#008F87}}
.art{{flex:0 1 440px;position:relative;z-index:2;min-width:300px}}
svg{{width:100%;height:auto;display:block;filter:drop-shadow(0 10px 22px rgba(41,41,41,.18))}}
.br{{stroke:#292929;stroke-width:2;stroke-linecap:round}} .nd{{fill:#292929}}
.lab{{font-size:10px;font-weight:700;letter-spacing:.16em;fill:#737373}}
.tree{{transform-box:fill-box;transform-origin:50% 0%}}
.leaf{{transform-box:fill-box;transform-origin:50% 100%;animation:pulse 2.4s ease-in-out infinite}}
@keyframes pulse{{50%{{transform:scaleY(1.35)}}}}
.ball{{opacity:0}} .meter{{transform-box:fill-box;transform-origin:0 50%}}
.plus{{font-size:30px;font-weight:700;fill:#292929;text-anchor:middle}}
.sticker{{position:absolute;right:34px;bottom:16px;z-index:3;background:#FF686B;color:#fff;font-weight:800;
         font-family:'Barlow Condensed',Oswald,Impact,sans-serif;text-transform:uppercase;letter-spacing:.06em;
         padding:8px 16px;border-radius:999px;transform:rotate(-5deg);font-size:15px;box-shadow:0 6px 14px rgba(255,104,107,.4)}}
.fl{{position:absolute;z-index:1;opacity:.9;animation:float 6s ease-in-out infinite}}
.f1{{left:46%;top:16px;width:22px;height:22px;border-radius:50%;background:#FF686B}}
.f2{{right:3%;top:14px;width:26px;height:26px;border:5px solid #008F87;border-radius:50%;animation-delay:-2s}}
.f3{{left:52%;bottom:22px;width:18px;height:18px;background:#292929;transform:rotate(20deg);animation-delay:-4s}}
@keyframes float{{50%{{translate:0 -12px}}}}
{chr(10).join(css)}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}} .ball{{opacity:0}} .meter{{transform:scaleX(.78)}}}}
</style></head><body>
<div class="hero">
  <i class="fl f1"></i><i class="fl f2"></i><i class="fl f3"></i>
  <div class="copy">
    <div class="eyebrow"><span class="dot"></span>CatBoost · gradient boosting · employee attrition</div>
    <h1>Predict who leaves. <span>Before they do.</span></h1>
    <p>Upload any HR dataset. Watch CatBoost grow its trees one by one, each correcting the last, then explain and score attrition risk.</p>
    <div class="steps"><b>01</b> Upload &nbsp;·&nbsp; <b>02</b> Train &nbsp;·&nbsp; <b>03</b> Animate &nbsp;·&nbsp; <b>04</b> Predict</div>
  </div>
  <div class="art">
    <svg viewBox="0 0 440 250" role="img" aria-label="Animated illustration: decision trees are added one by one and their scores add up to an attrition risk score">
      <rect x="0" y="0" width="440" height="250" rx="22" fill="#fff" opacity=".93"/>
      {trees}
      <text x="150" y="108" class="plus plus1">+</text><text x="290" y="108" class="plus plus2">+</text>
      {balls}
      <text x="24" y="196" class="lab">RISK SCORE = TREE 1 + TREE 2 + TREE 3 + …</text>
      <rect x="24" y="206" width="392" height="14" rx="7" fill="#F0ECDD"/>
      <defs><linearGradient id="g"><stop offset="0" stop-color="#008F87"/><stop offset="1" stop-color="#FF686B"/></linearGradient></defs>
      <rect x="24" y="206" width="392" height="14" rx="7" fill="url(#g)" class="meter"/>
      <text x="24" y="240" class="lab">LOW RISK</text><text x="416" y="240" class="lab" text-anchor="end">HIGH RISK</text>
    </svg>
  </div>
  <div class="sticker">Each tree fixes the last</div>
</div></body></html>"""


if __name__ == "__main__":
    open("/tmp/hero_test.html", "w").write(hero_html())
    print("written")
