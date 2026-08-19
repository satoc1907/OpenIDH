"""Pull each figure out of the report as a standalone SVG.

The report's charts are inline SVG styled by the page's stylesheet, so they do not
survive being copied out on their own. This lifts each one into its own file with
the styles it needs baked in — usable directly in a manuscript, in Illustrator or
Inkscape, and renderable by GitHub from the Markdown build.

    uv run python scripts/build_report.py       # writes results/stageb-report.html
    uv run python scripts/export_figures.py     # writes results/figures/fig-NN.svg

Light-mode tokens are resolved; figures for print should not depend on a viewer
theme. Interaction-only artefacts (hover hit areas) are dropped.
"""
from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# classes the charts actually paint with; anything else in the page CSS is layout
_CHART = {"grid", "sep", "tick", "pnl", "rowlab", "runlab", "val", "sm", "note", "dot",
          "ring", "whisk", "conn", "bar", "ood", "ind", "acc", "soft", "unused", "rel",
          "ideal", "dirlab", "muted", "diamond", "refline", "band", "hit", "mk"}
_FONT = ('font-family:system-ui,-apple-system,"Segoe UI","Hiragino Sans",'
         '"Yu Gothic UI","Noto Sans JP",sans-serif')


def _tokens(css: str) -> dict[str, str]:
    """Light-mode custom properties, taken from the first :root block."""
    m = re.search(r":root\{(.*?)\}", css, re.S)
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", m.group(1))) if m else {}


def _chart_css(css: str, tok: dict[str, str]) -> str:
    out = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        sel = sel.strip()
        if sel.startswith("@") or not sel.startswith("."):
            continue
        names = set(re.findall(r"\.([\w-]+)", sel))
        if not names or not names <= _CHART:
            continue
        body = re.sub(r"var\((--[\w-]+)\)", lambda m: tok.get(m.group(1), "#000"), body)
        body = body.replace("var(--surface)", "#fbfcfc")
        out.append(f"{sel}{{{body}}}")
    return "\n".join(out)


def export(html: str, out_dir: Path) -> list[dict]:
    tok = _tokens(html)
    style = _chart_css("\n".join(re.findall(r"<style>(.*?)</style>", html, re.S)), tok)
    figs, made = re.findall(r"(<svg\s+viewBox=.*?</svg>)", html, re.S), []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, svg in enumerate(figs, 1):
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        w, h = float(vb.group(1)), float(vb.group(2))
        title = re.search(r"<title>(.*?)</title>", svg)
        title = unescape(title.group(1)) if title else f"figure {i}"
        body = re.sub(r'<rect class="hit"[^>]*/>', "", svg)          # hover targets only
        body = re.sub(r"^<svg\s+", "", body).split(">", 1)[1].rsplit("</svg>", 1)[0]
        doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" '
               f'width="{w:g}" height="{h:g}" role="img">'
               f"<title>{title}</title>"
               f'<style>svg{{{_FONT};background:#fbfcfc}} text{{fill:{tok.get("--ink","#10161a")}}}\n'
               f"{style}</style>"
               f'<rect width="{w:g}" height="{h:g}" fill="#fbfcfc"/>{body}</svg>')
        p = out_dir / f"fig-{i:02d}.svg"
        p.write_text(doc, encoding="utf-8")
        made.append({"file": p.name, "title": title, "w": w, "h": h})
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(_ROOT / "results" / "stageb-report.html"))
    ap.add_argument("--out-dir", default=str(_ROOT / "results" / "figures"))
    a = ap.parse_args()
    src = Path(a.html)
    if not src.is_file():
        raise SystemExit(f"{src} not found — run scripts/build_report.py first")
    made = export(src.read_text(encoding="utf-8"), Path(a.out_dir))
    for m in made:
        print(f"  {m['file']}  {m['w']:.0f}x{m['h']:.0f}  {m['title']}")
    print(f"wrote {len(made)} figure(s) to {a.out_dir}")


if __name__ == "__main__":
    main()
