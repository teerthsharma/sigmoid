"""Looking at the imagined future, with no display server and no new dependency.

A world model works by imagining. A supervisor watching a robot needs to see the
imagined future to hit the stop *before* the arm goes through the table, and when
a grasp fails needs to know whether the encoder ever saw the cup. Both of those
are looking-at problems, so this module draws.

**Why hand-rolled SVG instead of matplotlib.** The deployment target is a Jetson
on a bench: no display server, no internet, and a filesystem you do not want to
spend ~400 MB of. matplotlib is that 400 MB plus an argument about headless
backends; PIL is another install for the two hundred bytes of PNG header written
below. A string of inline SVG has neither problem -- it renders in any browser,
scp's as one file, and diffs. So every function here writes ONE self-contained
HTML file: inline CSS, inline SVG, base64 `data:` images, zero external
references, dark/light aware, in the same visual register as docs/index.html.

Self-containment is a testable claim rather than an intention, and `audit()` is
the test: it parses every SVG fragment with the stdlib XML parser and reports
anything pointing off-file. Three failures forced the rules it enforces.

An unescaped `&` or `<` from a gate `reason` string blanks the page in a strict
parser, so `html.escape` goes on every single interpolation.

No markup here uses a *named* HTML entity. `&mdash;` and `&times;` are HTML5
names, undefined in XML, and the first audit run died on `undefined entity` at
column 12808 of a page that rendered perfectly in a browser -- i.e. valid HTML
that no XML tool can read, which is a page you cannot verify. Numeric character
references (`&#8212;`) are legal in both, so those are all this module emits.

Nothing declares an `xmlns` either: inline SVG in HTML needs none, and declaring
one would force the "no http:// anywhere in the file" rule to carry an exception
for namespace URIs -- a rule with an exception is a rule that stops catching the
real CDN link.

Panels
    render_rollout        filmstrip / entity traces, grounded step marked
    render_heatmap        attention or contact weights over a frame
    render_compute_graph  stage boxes, shapes on the edges, bottleneck marked
    render_gate_timeline  gate score against its fire threshold, reasons listed
    dashboard             all of the above on one responsive page
"""

from __future__ import annotations

import base64
import html
import struct
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

__all__ = [
    "audit",
    "dashboard",
    "render_compute_graph",
    "render_gate_timeline",
    "render_heatmap",
    "render_rollout",
]

# --------------------------------------------------------------------------
# page shell
# --------------------------------------------------------------------------

# Lifted from docs/index.html so a rendered page and the project page read as
# one thing. Trimmed to what these panels use; the palette and the
# prefers-color-scheme block are copied verbatim on purpose.
_CSS = """
:root{--bg:#0b0d10;--panel:#12151a;--line:#232830;--ink:#e6e9ef;--dim:#98a2b3;
      --accent:#6ee7b7;--accent2:#7dd3fc;--warn:#fbbf24;
      --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:light){
  :root{--bg:#fbfcfd;--panel:#fff;--line:#e3e8ef;--ink:#111827;--dim:#5b6472;
        --accent:#059669;--accent2:#0369a1;--warn:#b45309}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 20px}
h1{font-size:29px;margin:26px 0 4px;letter-spacing:-.02em}
h2{font-size:19px;margin:32px 0 4px;letter-spacing:-.01em}
h2 .num{color:var(--accent);font-family:var(--mono);font-size:13px;margin-right:9px}
p.lede{color:var(--dim);margin:0 0 14px;font-size:14.5px}
code{font-family:var(--mono);font-size:.88em;background:var(--panel);
     border:1px solid var(--line);border-radius:4px;padding:.1em .38em}
figure{margin:14px 0;padding:16px;background:var(--panel);border:1px solid var(--line);
       border-radius:12px;overflow-x:auto}
figcaption{color:var(--dim);font-size:13px;margin-top:10px}
svg{max-width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;
   letter-spacing:.06em}
td.n,th.n{text-align:right;font-family:var(--mono)}
td.r{font-family:var(--mono);color:var(--warn)}
.note{border-left:3px solid var(--accent);background:var(--panel);padding:12px 16px;
      border-radius:0 8px 8px 0;margin:14px 0;font-size:14.5px}
.note.warn{border-left-color:var(--warn)}
.note b{font-family:var(--mono);color:var(--warn)}
.film{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(96px,1fr))}
.cell{border:1px solid var(--line);border-radius:8px;padding:5px;background:var(--bg);
      margin:0}
.cell img{width:100%;height:auto;display:block;border-radius:3px;
          image-rendering:pixelated}
.cell figcaption{margin-top:5px;font-size:11px;font-family:var(--mono)}
.cell.uncert{opacity:.5;border-style:dashed}
.cell.grounded{border-color:var(--warn);outline:2px solid var(--warn)}
.st{font:600 12px var(--mono);fill:var(--ink)}
.sd{font:11px var(--mono);fill:var(--dim)}
.sw{font:600 11px var(--mono);fill:var(--warn)}
.bx{fill:none;stroke:var(--accent2);stroke-width:1.4}
.bxw{fill:none;stroke:var(--warn);stroke-width:2.2}
.ln{stroke:var(--accent2);stroke-width:1.4;fill:none}
.mk{fill:var(--accent2)}
.ax{stroke:var(--line);stroke-width:1;fill:none}
.thr{stroke:var(--warn);stroke-width:1.3;stroke-dasharray:5 4;fill:none}
footer{margin:40px 0 36px;padding-top:16px;border-top:1px solid var(--line);
       color:var(--dim);font-size:12.5px}
"""


def _esc(value: Any) -> str:
    """Every string reaching the page goes through here.

    A gate reason is model-authored text, and one containing `<` or `&` is enough
    to blank a page in a strict XML parser. There is no path around this function.
    """
    return html.escape(str(value))


def _page(title: str, panels: Sequence[str]) -> str:
    body = "\n".join(p for p in panels if p)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f'<div class="wrap">\n<h1>{_esc(title)}</h1>\n{body}\n'
        "<footer>sigmoid.viz &#8212; self-contained page: inline CSS, inline SVG, "
        "base64 <code>data:</code> images, no external reference. "
        "Rendered without matplotlib, PIL or a display server.</footer>\n"
        "</div>\n</body>\n</html>\n"
    )


def _write(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _fig(svg_or_html: str, caption: str) -> str:
    return f"<figure>\n{svg_or_html}\n<figcaption>{caption}</figcaption>\n</figure>"


def _svg(width: float, height: float, body: str, label: str) -> str:
    # No xmlns: inline SVG in HTML is put in the SVG namespace by the HTML parser,
    # and omitting it keeps the "no http:// anywhere in the file" audit absolute.
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{_esc(label)}">{body}</svg>'
    )


def _fmt(x: float) -> str:
    """Short number, no trailing noise. Ratios and milliseconds both land here."""
    v = float(x)
    if not np.isfinite(v):
        return "inf" if v > 0 else "-inf"
    return "0" if v == 0 else f"{v:.3g}"


def _mapper(lo: float, hi: float, a: float, b: float) -> Callable[[float], float]:
    """Linear data->pixel map. A degenerate span centres rather than divides by 0."""
    lo, hi = float(lo), float(hi)
    span = hi - lo
    if not np.isfinite(span) or abs(span) < 1e-12:
        mid = (a + b) / 2.0
        return lambda _v: mid
    return lambda v: a + (float(v) - lo) * (b - a) / span


# --------------------------------------------------------------------------
# PNG, in stdlib
# --------------------------------------------------------------------------


def _png(pixels: np.ndarray) -> bytes:
    """Encode uint8 (H,W), (H,W,3) or (H,W,4) as PNG. zlib + struct, no PIL.

    Filter type 0 (none) on every row, so the "compression" is whatever zlib
    manages on raw rows -- worse than a real filter would give, and irrelevant:
    the images here are small thumbnails, and a Paeth filter would be thirty more
    lines to maintain for a few kilobytes.
    """
    a = np.ascontiguousarray(pixels, dtype=np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected (H,W), (H,W,3) or (H,W,4), got shape {a.shape}")
    h, w, c = a.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    rows = np.concatenate([np.zeros((h, 1, c), dtype=np.uint8), a], axis=1)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)),
            chunk(b"IEND", b""),
        ]
    )


def _u8(frame: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Float frame in [0,1] (or any range, with normalize=True) to uint8."""
    a = np.asarray(frame, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    if normalize:
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _thumb(img: np.ndarray, max_side: int) -> np.ndarray:
    """Stride-downsample, because a self-contained page pays for every pixel.

    Measured on a 32-step rollout of 256x256 frames of uniform noise -- the worst
    case, since noise is incompressible and defeats zlib entirely:

        max_side=96 (default)      0.32 MiB
        max_side=256 (no cap)      2.69 MiB

    Real frames compress far better (the same rollout on a smooth blob is 19 KiB
    either way), but a robot camera is closer to noise than to a blob, so the cap
    is on by default and `max_side` raises it when the detail matters more than
    the bytes.

    Striding rather than area-averaging: a single stray bright pixel surviving is
    more honest for a supervisor than a smoothed frame that hides it.
    """
    a = np.asarray(img)
    side = max(a.shape[0], a.shape[1])
    step = max(1, int(np.ceil(side / max_side)))
    return a[::step, ::step] if step > 1 else a


# --------------------------------------------------------------------------
# shared bits of the rollout panel
# --------------------------------------------------------------------------


def _first_fire(rollout: Any) -> tuple[int | None, str]:
    """Where the model stopped trusting itself, and how we know.

    `Rollout.grounded_at` is only set when `imagine(stop_on_gate=True)` broke out
    of the loop. `VisualForesight.rollout` and every planner call pass
    `stop_on_gate=False`, which leaves `grounded_at` None with a fired reading
    sitting in the list -- so a renderer that trusted the field alone would draw
    a clean page for a rollout the gate rejected. That is the exact silent
    omission this module exists to prevent.
    """
    at = getattr(rollout, "grounded_at", None)
    if at is not None:
        return int(at), "imagination stopped here"
    readings = tuple(getattr(rollout, "readings", ()) or ())
    for i, r in enumerate(readings):
        if getattr(r, "fire", False):
            return i, "gate fired (rollout was told not to stop)"
    return None, ""


def _certified(rollout: Any, k: int, safe_horizon: int | None, tolerance: float) -> tuple[int, str]:
    """Steps the error certificate covers, and where that number came from."""
    if safe_horizon is not None:
        return max(0, min(int(safe_horizon), k)), "supplied by the caller"
    cert = getattr(rollout, "certificate", None)
    if cert is None or not hasattr(cert, "error_bound"):
        return k, "no certificate on this rollout -- nothing shaded"
    n = 0
    while n < k and cert.error_bound(n + 1) <= tolerance:
        n += 1
    # Reported, not hidden: on measured data the worst-case bound is ~100x loose
    # (see RolloutCertificate) and this often answers 0, i.e. the whole rollout
    # shades as uncertified. That is the honest reading of that bound.
    return n, f"worst-case error_bound &#8804; tolerance {_fmt(tolerance)}"


def _observations(rollout: Any, decode: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    if decode is None:
        return np.asarray(getattr(rollout, "hiddens", np.zeros((0, 0))), dtype=np.float64)
    states = np.asarray(getattr(rollout, "states", np.zeros((0, 0))), dtype=np.float64)
    if not len(states):
        return np.zeros((0, 0))
    return np.stack([np.asarray(decode(z), dtype=np.float64) for z in states])


def _tape(k: int, grounded: int | None, certified: int) -> str:
    """One cell per step: certified, uncertified, and the grounded step in warn.

    Shared by both rollout layouts -- images and traces disagree about how to
    show a frame but agree completely about how to show time.
    """
    w, h, pad = 760.0, 66.0, 34.0
    cw = (w - 2 * pad) / max(k, 1)
    parts = [f'<rect class="ax" x="{pad:.1f}" y="14" width="{w - 2 * pad:.1f}" height="24"/>']
    every = max(1, k // 16)
    for i in range(k):
        x = pad + i * cw
        if grounded is not None and i == grounded:
            fill, op = "var(--warn)", "1"
        elif i < certified:
            fill, op = "var(--accent)", ".55"
        else:
            fill, op = "var(--dim)", ".22"
        parts.append(
            f'<rect x="{x:.2f}" y="14" width="{max(cw - 1.0, 0.6):.2f}" height="24" '
            f'fill="{fill}" opacity="{op}"/>'
        )
        if i % every == 0 or (grounded is not None and i == grounded):
            parts.append(
                f'<text class="sd" x="{x + cw / 2:.2f}" y="52" text-anchor="middle">{i}</text>'
            )
    if grounded is not None:
        gx = pad + grounded * cw + cw / 2
        parts.append(
            f'<text class="sw" x="{gx:.2f}" y="10" text-anchor="middle">grounded</text>'
        )
    parts.append(
        f'<rect x="{pad:.1f}" y="60" width="10" height="5" fill="var(--accent)" opacity=".55"/>'
        f'<text class="sd" x="{pad + 15:.1f}" y="66">certified</text>'
        f'<rect x="{pad + 90:.1f}" y="60" width="10" height="5" fill="var(--dim)" opacity=".22"/>'
        f'<text class="sd" x="{pad + 105:.1f}" y="66">beyond certificate</text>'
        f'<rect x="{pad + 250:.1f}" y="60" width="10" height="5" fill="var(--warn)"/>'
        f'<text class="sd" x="{pad + 265:.1f}" y="66">gate fired</text>'
    )
    return _svg(w, h, "".join(parts), "step tape: certified, uncertified, grounded step")


def _film(obs: np.ndarray, grounded: int | None, certified: int, max_side: int) -> str:
    """Image observations: one thumbnail per step, PNG in a data URI."""
    cells = []
    for i, frame in enumerate(obs):
        cls = "cell" if i < certified else "cell uncert"
        if grounded is not None and i == grounded:
            cls = "cell grounded"
        uri = _uri(_png(_u8(_thumb(frame, max_side))))
        tag = f"step {i}" + (" &#8226; GROUNDED" if grounded is not None and i == grounded else "")
        cells.append(
            f'<figure class="{cls}"><img alt="imagined step {i}" src="{uri}">'
            f"<figcaption>{tag}</figcaption></figure>"
        )
    return '<div class="film">' + "".join(cells) + "</div>"


def _traces(obs: np.ndarray, entity_dim: int, grounded: int | None, certified: int) -> str:
    """Low-dimensional state vectors: entities as points, one polyline per entity.

    Solid up to the certified horizon, dashed past it -- the same certified /
    uncertified split as the tape, expressed as the only thing a spatial plot has
    to spare.
    """
    k, d = obs.shape
    n = max(1, d // entity_dim)
    pts = obs[:, : n * entity_dim].reshape(k, n, entity_dim)
    pts = pts[:, :, :2] if entity_dim >= 2 else np.concatenate(
        [np.tile(np.arange(k, dtype=np.float64).reshape(k, 1, 1), (1, n, 1)), pts], axis=2
    )

    w, h, pad = 760.0, 400.0, 36.0
    fx = _mapper(pts[:, :, 0].min(), pts[:, :, 0].max(), pad, w - pad)
    fy = _mapper(pts[:, :, 1].min(), pts[:, :, 1].max(), h - pad, pad)  # y flipped
    parts = [
        f'<rect class="ax" x="{pad:.1f}" y="{pad:.1f}" '
        f'width="{w - 2 * pad:.1f}" height="{h - 2 * pad:.1f}"/>'
    ]
    for e in range(n):
        colour = f"hsl({(202 + 47 * e) % 360},68%,58%)"
        xy = [(fx(pts[t, e, 0]), fy(pts[t, e, 1])) for t in range(k)]
        solid = xy[: max(certified, 1)]
        dashed = xy[max(certified - 1, 0) :]
        if len(solid) > 1:
            pth = " ".join(f"{x:.1f},{y:.1f}" for x, y in solid)
            parts.append(
                f'<polyline points="{pth}" fill="none" stroke="{colour}" stroke-width="1.6"/>'
            )
        if len(dashed) > 1:
            pth = " ".join(f"{x:.1f},{y:.1f}" for x, y in dashed)
            parts.append(
                f'<polyline points="{pth}" fill="none" stroke="{colour}" stroke-width="1.6" '
                f'stroke-dasharray="4 4" opacity=".55"/>'
            )
        for t, (x, y) in enumerate(xy):
            op = 0.28 + 0.72 * (t / max(k - 1, 1))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{colour}" '
                         f'opacity="{op:.2f}"/>')
        x0, y0 = xy[0]
        parts.append(
            f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="5" fill="none" stroke="{colour}" '
            f'stroke-width="1.3"/>'
        )
        if grounded is not None and grounded < k:
            gx, gy = xy[grounded]
            parts.append(
                f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="8" fill="none" stroke="var(--warn)" '
                f'stroke-width="2"/>'
            )
    parts.append(
        f'<text class="sd" x="{pad:.1f}" y="{h - 12:.1f}">'
        f"{n} entities &#215; {entity_dim}D &#8226; hollow ring = step 0 &#8226; "
        f"solid = certified, dashed = beyond the certificate"
        + (" &#8226; warn ring = grounded step" if grounded is not None else "")
        + "</text>"
    )
    return _svg(w, h, "".join(parts), "imagined entity trajectories")


def _raster(obs: np.ndarray) -> str:
    """Fallback for a vector that is not a stack of entities: the (k, D) matrix.

    One row per step, per-matrix normalized. Not pretty, but it never lies about
    the shape it was given -- the alternative was guessing an entity width that
    does not divide the observation, which is exactly the silent mis-measurement
    state.select_cloud exists to stop.
    """
    uri = _uri(_png(_u8(obs, normalize=True)))
    return (
        f'<img alt="rollout observation matrix" src="{uri}" '
        f'style="width:100%;image-rendering:pixelated;border-radius:6px">'
    )


def _rollout_panel(
    rollout: Any,
    *,
    decode: Callable[[np.ndarray], np.ndarray] | None = None,
    safe_horizon: int | None = None,
    tolerance: float = 1.0,
    entity_dim: int = 2,
    max_side: int = 96,
) -> str:
    obs = _observations(rollout, decode)
    k = len(obs)
    grounded, how = _first_fire(rollout)
    certified, source = _certified(rollout, k, safe_horizon, tolerance)

    head = ['<h2><span class="num">01</span>Imagined future</h2>']
    if grounded is None:
        head.append(
            f'<div class="note"><b>grounded_at = None</b> &#8212; the gate held for all '
            f"{k} imagined steps.</div>"
        )
    else:
        readings = tuple(getattr(rollout, "readings", ()) or ())
        fired = readings[grounded] if grounded < len(readings) else None
        reason = getattr(fired, "reason", "?")
        score = getattr(fired, "score", float("nan"))
        head.append(
            f'<div class="note warn"><b>grounded_at = {grounded}</b> &#8212; the model '
            f"stopped trusting itself at step {grounded} of {k}: "
            f"<code>{_esc(reason)}</code> at score {_fmt(score)}. {_esc(how)}.</div>"
        )
    head.append(
        f'<p class="lede">Certified through step {max(certified - 1, 0)} '
        f"({certified} of {k} steps, {source}). Everything past it is the model "
        f"extrapolating outside its own bound.</p>"
    )

    if k == 0:
        return "\n".join([*head, _fig("<p>empty rollout &#8212; nothing imagined</p>", "0 steps")])

    if obs.ndim >= 3:
        art = _film(obs, grounded, certified, max_side)
        cap = (
            f"Image observations, {obs.shape[1]}&#215;{obs.shape[2]} downsampled to at most "
            f"{max_side} px per side. Dashed and faded = beyond the certificate; "
            f"warn outline = the grounded step."
        )
    # entity_dim < 1 is not an error: SigmoidConfig.entity_dim defaults to 0 and
    # means "this is a sequence state, not a set of entities", so a caller passing
    # world.config.entity_dim straight through must get the raster, not a
    # ZeroDivisionError from the modulo below.
    elif entity_dim >= 1 and obs.shape[1] >= entity_dim and obs.shape[1] % entity_dim == 0:
        art = _traces(obs, entity_dim, grounded, certified)
        cap = f"State vectors read as {obs.shape[1] // entity_dim} entities of width {entity_dim}."
    else:
        art = _raster(obs)
        cap = (
            f"Observation width {obs.shape[1]} is not a multiple of entity_dim={entity_dim}, "
            f"so the ({k}, {obs.shape[1]}) matrix is drawn directly: one row per step, "
            f"normalized over the whole rollout."
        )
    return "\n".join([*head, _fig(art, cap), _fig(_tape(k, grounded, certified), "Step tape.")])


def render_rollout(
    rollout: Any,
    path: str | Path,
    *,
    decode: Callable[[np.ndarray], np.ndarray] | None = None,
    safe_horizon: int | None = None,
    tolerance: float = 1.0,
    entity_dim: int = 2,
    max_side: int = 96,
) -> Path:
    """Filmstrip of an imagined future, with the grounded step marked.

    `decode` maps one world state to an observation, e.g. `encoder.decode`. Omit
    it and `rollout.hiddens` is used. A 2D (or 2D+channel) return picks the image
    filmstrip; a 1D return picks entity traces when `entity_dim` divides its
    width, and a raw (steps, D) raster otherwise.

    `safe_horizon` is the certified step count -- pass `world.safe_horizon(tol)`
    or `ForesightResult.safe_steps`. Left None it is derived from the rollout's
    own certificate at `tolerance`, and the page says which.
    """
    panel = _rollout_panel(
        rollout,
        decode=decode,
        safe_horizon=safe_horizon,
        tolerance=tolerance,
        entity_dim=entity_dim,
        max_side=max_side,
    )
    return _write(path, _page("sigmoid rollout", [panel]))


# --------------------------------------------------------------------------
# heatmap
# --------------------------------------------------------------------------


def _grid(weights: np.ndarray, max_cells: int) -> tuple[np.ndarray, bool]:
    """Weights as a 2D grid, block-mean reduced to at most max_cells per side."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.ndim == 1:
        side = int(round(np.sqrt(w.size)))
        w = w.reshape(side, side) if side * side == w.size else w.reshape(1, -1)
    elif w.ndim != 2:
        raise ValueError(f"weights must be 1D or 2D, got shape {w.shape}")

    reduced = False
    for axis in (0, 1):
        n = w.shape[axis]
        if n <= max_cells:
            continue
        # block mean by trimming to a multiple then reshaping -- signed weights
        # must average, not sub-sample, or a +1/-1 pair vanishes at one stride
        # and survives at the next.
        step = int(np.ceil(n / max_cells))
        keep = (n // step) * step
        w = np.moveaxis(w, axis, 0)[:keep]
        w = w.reshape(keep // step, step, -1).mean(axis=1)
        w = np.moveaxis(w, 0, axis)
        reduced = True
    return w, reduced


def _heatmap_panel(frame: np.ndarray, weights: np.ndarray, *, max_cells: int = 40) -> str:
    img = np.asarray(frame, dtype=np.float64)
    if img.ndim not in (2, 3):
        raise ValueError(f"frame must be (H,W) or (H,W,C), got shape {img.shape}")
    uri = _uri(_png(_u8(img)))
    grid, reduced = _grid(weights, max_cells)
    gh, gw = grid.shape

    signed = bool(grid.min() < -1e-12)
    peak = float(np.abs(grid).max())
    if signed:
        # Diverging, because a signed weight has a meaningful zero and a
        # sequential ramp over it reads -1 and +1 as opposite ends of "more".
        scale = peak if peak > 1e-12 else 1.0
        norm = np.abs(grid) / scale
        kind = "diverging"
        legend = "amber = positive, blue = negative, transparent at zero"
    else:
        lo, hi = float(grid.min()), float(grid.max())
        norm = (grid - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(grid)
        kind = "sequential"
        legend = "single hue, opacity rising with weight"

    vw = 640.0
    vh = vw * img.shape[0] / max(img.shape[1], 1)
    cw, ch = vw / gw, vh / gh
    parts = [
        f'<image href="{uri}" x="0" y="0" width="{vw:.1f}" height="{vh:.1f}" '
        f'preserveAspectRatio="none" style="image-rendering:pixelated"/>'
    ]
    for i in range(gh):
        for j in range(gw):
            alpha = float(norm[i, j]) * 0.72
            if alpha < 0.01:
                continue
            colour = (
                ("var(--warn)" if grid[i, j] > 0 else "var(--accent2)")
                if signed
                else "var(--accent2)"
            )
            parts.append(
                f'<rect x="{j * cw:.2f}" y="{i * ch:.2f}" width="{cw:.2f}" height="{ch:.2f}" '
                f'fill="{colour}" opacity="{alpha:.3f}"/>'
            )
    pi, pj = np.unravel_index(int(np.argmax(np.abs(grid))), grid.shape)
    parts.append(
        f'<rect class="bxw" x="{pj * cw:.2f}" y="{pi * ch:.2f}" width="{cw:.2f}" '
        f'height="{ch:.2f}"/>'
        f'<text class="sw" x="{pj * cw + cw / 2:.2f}" y="{max(pi * ch - 4, 10):.2f}" '
        f'text-anchor="middle">peak {_fmt(grid[pi, pj])}</text>'
    )
    svg = _svg(vw, vh, "".join(parts), f"{kind} weight overlay on a frame")

    caption = (
        f"<b>{kind}</b> colormap ({legend}), chosen because the weights "
        f"{'contain negatives' if signed else 'are unsigned'}. "
        f"Grid {gh}&#215;{gw}"
        + (f" (block-mean reduced to &#8804;{max_cells} per side)" if reduced else "")
        + f", range {_fmt(grid.min())} to {_fmt(grid.max())}. "
        f"Frame {img.shape[0]}&#215;{img.shape[1]} written as a PNG by "
        f"<code>zlib</code> + <code>struct</code> and inlined as a "
        f"<code>data:</code> URI; the overlay is SVG rects on top."
    )
    return "\n".join(
        [
            '<h2><span class="num">02</span>Weight overlay</h2>',
            '<p class="lede">Did the encoder ever look at the thing that mattered?</p>',
            _fig(svg, caption),
        ]
    )


def render_heatmap(
    frame: np.ndarray, weights: np.ndarray, path: str | Path, *, max_cells: int = 40
) -> Path:
    """Overlay attention weights, contact points or per-patch salience on a frame.

    Signed weights get a diverging map, unsigned a sequential one, and the page
    states which. Frames are float arrays in [0, 1], greyscale or RGB.
    """
    panel = _heatmap_panel(frame, weights, max_cells=max_cells)
    return _write(path, _page("sigmoid heatmap", [panel]))


# --------------------------------------------------------------------------
# compute graph
# --------------------------------------------------------------------------


def _stage(entry: dict, index: int) -> tuple[str, str, float]:
    name = str(entry.get("name") or entry.get("stage") or f"stage {index}")
    shape = entry.get("shape", entry.get("output_shape", entry.get("output", "")))
    ms = entry.get("latency_ms", entry.get("ms", entry.get("latency", float("nan"))))
    return name, str(shape) if shape is not None else "", float(ms)


def _graph_panel(stages: Sequence[dict]) -> str:
    items = [_stage(dict(s), i) for i, s in enumerate(stages)]
    if not items:
        return "\n".join(
            [
                '<h2><span class="num">03</span>Compute graph</h2>',
                _fig("<p>no stages supplied</p>", "0 stages"),
            ]
        )
    times = np.asarray([ms for _, _, ms in items], dtype=np.float64)
    finite = np.where(np.isfinite(times), times, 0.0)
    slowest = int(np.argmax(finite))
    total = float(finite.sum())
    worst = float(finite[slowest]) or 1.0

    bw, bh, gap, pad = 152.0, 70.0, 66.0, 18.0
    tail = 118.0  # room for the terminal tensor label after the last box
    w = pad * 2 + len(items) * bw + max(len(items) - 1, 0) * gap + tail
    h = 150.0
    top = 44.0
    parts = [
        '<defs><marker id="sgv-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        'orient="auto"><path class="mk" d="M0,0 L9,4.5 L0,9 z"/></marker></defs>'
    ]
    for i, (name, shape, ms) in enumerate(items):
        x = pad + i * (bw + gap)
        cx = x + bw / 2
        cls = "bxw" if i == slowest else "bx"
        parts.append(f'<rect class="{cls}" rx="9" x="{x:.1f}" y="{top:.1f}" '
                     f'width="{bw:.1f}" height="{bh:.1f}"/>')
        parts.append(f'<text class="st" x="{cx:.1f}" y="{top + 26:.1f}" '
                     f'text-anchor="middle">{_esc(name)}</text>')
        # Name and latency in the box, tensor shape on the outgoing edge, and
        # nothing written twice. The first version printed shape and ms in both
        # places and the graph read as noise -- five labels per stage where three
        # carry information.
        label = "sw" if i == slowest else "sd"
        parts.append(f'<text class="{label}" x="{cx:.1f}" y="{top + 48:.1f}" '
                     f'text-anchor="middle">{_fmt(ms)} ms</text>')
        bar = (bw - 24) * (float(finite[i]) / worst)
        parts.append(
            f'<rect x="{x + 12:.1f}" y="{top + 58:.1f}" width="{max(bar, 0.0):.1f}" height="6" '
            f'rx="3" fill="{"var(--warn)" if i == slowest else "var(--accent)"}" opacity=".8"/>'
        )
        if i == slowest:
            share = 100.0 * float(finite[i]) / total if total > 0 else 0.0
            parts.append(
                f'<text class="sw" x="{cx:.1f}" y="{top - 12:.1f}" text-anchor="middle">'
                f"&#9660; bottleneck &#8226; {share:.0f}% of {_fmt(total)} ms</text>"
            )
        ey = top + bh / 2
        if i + 1 < len(items):
            x1, x2 = x + bw + 4, x + bw + gap - 6
            parts.append(
                f'<line class="ln" x1="{x1:.1f}" y1="{ey:.1f}" x2="{x2:.1f}" y2="{ey:.1f}" '
                f'marker-end="url(#sgv-arrow)"/>'
                f'<text class="sd" x="{(x1 + x2) / 2:.1f}" y="{ey - 8:.1f}" '
                f'text-anchor="middle">{_esc(shape) or "tensor"}</text>'
                f'<text class="sd" x="{(x1 + x2) / 2:.1f}" y="{ey + 17:.1f}" '
                f'text-anchor="middle">{_fmt(ms)} ms</text>'
            )
        else:
            # the last stage has no outgoing edge, so its output shape would
            # otherwise appear only in the table
            parts.append(
                f'<text class="sd" x="{x + bw + 8:.1f}" y="{ey + 4:.1f}">'
                f"{_esc(shape) or 'tensor'}</text>"
            )
    svg = _svg(w, h, "".join(parts), "compute graph with the bottleneck stage marked")
    rows = "".join(
        f"<tr><td>{_esc(n)}</td><td>{_esc(s) or '&#8212;'}</td><td class='n'>{_fmt(ms)}</td>"
        f"<td class='n'>{(100 * float(finite[i]) / total) if total else 0:.0f}%</td>"
        f"<td>{'bottleneck' if i == slowest else ''}</td></tr>"
        for i, (n, s, ms) in enumerate(items)
    )
    table = (
        "<table><thead><tr><th>stage</th><th>tensor</th><th class='n'>ms</th>"
        "<th class='n'>share</th><th>note</th></tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )
    return "\n".join(
        [
            '<h2><span class="num">03</span>Compute graph</h2>',
            f'<p class="lede">{len(items)} stages, {_fmt(total)} ms end to end. '
            f"Bottleneck: <code>{_esc(items[slowest][0])}</code>.</p>",
            _fig(
                svg,
                "Edges carry the tensor flowing along them and the latency of the stage "
                "it left; bars scale to the slowest stage. Latency is a property of a "
                "stage, so it is also written inside the box.",
            ),
            table,
        ]
    )


def render_compute_graph(stages: Sequence[dict], path: str | Path) -> Path:
    """rqt_graph-style stage boxes with the bottleneck marked.

    `stages` is a plain list of dicts so any caller can feed it:

        [{"name": "vision encode", "shape": "(64,64)", "latency_ms": 1.8}, ...]

    Accepts `name`/`stage`, `shape`/`output_shape`/`output`, and
    `latency_ms`/`ms`/`latency`.
    """
    return _write(path, _page("sigmoid compute graph", [_graph_panel(stages)]))


# --------------------------------------------------------------------------
# gate timeline
# --------------------------------------------------------------------------


def _timeline_panel(readings: Sequence[Any], *, threshold: float = 1.0) -> str:
    n = len(readings)
    if n == 0:
        return "\n".join(
            [
                '<h2><span class="num">04</span>Gate timeline</h2>',
                _fig("<p>no gate readings</p>", "0 readings"),
            ]
        )

    def col(attr: str) -> np.ndarray:
        return np.asarray(
            [float(getattr(r, attr, float("nan")) or 0.0) for r in readings], dtype=np.float64
        )

    score = col("score")
    series = [
        ("score", score, "var(--ink)", 2.0),
        ("sheaf", col("sheaf_score"), "var(--accent)", 1.1),
        ("manifold", col("manifold_score"), "var(--accent2)", 1.1),
    ]
    rank = col("rank_score")
    if np.nanmax(rank) > 0:
        series.append(("rank", rank, "hsl(292,60%,64%)", 1.1))

    # Log axis: the score is a ratio in calibration-quantile units, so it is a
    # multiplicative scale -- real states measured a median 0.56 and rollouts
    # under random actions a median 26 (control.py). On a linear axis the
    # threshold line, which is the whole point of the plot, lands in the noise.
    flat = np.concatenate([s for _, s, _, _ in series])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    lo = min(float(flat.min()) if flat.size else 0.1, threshold * 0.5)
    hi = max(float(flat.max()) if flat.size else 1.0, threshold * 2.0)
    lo, hi = max(lo, 1e-3), max(hi, 1e-2)

    w, h = 820.0, 300.0
    l, r, t, b = 54.0, 16.0, 22.0, 62.0
    fx = _mapper(0, max(n - 1, 1), l, w - r)
    fy = _mapper(np.log10(lo), np.log10(hi), h - b, t)

    def ypx(v: float) -> float:
        return fy(np.log10(max(float(v), lo * 0.5)))

    parts = [f'<rect class="ax" x="{l:.1f}" y="{t:.1f}" width="{w - l - r:.1f}" '
             f'height="{h - t - b:.1f}"/>']
    decade = int(np.floor(np.log10(lo)))
    while decade <= int(np.ceil(np.log10(hi))):
        v = 10.0**decade
        if lo <= v <= hi:
            y = ypx(v)
            parts.append(
                f'<line class="ax" x1="{l:.1f}" y1="{y:.1f}" x2="{w - r:.1f}" y2="{y:.1f}"/>'
                f'<text class="sd" x="{l - 6:.1f}" y="{y + 4:.1f}" text-anchor="end">'
                f"{_fmt(v)}</text>"
            )
        decade += 1
    ty = ypx(threshold)
    parts.append(
        f'<line class="thr" x1="{l:.1f}" y1="{ty:.1f}" x2="{w - r:.1f}" y2="{ty:.1f}"/>'
        f'<text class="sw" x="{w - r - 2:.1f}" y="{ty - 6:.1f}" text-anchor="end">'
        f"fire threshold {_fmt(threshold)}</text>"
    )
    for label, values, colour, width in series:
        pth = " ".join(f"{fx(i):.1f},{ypx(v):.1f}" for i, v in enumerate(values))
        op = "1" if label == "score" else ".6"
        parts.append(
            f'<polyline points="{pth}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" opacity="{op}"/>'
        )
    fires = [i for i, rd in enumerate(readings) if getattr(rd, "fire", False)]
    for i in fires:
        parts.append(
            f'<circle cx="{fx(i):.1f}" cy="{ypx(score[i]):.1f}" r="4.5" fill="var(--warn)"/>'
        )
    # Only the first fire is labelled on the chart. Once a rollout leaves the
    # manifold it keeps firing -- 19 fires in a 32-step rollout is normal -- and
    # labelling each one stacked nineteen numbers into an unreadable smear. The
    # first is the actionable one; the table below carries the rest.
    if fires:
        first = fires[0]
        # flip the label inward near the right edge, or it draws off the canvas
        right = fx(first) > (l + w - r) / 2
        anchor, dx = ("end", -6.0) if right else ("start", 6.0)
        parts.append(
            f'<text class="sw" x="{fx(first) + dx:.1f}" y="{ypx(score[first]) - 9:.1f}" '
            f'text-anchor="{anchor}">first fire, step {first}</text>'
        )
    every = max(1, -(-n // 12))
    for i in range(0, n, every):
        parts.append(
            f'<text class="sd" x="{fx(i):.1f}" y="{h - b + 16:.1f}" text-anchor="middle">'
            f"{i}</text>"
        )
    parts.append(f'<text class="sd" x="{l:.1f}" y="{h - 26:.1f}">step &#8226; log axis</text>')
    lx = l
    for label, _v, colour, _wd in series:
        parts.append(
            f'<rect x="{lx:.1f}" y="{h - 14:.1f}" width="10" height="4" fill="{colour}"/>'
            f'<text class="sd" x="{lx + 14:.1f}" y="{h - 9:.1f}">{_esc(label)}</text>'
        )
        lx += 34 + 8 * len(label)
    svg = _svg(w, h, "".join(parts), "gate score over time against the fire threshold")

    if fires:
        rows = "".join(
            f"<tr><td class='n'>{i}</td><td class='n'>{_fmt(score[i])}</td>"
            f"<td class='r'>{_esc(getattr(readings[i], 'reason', '?'))}</td></tr>"
            for i in fires
        )
        table = (
            "<table><thead><tr><th class='n'>step</th><th class='n'>score</th>"
            f"<th>reason</th></tr></thead><tbody>{rows}</tbody></table>"
        )
        note = (
            f'<div class="note warn"><b>{len(fires)} fire(s)</b> &#8212; first at step '
            f"{fires[0]}: <code>{_esc(getattr(readings[fires[0]], 'reason', '?'))}</code>.</div>"
        )
    else:
        table = ""
        note = (
            f'<div class="note">No fire in {n} steps &#8212; the gate held. '
            f"Peak score {_fmt(np.nanmax(score))} against a threshold of "
            f"{_fmt(threshold)}.</div>"
        )
    return "\n".join(
        [
            '<h2><span class="num">04</span>Gate timeline</h2>',
            '<p class="lede">Why it refused: which stalk crossed, and when.</p>',
            note,
            _fig(
                svg,
                "Faint lines are the component stalks; the solid line is their max, "
                "which is what fires. Every amber dot is a fire and only the first is "
                "labelled &#8212; the table below lists them all.",
            ),
            table,
        ]
    )


def render_gate_timeline(
    readings: Sequence[Any], path: str | Path, *, threshold: float = 1.0
) -> Path:
    """Gate score over time, the fire threshold as a line, every fire's reason listed.

    `readings` is a sequence of `sigmoid.sheaf.GateReading` (or anything with
    `.score`, `.fire`, `.reason`).
    """
    panel = _timeline_panel(readings, threshold=threshold)
    return _write(path, _page("sigmoid gate timeline", [panel]))


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------

_PANEL_KEYS = frozenset(
    {
        "rollout",
        "heatmap",
        "stages",
        "readings",
        "title",
        "decode",
        "safe_horizon",
        "tolerance",
        "entity_dim",
        "max_side",
        "max_cells",
        "threshold",
    }
)


def dashboard(path: str | Path, **panels: Any) -> Path:
    """One page with every supplied panel on it.

        dashboard(path, rollout=r, safe_horizon=4, heatmap=(frame, weights),
                  stages=[...], readings=r.readings)

    An unknown keyword raises rather than being dropped: a page that silently
    omits the panel you asked for is worse than no page.
    """
    unknown = set(panels) - _PANEL_KEYS
    if unknown:
        raise TypeError(
            f"unknown panel(s) {sorted(unknown)}; known: {sorted(_PANEL_KEYS)}"
        )
    out: list[str] = []
    if panels.get("rollout") is not None:
        out.append(
            _rollout_panel(
                panels["rollout"],
                decode=panels.get("decode"),
                safe_horizon=panels.get("safe_horizon"),
                tolerance=float(panels.get("tolerance", 1.0)),
                entity_dim=int(panels.get("entity_dim", 2)),
                max_side=int(panels.get("max_side", 96)),
            )
        )
    if panels.get("heatmap") is not None:
        frame, weights = panels["heatmap"]
        out.append(_heatmap_panel(frame, weights, max_cells=int(panels.get("max_cells", 40))))
    if panels.get("stages") is not None:
        out.append(_graph_panel(panels["stages"]))
    if panels.get("readings") is not None:
        out.append(
            _timeline_panel(panels["readings"], threshold=float(panels.get("threshold", 1.0)))
        )
    if not out:
        out.append('<p class="lede">no panels supplied</p>')
    return _write(path, _page(str(panels.get("title", "sigmoid dashboard")), out))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def audit(page: str | Path) -> dict[str, Any]:
    """Parse every SVG fragment and hunt for anything pointing off-file.

    A renderer that emits broken HTML is worthless, and both failure modes are
    silent in a browser: an unescaped `&` makes a strict parser drop the subtree,
    and a CDN link makes the page blank on a robot with no route to the internet.
    So this is not a test helper that happens to live here -- `demo()` runs it on
    its own output, and so does every test.

    Returns `{"svg_fragments": n, "external": [...], "mentions": n, "bytes": n}`
    and raises `xml.etree.ElementTree.ParseError` on a malformed fragment.

    `external` is what a browser would actually fetch: every `src`/`href`/CSS
    `url()` value that is not a `data:` URI or a fragment, plus any `@import`.
    `mentions` counts bare `http://` / `https://` occurrences anywhere in the
    file. The two are deliberately separate, because a *gate reason* containing a
    URL is escaped inert text and fails no robot, while one `href` to a CDN
    blanks the page -- collapsing them would mean either tolerating the CDN or
    failing every page whose data happens to quote a link.
    """
    text = str(page) if "<" in str(page) else Path(page).read_text(encoding="utf-8")

    fragments = 0
    cursor = 0
    while True:
        start = text.find("<svg", cursor)
        if start < 0:
            break
        end = text.find("</svg>", start)
        if end < 0:
            raise ET.ParseError(f"unterminated <svg> at offset {start}")
        ET.fromstring(text[start : end + 6])  # raises on a stray < or a bare &
        fragments += 1
        cursor = end + 6

    external: list[str] = []
    if "@import" in text:
        external.append("@import")
    for opener, closer in (('src="', '"'), ('href="', '"'), ("url(", ")")):
        cursor = 0
        while True:
            at = text.find(opener, cursor)
            if at < 0:
                break
            close = text.find(closer, at + len(opener))
            value = text[at + len(opener) : close].strip("'\"")
            if not value.startswith(("data:", "#")):
                external.append(f"{opener}{value[:60]}")
            cursor = close + 1
    return {
        "svg_fragments": fragments,
        "external": external,
        "mentions": text.count("http://") + text.count("https://"),
        "bytes": len(text.encode("utf-8")),
    }


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def demo() -> None:
    """Render all five panels off a real fitted world model and audit every byte."""
    import tempfile
    import time

    from .engine import SigmoidConfig, SigmoidWorldModel
    from .foresight import InverseDynamics, _swarm
    from .sheaf import GateReading
    from .vision import TopoImageEncoder

    outdir = Path(tempfile.gettempdir()) / "sigmoid_viz"
    obs, acts = _swarm()
    world = SigmoidWorldModel(
        config=SigmoidConfig(window=10, linear_dim=12, entity_dim=2, action_dim=8)
    ).fit(obs, actions=acts)
    window = obs[0][:10]

    # An action sequence that starts inside the calibration distribution and is
    # then ramped out of it, because that is the case a supervisor page exists
    # for: the gate holds (score ~0.45) for fourteen steps and then the rollout
    # leaves the manifold and the score runs to 1e10. Zero actions never fire, so
    # a demo on zeros would render the one page that proves nothing.
    rng = np.random.default_rng(7)
    drive, plan = np.zeros(8), []
    for t in range(32):
        drive = 0.85 * drive + 0.3 * rng.normal(scale=0.5, size=8)
        plan.append(np.clip(drive * (1.0 + max(0, t - 15) * 0.55), -1.0, 1.0))

    # stop_on_gate=False on purpose: it leaves grounded_at None with a fired
    # reading in the list, so the renderer has to find the fire itself.
    roll = world.imagine(window, 32, actions=np.asarray(plan), stop_on_gate=False)
    safe = world.safe_horizon(5.0)
    p = render_rollout(roll, outdir / "rollout.html", safe_horizon=safe)
    a = audit(p)
    grounded, _ = _first_fire(roll)
    print(
        f"  rollout 32 steps   {a['bytes'] / 1024:7.1f} KiB   {a['svg_fragments']} svg   "
        f"grounded_at={grounded}   certified={safe}"
    )
    assert not a["external"], a["external"]
    assert f"grounded_at = {grounded}" in p.read_text(encoding="utf-8")
    assert a["bytes"] < 2_000_000, "a 32-step page must not need megabytes"

    # ---- the image branch, on a world model fitted to actual pixels so that
    # decode() really does return a frame rather than a reshaped feature vector
    side = 16
    yy, xx = np.mgrid[0:side, 0:side]
    frames = np.stack(
        [
            np.clip(
                0.18 + 0.75 * np.exp(-((xx - (2.5 + 0.34 * t)) ** 2 + (yy - 8) ** 2) / 5.5), 0, 1
            )
            for t in range(72)
        ]
    )
    pixels = frames.reshape(len(frames), -1)
    pworld = SigmoidWorldModel(config=SigmoidConfig(window=8, linear_dim=16)).fit([pixels])
    proll = pworld.imagine(pixels[:8], 12, stop_on_gate=False)
    film = render_rollout(
        proll,
        outdir / "rollout_frames.html",
        decode=lambda z: pworld.encoder.decode(z).reshape(side, side),
        safe_horizon=4,
    )
    fa = audit(film)
    print(
        f"  rollout filmstrip  {fa['bytes'] / 1024:7.1f} KiB   {len(proll)} png thumbnails "
        f"inlined from a {side}x{side} pixel world model, {fa['svg_fragments']} svg"
    )
    assert not fa["external"] and "data:image/png;base64," in film.read_text(encoding="utf-8")

    # ---- heatmap: the PNG writer, and both colormaps
    venc = TopoImageEncoder()
    frame = frames[40]
    patch = venc.encode(frame)[venc.config.n_thresholds + 2 :][: venc.config.patch_grid**2]
    hp = render_heatmap(frame, patch, outdir / "heatmap_seq.html")
    signed = patch - patch.mean()
    hs = render_heatmap(frame, signed, outdir / "heatmap_div.html")
    seq_text, div_text = hp.read_text(encoding="utf-8"), hs.read_text(encoding="utf-8")
    assert "<b>sequential</b>" in seq_text and "<b>diverging</b>" in div_text
    png = _png(_u8(frame))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (frame.shape[1], frame.shape[0])
    print(
        f"  heatmap            sequential {audit(hp)['bytes'] / 1024:.1f} KiB, "
        f"diverging {audit(hs)['bytes'] / 1024:.1f} KiB   "
        f"png writer ok ({len(png)} B for {frame.shape[0]}x{frame.shape[1]})"
    )

    # ---- compute graph, on measured latencies rather than invented ones
    idm = InverseDynamics(history=2).fit(obs[0], acts[0])
    z0 = world.observe(window)

    def timed(fn: Callable[[], Any], reps: int) -> float:
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    stages = [
        {
            "name": "vision encode",
            "shape": f"(48,48)->({venc.n_features},)",
            "latency_ms": timed(lambda: venc.encode(frame), 20),
        },
        {
            "name": "world model",
            "shape": f"({world.state_dim},)",
            "latency_ms": timed(lambda: world.operator.step(z0, np.zeros(8)), 500),
        },
        {
            "name": "gate",
            "shape": f"({world.state_dim},)->score",
            "latency_ms": timed(lambda: world.gate.read(z0), 500),
        },
        {
            "name": "IDM",
            "shape": "(2,8)->(8,)",
            "latency_ms": timed(lambda: idm.predict(obs[0][4:6], obs[0][6]), 500),
        },
    ]
    g = render_compute_graph(stages, outdir / "graph.html")
    gt = g.read_text(encoding="utf-8")
    slow = max(stages, key=lambda s: s["latency_ms"])["name"]
    assert "bottleneck" in gt and slow in gt
    print(
        "  compute graph      "
        + "  ".join(f"{s['name']} {s['latency_ms']:.3f}ms" for s in stages)
        + f"   -> bottleneck {slow}"
    )

    # ---- gate timeline, including a reason string that would break the page
    hostile = GateReading(
        sheaf_residual=1.0,
        manifold_distance=1.0,
        sheaf_score=3.0,
        manifold_score=0.4,
        score=3.0,
        fire=True,
        reason='off_manifold <script>alert("x")</script> & <img src="http://evil/x">',
    )
    readings = [*roll.readings, hostile]
    tl = render_gate_timeline(readings, outdir / "timeline.html")
    tt = tl.read_text(encoding="utf-8")
    ta = audit(tl)  # would raise ParseError if the hostile string leaked into SVG
    assert "<script>" not in tt, "a reason string escaped into live markup"
    assert "&lt;script&gt;" in tt
    # the reason quotes a URL and is still not a reference: escaped text fetches
    # nothing, which is exactly why audit() reports the two separately
    assert not ta["external"] and ta["mentions"] == 1
    fires = sum(1 for r in readings if r.fire)
    print(
        f"  gate timeline      {len(readings)} readings, {fires} fire(s), "
        f"hostile reason escaped ({ta['mentions']} inert url mention, "
        f"{len(ta['external'])} live refs), {ta['svg_fragments']} svg well-formed"
    )

    # ---- dashboard
    dash = dashboard(
        outdir / "dashboard.html",
        title="sigmoid supervisor",
        rollout=roll,
        safe_horizon=safe,
        heatmap=(frame, signed),
        stages=stages,
        readings=readings,
    )
    da = audit(dash)
    text = dash.read_text(encoding="utf-8")
    for probe in ("Imagined future", "Weight overlay", "Compute graph", "Gate timeline"):
        assert probe in text, f"dashboard dropped the {probe} panel"
    print(
        f"  dashboard          {da['bytes'] / 1024:7.1f} KiB   {da['svg_fragments']} svg "
        f"well-formed   external refs {len(da['external'])}   "
        f"url mentions {da['mentions']} (the escaped one in the hostile reason)"
    )
    assert not da["external"]
    print(f"\n  wrote {outdir}")
    print("demo ok")


if __name__ == "__main__":
    demo()
