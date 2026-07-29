"""Runnable checks for the renderers. `python tests/test_viz.py`.

A renderer that emits broken HTML is worthless, and every way it breaks is silent
in a browser -- a dropped subtree, a blank panel, an image that never loads on a
robot with no route to a CDN. So this file does not check that rendering "worked";
it checks the four things that actually fail:

    well-formed XML      every SVG fragment parses with the stdlib parser
    self-contained       nothing a browser would fetch off-file
    inside the canvas    no element drawn outside its own viewBox
    the data is there    grounded_at, the bottleneck, every fire reason

No world model is fitted here. `Rollout` is a frozen dataclass of arrays and the
renderers only read it, so the fixtures build one directly -- a `fit()` per test
would spend seconds proving nothing about the drawing. `sigmoid/viz.py demo()`
covers the real-model path.

Everything is written under `tempfile.TemporaryDirectory`; nothing lands in the
repo.
"""

from __future__ import annotations

import base64
import contextlib
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.engine import Rollout
from sigmoid.operator import RolloutCertificate
from sigmoid.sheaf import GateReading
from sigmoid.viz import (
    audit,
    dashboard,
    render_compute_graph,
    render_gate_timeline,
    render_heatmap,
    render_rollout,
)

HOSTILE = 'off_manifold <script>alert("x")</script> & <img src="http://evil/x"> \'q\''
"""A reason string built to break the page: raw <, raw &, quotes, and a URL."""


# ---- fixtures ---------------------------------------------------------------


@contextlib.contextmanager
def out(name: str = "page.html"):
    """A path in a throwaway directory. Nothing is ever written to the repo."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / name


def reading(score: float, *, fire: bool = False, reason: str = "ok") -> GateReading:
    return GateReading(
        sheaf_residual=score * 0.1,
        manifold_distance=score * 0.2,
        sheaf_score=score * 0.9,
        manifold_score=score * 0.5,
        score=score,
        fire=fire,
        reason=reason,
    )


def rollout(
    steps: int = 8,
    *,
    width: int = 8,
    grounded_at: int | None = None,
    fire_at: int | None = None,
    reason: str = "off_manifold",
    rho: float = 0.8,
    step_rmse: float = 0.02,
) -> Rollout:
    """A Rollout of the shape the engine produces, without paying for a fit.

    `grounded_at` is the stop_on_gate=True case; `fire_at` is the stop_on_gate=
    False case, where a reading fires but the field stays None.
    """
    rng = np.random.default_rng(0)
    hiddens = np.cumsum(rng.normal(scale=0.4, size=(steps, width)), axis=0)
    marked = grounded_at if grounded_at is not None else fire_at
    readings = tuple(
        reading(
            9.0 if marked is not None and i >= marked else 0.45,
            fire=marked is not None and i >= marked,
            reason=reason if marked is not None and i >= marked else "ok",
        )
        for i in range(steps)
    )
    return Rollout(
        states=rng.normal(size=(steps, 6)),
        hiddens=hiddens,
        readings=readings,
        grounded_at=grounded_at,
        certificate=RolloutCertificate(
            rho=rho, contractive=rho < 1.0, step_rmse=step_rmse, horizon=steps, bound=0.1
        ),
    )


def frame(h: int = 24, w: int = 24) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(0.2 + 0.7 * np.exp(-((xx - w / 3) ** 2 + (yy - h / 2) ** 2) / 12.0), 0, 1)


STAGES = [
    {"name": "vision encode", "shape": "(1,3,64,64)", "latency_ms": 1.84},
    {"name": "world model", "shape": "(1,44)", "latency_ms": 0.09},
    {"name": "planner", "shape": "(8,8)", "latency_ms": 12.7},
    {"name": "IDM", "shape": "(1,8)", "latency_ms": 0.01},
]


def every_page() -> list[tuple[str, str]]:
    """(label, markup) for one page from each renderer, hostile strings included."""
    pages = []
    with out() as p:
        rd = [reading(0.4), reading(0.9), reading(14.0, fire=True, reason=HOSTILE)]
        r = rollout(12, fire_at=6, reason=HOSTILE)
        pages.append(("rollout", render_rollout(r, p, safe_horizon=4).read_text("utf-8")))
    with out() as p:
        pages.append(
            (
                "rollout-images",
                render_rollout(
                    rollout(6, width=36),
                    p,
                    decode=lambda z: frame(6, 6),
                    safe_horizon=3,
                ).read_text("utf-8"),
            )
        )
    with out() as p:
        heat = render_heatmap(frame(), np.linspace(-1, 2, 16), p)
        pages.append(("heatmap", heat.read_text("utf-8")))
    with out() as p:
        hostile_stages = [*STAGES, {"name": HOSTILE, "shape": "<&>", "latency_ms": 0.5}]
        pages.append(("graph", render_compute_graph(hostile_stages, p).read_text("utf-8")))
    with out() as p:
        pages.append(("timeline", render_gate_timeline(rd, p).read_text("utf-8")))
    with out() as p:
        pages.append(
            (
                "dashboard",
                dashboard(
                    p,
                    rollout=rollout(10, grounded_at=5, reason=HOSTILE),
                    heatmap=(frame(), np.linspace(-1, 1, 9)),
                    stages=STAGES,
                    readings=rd,
                ).read_text("utf-8"),
            )
        )
    return pages


def svg_fragments(markup: str) -> list[str]:
    frags, cursor = [], 0
    while (start := markup.find("<svg", cursor)) >= 0:
        end = markup.find("</svg>", start)
        assert end > start, "unterminated <svg>"
        frags.append(markup[start : end + 6])
        cursor = end + 6
    return frags


# ---- the four things that actually fail -------------------------------------


def test_every_svg_fragment_is_wellformed_xml():
    """The check that catches an unescaped & or a stray <, on every renderer."""
    total = 0
    for label, markup in every_page():
        info = audit(markup)  # raises ParseError on a malformed fragment
        assert info["svg_fragments"] == len(svg_fragments(markup))
        assert info["svg_fragments"] > 0 or label == "rollout-images"
        total += info["svg_fragments"]
    assert total >= 8, f"only {total} fragments checked"


def test_no_page_references_anything_off_file():
    """A robot has no internet. Every src/href/url() must be a data: URI."""
    for label, markup in every_page():
        assert not audit(markup)["external"], f"{label}: {audit(markup)['external']}"


def test_a_url_in_a_reason_is_a_mention_not_a_reference():
    """The distinction that keeps the no-external rule usable on hostile data."""
    with out() as p:
        markup = render_gate_timeline(
            [reading(3.0, fire=True, reason=HOSTILE)], p
        ).read_text("utf-8")
    info = audit(markup)
    assert info["mentions"] >= 1, "the escaped URL should still be counted"
    assert not info["external"], "escaped text is not a fetch"


def test_nothing_is_drawn_outside_its_own_viewbox():
    """Catches the silent layout bug: a label placed off-canvas is invisible."""
    checked = 0
    for label, markup in every_page():
        for fragment in svg_fragments(markup):
            root = ET.fromstring(fragment)
            _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
            for el in root.iter():
                for attr, limit in (
                    ("x", vw), ("x1", vw), ("x2", vw), ("cx", vw),
                    ("y", vh), ("y1", vh), ("y2", vh), ("cy", vh),
                ):
                    value = el.get(attr)
                    if value is None:
                        continue
                    assert -2 <= float(value) <= limit + 2, (
                        f"{label}: <{el.tag} {attr}={value}> outside 0..{limit}"
                    )
            checked += 1
    assert checked >= 8


def test_a_hostile_reason_string_cannot_corrupt_the_page():
    with out() as p:
        markup = render_gate_timeline(
            [reading(0.5), reading(9.0, fire=True, reason=HOSTILE)], p
        ).read_text("utf-8")
    ET.fromstring(svg_fragments(markup)[0])  # parses at all
    assert "<script>" not in markup, "raw markup from a reason string reached the page"
    assert "&lt;script&gt;" in markup, "the reason was dropped instead of escaped"
    assert "alert" in markup, "the reason must still be readable to a human"


# ---- rollout ----------------------------------------------------------------


def test_rollout_marks_grounded_at_prominently():
    with out() as p:
        markup = render_rollout(rollout(10, grounded_at=6), p).read_text("utf-8")
    assert "grounded_at = 6" in markup, "the most important number is missing"
    assert markup.count("grounded") >= 3, "one mention is not prominent"
    assert "note warn" in markup, "the grounded banner must be the warn variant"


def test_rollout_finds_the_fire_when_grounded_at_is_none():
    """stop_on_gate=False leaves the field None with a fired reading in the list."""
    r = rollout(10, fire_at=4)
    assert r.grounded_at is None
    with out() as p:
        markup = render_rollout(r, p).read_text("utf-8")
    assert "grounded_at = 4" in markup, "a rejected rollout rendered as clean"
    assert "told not to stop" in markup, "the page must say where the number came from"


def test_rollout_says_so_when_the_gate_held():
    with out() as p:
        markup = render_rollout(rollout(6), p).read_text("utf-8")
    assert "grounded_at = None" in markup and "gate held" in markup


def test_rollout_shades_beyond_the_safe_horizon():
    with out() as p:
        markup = render_rollout(rollout(12), p, safe_horizon=5).read_text("utf-8")
    assert "5 of 12 steps" in markup, "the certified count is not stated"
    assert "supplied by the caller" in markup
    assert "beyond certificate" in markup, "the tape needs its legend"


def test_rollout_derives_the_horizon_from_the_certificate_when_not_given():
    # rho=0.5, step_rmse=0.1 -> bound converges to 0.2, so tolerance 1.0 covers all
    with out() as p:
        markup = render_rollout(rollout(9, rho=0.5, step_rmse=0.1), p).read_text("utf-8")
    assert "9 of 9 steps" in markup and "error_bound" in markup
    # a large one-step error certifies nothing, and the page must say so, not hide it
    with out() as p:
        tight = render_rollout(rollout(9, rho=0.5, step_rmse=8.0), p).read_text("utf-8")
    assert "0 of 9 steps" in tight


def test_rollout_picks_the_filmstrip_for_image_observations():
    with out() as p:
        markup = render_rollout(
            rollout(5, width=36), p, decode=lambda z: frame(8, 8), safe_horizon=2
        ).read_text("utf-8")
    assert markup.count("data:image/png;base64,") == 5, "one PNG per step"
    assert markup.count("cell uncert") == 3, "steps past the horizon must be shaded"
    assert "Image observations" in markup


def test_rollout_picks_traces_for_entity_vectors():
    with out() as p:
        markup = render_rollout(rollout(7, width=8), p, safe_horizon=7).read_text("utf-8")
    # 8 wide / entity_dim 2 = 4 entities, all-certified so one polyline each
    assert markup.count("<polyline") == 4, "one trace per entity"
    assert "4 entities" in markup


def test_rollout_falls_back_to_a_raster_for_an_indivisible_width():
    with out() as p:
        markup = render_rollout(rollout(6, width=7), p).read_text("utf-8")
    assert "not a multiple of entity_dim" in markup, "a silent guess would be worse"
    assert "data:image/png;base64," in markup


def test_rollout_accepts_entity_dim_zero():
    """SigmoidConfig.entity_dim defaults to 0 for sequence states; passing it
    through must reach the raster, not divide by zero."""
    with out() as p:
        markup = render_rollout(rollout(6, width=8), p, entity_dim=0).read_text("utf-8")
    assert "entity_dim=0" in markup and "data:image/png;base64," in markup


def test_empty_rollout_renders_instead_of_crashing():
    empty = Rollout(
        states=np.zeros((0, 6)),
        hiddens=np.zeros((0, 8)),
        readings=(),
        grounded_at=None,
        certificate=RolloutCertificate(
            rho=0.5, contractive=True, step_rmse=0.01, horizon=0, bound=0.0
        ),
    )
    with out() as p:
        markup = render_rollout(empty, p).read_text("utf-8")
    assert "empty rollout" in markup


def test_a_thirtytwo_step_rollout_stays_small():
    with out() as p:
        info = audit(render_rollout(rollout(32, fire_at=14), p, safe_horizon=2))
    assert info["bytes"] < 200_000, f"{info['bytes']} bytes for 32 vector steps"
    with out() as p:
        images = audit(
            render_rollout(rollout(32, width=64), p, decode=lambda z: frame(64, 64))
        )
    assert images["bytes"] < 2_000_000, f"{images['bytes']} bytes for 32 image steps"


# ---- heatmap ----------------------------------------------------------------


def test_heatmap_states_which_colormap_it_used():
    with out() as p:
        signed = render_heatmap(frame(), np.linspace(-1, 1, 16), p).read_text("utf-8")
    with out() as p:
        unsigned = render_heatmap(frame(), np.linspace(0, 1, 16), p).read_text("utf-8")
    assert "<b>diverging</b>" in signed and "contain negatives" in signed
    assert "<b>sequential</b>" in unsigned and "are unsigned" in unsigned
    assert "<b>diverging</b>" not in unsigned


def test_heatmap_writes_a_real_png():
    """Decode the data URI and read the header: this is the PNG writer's test."""
    with out() as p:
        markup = render_heatmap(frame(30, 40), np.ones((4, 4)), p).read_text("utf-8")
    tag = 'href="data:image/png;base64,'
    start = markup.index(tag) + len(tag)
    png = base64.b64decode(markup[start : markup.index('"', start)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG signature"
    assert png[12:16] == b"IHDR"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (40, 30), f"IHDR says {width}x{height}, frame was 40x30"
    assert png[24] == 8 and png[25] == 0, "expected 8-bit greyscale"
    assert png[-8:] == b"IEND\xae\x42\x60\x82", "missing or corrupt IEND"


def test_heatmap_png_survives_rgb_frames():
    with out() as p:
        markup = render_heatmap(
            np.stack([frame(12, 12)] * 3, axis=-1), np.ones((3, 3)), p
        ).read_text("utf-8")
    tag = 'href="data:image/png;base64,'
    start = markup.index(tag) + len(tag)
    png = base64.b64decode(markup[start : markup.index('"', start)])
    assert struct.unpack(">II", png[16:24]) == (12, 12) and png[25] == 2, "not RGB colour type"


def test_heatmap_marks_the_peak_cell():
    weights = np.zeros((4, 4))
    weights[2, 1] = 5.0
    with out() as p:
        markup = render_heatmap(frame(), weights, p).read_text("utf-8")
    assert "peak 5" in markup, "the strongest cell must be labelled"


def test_heatmap_block_means_a_large_grid_instead_of_emitting_10000_rects():
    with out() as p:
        markup = render_heatmap(frame(64, 64), np.ones((100, 100)), p).read_text("utf-8")
    assert "block-mean reduced" in markup, "a silent downsample is a lie"
    assert markup.count("<rect") < 1700, "grid was not reduced"
    # Averaging vs sub-sampling is not cosmetic on signed weights: striding an
    # alternating +1/-1 row by 3 reproduces +-1 exactly, so a page showing the
    # full range is a page that threw two thirds of the weights away.
    with out() as p:
        flat = render_heatmap(
            frame(8, 8), np.tile(np.array([[1.0, -1.0]]), (2, 50)), p
        ).read_text("utf-8")
    assert "range -0.333 to 0.333" in flat, "a +1/-1 pair was sub-sampled, not averaged"


def test_heatmap_accepts_a_flat_patch_vector():
    """TopoImageEncoder emits per-patch occupancy flat; a square must be recovered."""
    with out() as p:
        markup = render_heatmap(frame(), np.arange(16.0), p).read_text("utf-8")
    assert "Grid 4&#215;4" in markup


# ---- compute graph ----------------------------------------------------------


def test_compute_graph_marks_the_bottleneck():
    with out() as p:
        markup = render_compute_graph(STAGES, p).read_text("utf-8")
    assert "bottleneck" in markup
    slowest = "planner"
    at = markup.index("bottleneck")
    assert slowest in markup[max(0, at - 400) : at + 400], "the wrong stage is marked"
    assert "bxw" in markup, "the bottleneck box must be visually distinct"
    assert markup.count("bottleneck") >= 3, "mark it in the box, the lede and the table"


def test_compute_graph_shows_every_shape_and_latency():
    with out() as p:
        markup = render_compute_graph(STAGES, p).read_text("utf-8")
    for stage in STAGES:
        assert stage["name"] in markup, f"{stage['name']} vanished"
        assert stage["shape"].replace("&", "&amp;") in markup, f"{stage['shape']} vanished"
    assert "12.7" in markup and "1.84" in markup


def test_compute_graph_accepts_alternative_key_names():
    with out() as p:
        markup = render_compute_graph(
            [{"stage": "a", "output_shape": "(1,)", "ms": 3.0}, {"stage": "b", "ms": 9.0}], p
        ).read_text("utf-8")
    assert "&#215;" in markup or ">a<" in markup
    at = markup.index("bottleneck")
    assert ">b<" in markup[max(0, at - 400) : at + 400]


def test_compute_graph_with_no_stages_does_not_crash():
    with out() as p:
        assert "no stages" in render_compute_graph([], p).read_text("utf-8")


# ---- gate timeline ---------------------------------------------------------


def test_gate_timeline_labels_every_fire_reason():
    reasons = ["sheaf_inconsistent", "off_manifold", "rank_collapsed"]
    fires = [reading(3.0 + i, fire=True, reason=r) for i, r in enumerate(reasons)]
    readings = [reading(0.4), *fires]
    with out() as p:
        markup = render_gate_timeline(readings, p).read_text("utf-8")
    for r in reasons:
        assert r in markup, f"fire reason {r} was not labelled"
    assert "3 fire(s)" in markup
    assert "fire threshold 1" in markup, "the threshold line needs its label"


def test_gate_timeline_says_when_the_gate_held():
    with out() as p:
        markup = render_gate_timeline([reading(0.3), reading(0.7)], p).read_text("utf-8")
    assert "No fire in 2 steps" in markup and "gate held" in markup


def test_gate_timeline_draws_the_component_stalks():
    with out() as p:
        markup = render_gate_timeline([reading(0.4), reading(2.0, fire=True)], p).read_text("utf-8")
    for stalk in ("sheaf", "manifold"):
        assert f">{stalk}<" in markup, f"{stalk} stalk missing from the legend"
    assert markup.count("<polyline") == 3, "score plus two stalks"


def test_gate_timeline_survives_a_score_range_of_ten_decades():
    """Real fires run to 1e10 while the threshold is 1.0; a linear axis loses both."""
    readings = [reading(0.45), reading(1.2, fire=True), reading(4.1e10, fire=True)]
    with out() as p:
        markup = render_gate_timeline(readings, p).read_text("utf-8")
    ET.fromstring(svg_fragments(markup)[0])
    assert "log axis" in markup
    assert "4.1e+10" in markup, "the extreme score must still be reported"


def test_gate_timeline_with_no_readings_does_not_crash():
    with out() as p:
        assert "no gate readings" in render_gate_timeline([], p).read_text("utf-8")


# ---- dashboard -------------------------------------------------------------


def test_dashboard_carries_every_panel():
    with out() as p:
        markup = dashboard(
            p,
            title="supervisor",
            rollout=rollout(8, grounded_at=3),
            heatmap=(frame(), np.linspace(-1, 1, 16)),
            stages=STAGES,
            readings=[reading(0.5), reading(6.0, fire=True, reason="off_manifold")],
        ).read_text("utf-8")
    for panel in ("Imagined future", "Weight overlay", "Compute graph", "Gate timeline"):
        assert panel in markup, f"the {panel} panel was dropped"
    assert "grounded_at = 3" in markup and "bottleneck" in markup
    assert "<title>supervisor</title>" in markup


def test_dashboard_renders_a_single_panel():
    with out() as p:
        markup = dashboard(p, stages=STAGES).read_text("utf-8")
    assert "Compute graph" in markup and "Gate timeline" not in markup


def test_dashboard_rejects_an_unknown_panel():
    """A typo must raise, not silently produce a page missing what was asked for."""
    with out() as p:
        try:
            dashboard(p, rollouts=rollout(4))
        except TypeError as exc:
            assert "rollouts" in str(exc) and "known" in str(exc)
        else:
            raise AssertionError("an unknown panel keyword was accepted")


def test_dashboard_does_not_scroll_the_body_sideways():
    with out() as p:
        markup = dashboard(p, rollout=rollout(6), stages=STAGES).read_text("utf-8")
    css = markup.split("<style>")[1].split("</style>")[0]
    assert "svg{max-width:100%;height:auto" in css, "svg must scale down to the page"
    assert "overflow-x:auto" in css, "a wide panel must scroll inside its own figure"
    assert "minmax(" in css, "the filmstrip grid must reflow, not overflow"
    assert ".wrap{max-width:" in css, "the column must be bounded"
    assert "min-width" not in css, "a min-width in px is what forces a body scrollbar"
    assert 'meta name="viewport"' in markup, "no viewport meta means no mobile scaling"


# ---- the auditor itself ----------------------------------------------------


def test_audit_rejects_malformed_svg():
    try:
        audit("<svg><text>bare & ampersand</text></svg>")
    except ET.ParseError:
        pass
    else:
        raise AssertionError("audit passed a bare ampersand")


def test_audit_flags_a_live_external_reference():
    assert audit('<img src="https://cdn.example/x.png">')["external"]
    assert audit('<div style="background:url(https://cdn/x)">')["external"]
    assert audit("<style>@import url(x.css)</style>")["external"]
    assert not audit('<img src="data:image/png;base64,AAA">')["external"]
    assert not audit('<a href="#top">t</a>')["external"]


def test_audit_reads_a_file_or_a_string():
    with out() as p:
        written = render_compute_graph(STAGES, p)
        assert audit(written)["svg_fragments"] == 1
        assert audit(written.read_text("utf-8"))["svg_fragments"] == 1


def test_renderers_return_the_path_they_wrote():
    with out("nested/deep/page.html") as p:
        result = render_compute_graph(STAGES, p)
    assert result == p


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    sys.exit(1 if failures else 0)
