"""Photo -> 8x13 LED-matrix frame for the UNO Q (ported from qwen-vision-matrix/matrix_sim.py).

A frame is a list of 8 rows of 13 ints, each one of the 6 levels that look different
on the real matrix: 0, 1, 2, 3, 4, 7. On the board the timer ISR PWMs each LED by
level (on 1/23, 1/15, 1/5, 1/3 of the time for 1..4; 5, 6 and 7 are all fully on),
so values are chosen by *perceived* brightness (CIE L*) of those duty cycles.

Only the pixelisation is here (handoff CLAUDE.md section 3). The simulator window,
PNG output and the font are not needed on the board: main.py draws text itself.

The object finder uses U2-Net-p (models/saliency/u2netp.onnx, 4.6 MB, Apache-2.0)
through onnxruntime. Without either, "subject" mode falls back to the colour-region
guess, exactly as on the PC.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

ROWS, COLS = 8, 13
MAX = 7                                              # brightest value draw() accepts
DUTY = (0, 1 / 23, 1 / 15, 1 / 5, 1 / 3, 1, 1, 1)    # on-time per level (from the ISR)
LEVELS = (0, 1, 2, 3, 4, 7)                          # levels that actually look different


def _lstar(y: float) -> float:
    """CIE lightness (0..1) of relative luminance y: how bright it *looks*."""
    return (116 * y ** (1 / 3) - 16) / 100 if y > 0.008856 else 9.033 * y


_LOOK = [_lstar(DUTY[v]) for v in LEVELS]            # perceived brightness per level

# DEBUG: print each decision of the "subject" pixelisation (which path, and why).
DEBUG = True


def _log(msg: str) -> None:
    if DEBUG:
        print("[pixels] " + msg, flush=True)


# ---------------------------------------------------------- image -> frame
def _otsu(values: list[int]) -> int:
    hist = [0] * 256
    for v in values:
        hist[v] += 1
    total, sum_all = len(values), sum(i * h for i, h in enumerate(hist))
    best_t, best_var, w_b, sum_b = 128, -1.0, 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def _nearest_level(x: float) -> int:
    return LEVELS[min(range(len(LEVELS)), key=lambda i: abs(_LOOK[i] - x))]


def _gray(vals: list[int], dither: bool) -> list[int]:
    """Map 0..255 pixels to the 6 real LED levels by perceived brightness.

    With dither, the rounding error is diffused (Floyd-Steinberg) so smooth
    gradients survive the big jump between the top levels (1/3 -> full on).
    """
    x = [v / 255 for v in vals]
    out = [0] * len(x)
    for i in range(len(x)):
        lvl = _nearest_level(x[i])
        out[i] = lvl
        if not dither:
            continue
        err = x[i] - _LOOK[LEVELS.index(lvl)]
        r, c = divmod(i, COLS)
        for dr, dc, w in ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)):
            rr, cc = r + dr, c + dc
            if rr < ROWS and 0 <= cc < COLS:
                x[rr * COLS + cc] += err * w
    return out


_TO_LAB = None


def _lab(rgb: Image.Image) -> list[tuple[int, int, int]] | None:
    """Pixels in CIELAB (Pillow encoding: L 0..255, a/b 0..255 with 128 = neutral).

    Lab distances roughly match how different two colours look. None if this
    Pillow has no littlecms (then "subject" falls back to plain "gray").
    """
    global _TO_LAB
    try:
        from PIL import ImageCms
        if _TO_LAB is None:
            _TO_LAB = ImageCms.buildTransform(ImageCms.createProfile("sRGB"),
                                              ImageCms.createProfile("LAB"), "RGB", "LAB")
        lab = ImageCms.applyTransform(rgb, _TO_LAB)
    except (ImportError, OSError):
        return None
    return list(zip(*(ch.tobytes() for ch in lab.split())))


def _dist(c, ref) -> float:
    return sum((a - b) ** 2 for a, b in zip(c, ref)) ** 0.5


def _median(cs) -> list[int]:
    cs = list(cs)
    return [sorted(c[i] for c in cs)[len(cs) // 2] for i in range(3)]


def _skyline(px, W: int, H: int) -> list[float] | str | None:
    """Landscape check. Returns sky coverage per LED (0..1) for a landscape,
    "gray" for a sky scene whose skyline can't be traced, or None otherwise.

    A landscape has no single subject; what matters is sky vs land. Signs:
    the top edge differs from the bottom edge, a top corner looks like sky
    rather than ground, the lower side edges look like the ground, the top is
    brighter or bluer than the bottom, and its colour is plausible for a sky
    (not a red or brick wall). Then each pixel gets a "skyness" score - bright
    (clouds, haze, overcast) or close to a sky colour - and we walk down each
    column until three non-sky pixels in a row.
    """
    def area(xs, ys):
        return _median(px[y * W + x] for y in ys for x in xs)

    top, bot = area(range(W), range(4)), area(range(W), range(H - 4, H))
    side_lo = area((*range(4), *range(W - 4, W)), range(H // 2, H))
    corners = [area(range(8), range(6)), area(range(W - 8, W), range(6))]
    checks = {
        "top differs from bottom": _dist(top, bot) >= 25,
        "lower sides look like ground": _dist(side_lo, top) - _dist(side_lo, bot) > 10,
        "sky reaches a top corner": any(_dist(c, top) < _dist(c, bot) for c in corners),
        "top brighter or bluer": top[0] - bot[0] > 15 or top[2] < 118,
        "top has a sky-like colour": top[1] <= 140 and top[2] <= 145,
    }
    _log("landscape check (Lab top=%s bottom=%s): %s" % (
        top, bot, ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in checks.items())))
    if not all(checks.values()):
        return None
    Ls = sorted(c[0] for c in px)
    lo, hi = Ls[len(Ls) // 20], Ls[-len(Ls) // 50]
    span = max(1.0, _dist(top, bot))
    # Sky colours: the typical top colour, plus its bluest quarter (when the
    # top median is a white cloud, the blue patches are sky too).
    strip = sorted((px[y * W + x] for y in range(4) for x in range(W)), key=lambda c: c[2])
    refs = [top] + ([_median(strip[:len(strip) // 4])] if strip[len(strip) // 8][2] < 120 else [])
    score = [0.5 * max(0.0, min(1.0, (c[0] - lo) / max(1, hi - lo)))
             + 0.5 * max(0.0, 1 - min(_dist(c, r) for r in refs) / span) for c in px]
    t = _otsu([int(255 * v) for v in score]) / 255
    hz = []
    for x in range(W):
        y, streak = 0, 0
        while y < H:
            streak = streak + 1 if score[y * W + x] < t else 0
            if streak >= 3:
                y -= 2
                break
            y += 1
        hz.append(y)
    hz = [sorted(hz[max(0, i - 3):i + 4])[len(hz[max(0, i - 3):i + 4]) // 2] for i in range(W)]
    _log(f"skyline traced at {100 * sum(hz) / W / H:.0f}% of the height (usable: 20-85%)")
    if not 0.2 * H <= sum(hz) / W <= 0.85 * H:    # skyline stuck at an edge: can't trace it
        return "gray"
    mask = Image.new("L", (W, H))
    mask.frombytes(bytes(255 if y < hz[x] else 0 for y in range(H) for x in range(W)))
    return [v / 255 for v in mask.reduce(8).tobytes()]


def _segment(rgb: Image.Image, k: int = 5, wl: float = 0.35, step: int = 1) -> tuple[list[int], list, int]:
    """Split the picture into k colour regions (shapes). Returns (labels, lab px, k).

    k-means in Lab with lightness down-weighted (wl), so a shadow on an orange
    cat is still "orange cat" and doesn't become its own shape. A majority
    filter removes speckles.
    """
    from PIL import ImageFilter
    W, H = rgb.size
    soft = rgb.filter(ImageFilter.GaussianBlur(1))
    px = _lab(soft)
    feat = [(l * wl, a, b) for l, a, b in px]
    pal = soft.quantize(colors=k, method=Image.Quantize.MEDIANCUT).getpalette()[:3 * k]
    k = len(pal) // 3
    swatch = Image.new("RGB", (k, 1))
    swatch.putdata([tuple(pal[i * 3:i * 3 + 3]) for i in range(k)])
    cent = [(l * wl, a, b) for l, a, b in _lab(swatch)]

    def nearest(f):
        return min(range(k), key=lambda j: (f[0] - cent[j][0]) ** 2
                   + (f[1] - cent[j][1]) ** 2 + (f[2] - cent[j][2]) ** 2)

    sample = [feat[y * W + x] for y in range(0, H, step) for x in range(0, W, step)]
    for _ in range(8):
        sums = [[0.0, 0.0, 0.0, 0] for _ in range(k)]
        for f in sample:
            acc = sums[nearest(f)]
            acc[0] += f[0]; acc[1] += f[1]; acc[2] += f[2]; acc[3] += 1
        cent = [(a[0] / a[3], a[1] / a[3], a[2] / a[3]) if a[3] else c for a, c in zip(sums, cent)]
    lab_img = Image.new("L", (W, H))
    lab_img.frombytes(bytes(40 * nearest(f) for f in feat))
    labels = [v // 40 for v in lab_img.filter(ImageFilter.ModeFilter(5)).tobytes()]
    return labels, px, k


def _shapes_and_shading(rgb: Image.Image, lum: bytes) -> list[float]:
    """Per pixel 0..1: subject regions lit (base level), shading inside them.

    1. Colour regions (_segment).
    2. Background: the region owning most of the border ring, any region close
       to its colour (lightness down-weighted), and any region owning >= 30%
       of the border.
    3. Subject score per region = how central it is x how different its colour
       is from the background. Low score -> background (dark).
    4. Subject regions get a base of 0.82-1.0 by score, then shading: each
       pixel's brightness relative to its own region moves it up or down.
    """
    W, H = rgb.size
    labels, px, K = _segment(rgb)
    ring = [i for i in range(W * H) if min(i % W, i // W, W - 1 - i % W, H - 1 - i // W) < 4]
    share = [0] * K
    for i in ring:
        share[labels[i]] += 1
    members = [[] for _ in range(K)]
    for i, j in enumerate(labels):
        members[j].append(i)
    mean_lab = [tuple(sum(px[i][c] for i in idx) / len(idx) for c in range(3)) if idx else (0, 128, 128)
                for idx in members]
    mean_l = [sum(lum[i] for i in idx) / len(idx) if idx else 0.0 for idx in members]
    std_l = [(sum((lum[i] - mu) ** 2 for i in idx) / len(idx)) ** 0.5 if idx else 1.0
             for idx, mu in zip(members, mean_l)]
    bgj = max(range(K), key=lambda j: share[j])
    bg = mean_lab[bgj]

    def colour_dist(a, b):
        # Same lightness down-weighting as the segmentation: light and dark grass
        # (or sunlit and shaded forest) are the same background colour.
        return ((0.35 * (a[0] - b[0])) ** 2 + (1.5 * (a[1] - b[1])) ** 2
                + (1.5 * (a[2] - b[2])) ** 2) ** 0.5

    dists = [colour_dist(mean_lab[j], bg) for j in range(K)]
    dmax = max(dists) or 1.0
    score = []
    for j, idx in enumerate(members):
        if (not idx or j == bgj or dists[j] < 22          # background, or nearly its colour
                or share[j] / len(ring) >= 0.3):          # owns lots of border: background too
            score.append(0.0)
            continue
        edge = sum(1 for i in idx if min(i % W, i // W, W - 1 - i % W, H - 1 - i // W) < 10) / len(idx)
        central = max(0.0, min(1.0, (0.62 - edge) / 0.4))
        score.append(central * (0.4 + 0.6 * dists[j] / dmax))
    fg = [sc for sc in score if sc >= 0.25] or [1.0]
    smin, smax = min(fg), max(fg)
    base = [0.0 if sc < 0.25 else 0.82 + 0.18 * ((sc - smin) / (smax - smin) if smax > smin else 1.0)
            for sc in score]
    # Not clamped here: the caller averages 8x8 pixels per LED first, then clamps.
    out = []
    for i, j in enumerate(labels):
        z = (lum[i] - mean_l[j]) / (std_l[j] + 12)
        out.append(max(0.0, 0.06 * z) if base[j] == 0 else base[j] + 0.35 * z)
    return out


# ------------------------------------------------- object finder (optional)
SALIENCY_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
_APP_DIR = Path(__file__).resolve().parent.parent
# assets/ if the file was copied onto the board by hand, else data/ (downloaded by the app)
SALIENCY_CANDIDATES = [_APP_DIR / "assets" / "u2netp.onnx", _APP_DIR / "data" / "u2netp.onnx"]
SALIENCY_INPUT = 160      # the model ships at 320x320; we only need 104x64 out
_SAL = None
_SAL_LOCK = threading.Lock()


def saliency_model_path() -> Path | None:
    for p in SALIENCY_CANDIDATES:
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def ensure_saliency_model() -> None:
    """Download U2-Net-p (4.6 MB) into data/ once, if it isn't on the board yet."""
    if saliency_model_path():
        return
    dest = SALIENCY_CANDIDATES[-1]
    tmp = dest.with_suffix(".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("matrix_art: downloading the object model (4.6 MB)...")
        urllib.request.urlretrieve(SALIENCY_URL, tmp)
        tmp.replace(dest)
        print("matrix_art: object model ready")
    except Exception as e:
        print("matrix_art: object model download failed, using the colour-region fallback:", e)
        tmp.unlink(missing_ok=True)


def object_finder_status() -> str:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return "off (onnxruntime not installed)"
    return "on" if saliency_model_path() else "off (u2netp.onnx not on the board yet)"


def _saliency_session():
    global _SAL
    with _SAL_LOCK:
        if _SAL is None or (_SAL is False and saliency_model_path()):
            try:
                import onnxruntime as ort
                path = saliency_model_path()
                if path is None:
                    raise FileNotFoundError("u2netp.onnx")
                data = path.read_bytes()
                # The file fixes its size as `dim { dim_value: 320 }` = 0A 03 08 C0 02.
                # Any size 128..16383 is also a 2-byte varint, so it can be patched in
                # place. U2-Net is fully convolutional: 160 px gives near-identical
                # masks at ~1/3 the time and less than half the memory (~143 MB peak).
                s = SALIENCY_INPUT
                assert 128 <= s < 16384
                data = data.replace(b"\x0a\x03\x08\xc0\x02", b"\x0a\x03\x08" + bytes([0x80 | (s & 0x7F), s >> 7]))
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4                 # UNO Q: quad-core
                # No memory arena: memory goes back after each image instead of being
                # held for the life of the process.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                _SAL = ort.InferenceSession(data, opts, providers=["CPUExecutionProvider"])
            except Exception:                                 # no onnxruntime / no model file
                _SAL = False
        return _SAL or None


def _object_mask(big: Image.Image):
    """numpy (ROWS*8, COLS*8) 0..1: how much each pixel belongs to the main object."""
    sess = _saliency_session()
    if sess is None:
        return None
    import numpy as np
    x = np.asarray(big.resize((SALIENCY_INPUT, SALIENCY_INPUT), Image.LANCZOS), dtype=np.float32)
    x = x / max(1.0, float(x.max()))
    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    out = (out - out.min()) / max(1e-6, float(out.max() - out.min()))
    mask = Image.fromarray((out * 255).astype(np.uint8)).resize((COLS * 8, ROWS * 8), Image.BILINEAR)
    return np.asarray(mask, dtype=np.float32) / 255


def _object_stats(mask) -> tuple[float, float, float]:
    """(area, border, contrast) of the object mask.

    area: fraction of the picture that is object. border: fraction of a 3 px
    edge ring that is object. contrast: mean mask value inside minus outside -
    ~0.95 for a real object, well under 0.85 when the model is guessing.
    """
    import numpy as np
    fg = mask > 0.5
    ring = np.ones_like(fg)
    ring[3:-3, 3:-3] = False
    inside = float(mask[fg].mean()) if fg.any() else 0.0
    outside = float(mask[~fg].mean()) if (~fg).any() else 1.0
    return float(fg.mean()), float(fg[ring].mean()), inside - outside


def _open(a, k: int):
    """Morphological opening (min then max filter, k x k) of a 0..1 numpy array."""
    import numpy as np
    from PIL import ImageFilter
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k)),
                      dtype=np.float32) / 255


def _components(binary) -> list[list[tuple[int, int]]]:
    """4-connected blobs of a boolean numpy array (plain BFS; blobs are few/small)."""
    import numpy as np
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    blobs = []
    for y0, x0 in zip(*np.nonzero(binary)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, blob = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < h and 0 <= xx < w and binary[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        blobs.append(blob)
    return blobs


def _same_colour(lab, mask):
    """Other blobs (>= 10 px) of the main object's hue: the second poppy.

    Same hue (within 20 degrees) and at least half as saturated as the object -
    a lighter or darker red poppy matches, a dull brick wall behind a red car
    or a beige background behind an orange cat doesn't.
    """
    import numpy as np
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    obj = (mask > 0.5) & (chroma > 15)
    out = np.zeros_like(mask)
    if obj.sum() < max(20, 0.3 * (mask > 0.5).sum()):
        return out                                    # mostly grey/white object: skip
    hue0 = np.arctan2(np.median(b[obj]), np.median(a[obj]))
    # If the background itself has that hue (sea around a boat, paper behind a
    # sepia drawing), "same colour" would just mean "background": skip.
    ring = np.ones_like(mask, dtype=bool)
    ring[4:-4, 4:-4] = False
    bga, bgb = np.median(a[ring]), np.median(b[ring])
    bg_hue_diff = abs((np.arctan2(bgb, bga) - hue0 + np.pi) % (2 * np.pi) - np.pi)
    if np.hypot(bga, bgb) > 15 and bg_hue_diff < np.radians(35):
        return out
    dhue = np.abs((np.arctan2(b, a) - hue0 + np.pi) % (2 * np.pi) - np.pi)
    cand = (dhue < np.radians(20)) & (chroma > max(20.0, 0.5 * float(np.median(chroma[obj])))) & (mask < 0.3)
    main = int((mask > 0.5).sum())
    for blob in _components(_open(cand.astype(np.float32), 3) > 0.5):
        if 10 <= len(blob) <= main:                   # a sibling, not a whole backdrop
            ys, xs = zip(*blob)
            out[list(ys), list(xs)] = 1.0
    return out


def _pool_half(v):
    """Per LED: mean of the brighter half of its 64 pixels (thin parts survive)."""
    import numpy as np
    blk = np.sort(v.reshape(ROWS, 8, COLS, 8).transpose(0, 2, 1, 3).reshape(ROWS, COLS, 64), axis=2)
    return np.clip(blk[..., 32:].mean(axis=2), 0, 1)


def _render_object(mask, lum, lab) -> list[float]:
    """Per LED 0..1: object(s) lit and shaded by their own light.

    - faint mask values (< 0.3) are dropped, 0.3..0.7 ramps to full;
    - same-coloured blobs elsewhere count as object too (several poppies);
    - brightness is normalised over the object only (5th..98th percentile), so
      its shadows sit at levels 3-4 and its highlights at full.
    """
    import numpy as np
    sel = np.maximum(np.clip((mask - 0.3) / 0.4, 0, 1), _same_colour(lab, mask))
    inside = lum[sel > 0.5]
    lo, hi = np.percentile(inside, 5), np.percentile(inside, 98)
    shaded = sel * (0.85 + 0.5 * (np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1) - 0.5))
    return [float(x) for x in _pool_half(shaded).ravel()]


def _band(vals: list[float], lo: float, hi: float, from_top: bool) -> list[float]:
    """Spread one block's brightness over [lo, hi], relative to the block itself.

    from_top: the block's brightest parts sit at hi and only clearly darker parts
    drop towards lo (a plain blue sky stays bright instead of turning to noise);
    otherwise the darkest parts sit at lo and brighter parts rise towards hi.
    The span is at least 0.25, so tiny variations stay small.
    """
    if not vals:
        return []
    s = sorted(vals)
    p5, p95 = s[len(s) // 20], s[min(len(s) - 1, len(s) * 95 // 100)]
    span = max(p95 - p5, 0.25)
    if from_top:
        return [hi - (hi - lo) * max(0.0, min(1.0, (p95 - v) / span)) for v in vals]
    return [lo + (hi - lo) * max(0.0, min(1.0, (v - p5) / span)) for v in vals]


def _render_landscape(sky: list[float], lum_bytes: bytes,
                      sky_band=(0.64, 1.0), land_band=(0.0, 0.38)) -> list[float]:
    """Per LED: a strict sky/land split, with brightness detail inside each block.

    Default: sky uses levels 4..7, land levels 0..2. Level 3 is never used inside
    either block, so the skyline stays crisp; it only appears on LEDs that
    straddle it. As a backdrop behind an object the bands are dimmed.
    """
    W = COLS * 8
    blk = [sum(lum_bytes[(r * 8 + y) * W + c * 8 + x] for y in range(8) for x in range(8)) / 64 / 255
           for r in range(ROWS) for c in range(COLS)]
    is_sky = [cv > 0.5 for cv in sky]
    sky_v = iter(_band([b for b, s in zip(blk, is_sky) if s], *sky_band, from_top=True))
    land_v = iter(_band([b for b, s in zip(blk, is_sky) if not s], *land_band, from_top=False))
    # Each LED's own block value; LEDs on the skyline blend both by coverage.
    own = [next(sky_v) if s else next(land_v) for s in is_sky]
    return [cv * (o if s else sky_band[1]) + (1 - cv) * (o if not s else land_band[0])
            for cv, o, s in zip(sky, own, is_sky)]


def _is_terrain(mask, sky: list[float]) -> bool:
    """Is the "object" really part of the land (a dune, a hillside)?

    Yes if almost all of it lies below the skyline and it stretches across most
    of the picture. A boat, a horse on a ridge or a lighthouse doesn't.
    """
    import numpy as np
    obj = (mask > 0.5).reshape(ROWS, 8, COLS, 8).mean(axis=(1, 3))      # object share per LED
    land = 1 - np.array(sky).reshape(ROWS, COLS)
    below = float((obj * land).sum() / max(1e-6, obj.sum()))
    span = float((obj.max(axis=0) > 0.3).mean())                         # columns it reaches
    return below > 0.85 and span >= 0.8


# --------------------------------------------------- knob: object + background
# (ported from qwen-vision-matrix/matrix_sim.py, handoff CLAUDE.md section 3.1 + 3.8)
# The photo is split into two per-LED layers (0..255 each), computed once per photo.
# The knob mixes them, per LED:
#     v = max(object, background * gain / 255)        -> level = LEVEL_LUT[v]
# gain 0 = object only, ~128 = dim backdrop, 255 = full scene (object stays on top).
# With no object the scene is all there is, so the gain runs soft -> full instead
# (140..255) and the frame never goes dark.

LEVEL_LUT = [_nearest_level(v / 255) for v in range(256)]   # 0..255 -> nearest perceived level
NO_OBJECT_MIN_GAIN = 140          # gain floor (of 255) when there is no object layer
KNOB_STEPS = 32                   # knob steps per full cycle
KNOB_START = 6                    # a new photo starts with a dim backdrop


def _scene_layer(lum: bytes, W: int) -> list[float]:
    """Background for non-landscapes: brightness + texture, spread over all levels.

    Per LED: 0.7 x mean brightness + 0.3 x local detail (std of its 64 pixels),
    then half rank-equalised across the 104 LEDs - so a white wall can't flood
    the matrix and a textured scene (wood grain, a sofa, foliage) keeps structure.
    """
    raw = []
    for r in range(ROWS):
        for c in range(COLS):
            px = [lum[(r * 8 + y) * W + c * 8 + x] / 255 for y in range(8) for x in range(8)]
            mu = sum(px) / 64
            sd = (sum((p - mu) ** 2 for p in px) / 64) ** 0.5
            raw.append(0.7 * mu + 0.3 * min(1.0, 3 * sd))
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    eq = [0.0] * len(raw)
    for rank, i in enumerate(order):
        eq[i] = rank / (len(raw) - 1)
    return [0.5 * e + 0.5 * s for e, s in zip(eq, _band(raw, 0.0, 1.0, from_top=False))]


def image_layers(img) -> tuple[list[int] | None, list[int], str]:
    """(object layer or None, background layer, kind) - each 104 values 0..255.

    background: landscapes -> the full sky/land rendering (sky 4..7, land 0..2);
                otherwise  -> the photo's own brightness per LED, stretched 0..1.
    object:     landscape  -> a confident object that isn't terrain, else None;
                otherwise  -> the object (confident or unsure mask), or the
                              colour-region guess when no object was found.
    kind: "landscape+object", "landscape", "object", "regions" or "scene".
    """
    import time
    t_start = time.monotonic()
    if not isinstance(img, Image.Image):
        img = Image.open(img)
        img.draft("RGB", (512, 512))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size
    big = ImageOps.fit(img, (COLS * 32, ROWS * 32), Image.LANCZOS)  # crop to 13:8
    img = big.resize((COLS * 8, ROWS * 8), Image.LANCZOS)
    _log(f"layers: {w0}x{h0} centre-cropped to 13:8 -> {big.width}x{big.height} (object model) "
         f"and {img.width}x{img.height} (8x8 pixels per LED)")
    W, H = img.size
    lum_img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    lum = lum_img.tobytes()
    px = _lab(img)
    if px is None:
        _log("no colour management (littlecms) in Pillow -> no landscape check, no object finder")
    sky = _skyline(px, W, H) if px is not None else None
    landscape = isinstance(sky, list)
    _log("landscape: " + ("no" if sky is None else "sky scene, skyline not traceable" if sky == "gray"
                           else f"yes, sky covers {100 * sum(sky) / len(sky):.0f}% of the LEDs"))
    bg = _render_landscape(sky, lum) if landscape else _scene_layer(lum, W)
    _log("background layer: " + ("sky at 4-7 over land at 0-2" if landscape
                                 else "brightness + texture, equalised over all levels"))
    obj, kind = None, "landscape" if landscape else "scene"
    if sky != "gray" and px is not None:
        t_obj = time.monotonic()
        mask = _object_mask(big)
        if mask is None:
            _log("object finder unavailable (onnxruntime or u2netp.onnx missing)")
        else:
            area, border, contrast = _object_stats(mask)
            confident = contrast >= 0.85 and 0.03 <= area <= 0.75 and border <= 0.6
            usable = confident or (contrast >= 0.55 and 0.015 <= area <= 0.75 and border <= 0.6)
            terrain = landscape and confident and _is_terrain(mask, sky)
            _log(f"object finder ({time.monotonic() - t_obj:.1f}s): area={area:.2f} (need 0.03-0.75), "
                 f"border={border:.2f} (need <=0.6), contrast={contrast:.2f} (confident >=0.85, "
                 f"usable >=0.55) -> confident={confident}, usable={usable}"
                 + (", part of the terrain" if terrain else ""))
            if (confident and not terrain) or (usable and not landscape):
                import numpy as np
                obj = _render_object(mask, np.asarray(lum_img, np.float32) / 255,
                                     np.array(px, np.float32).reshape(H, W, 3))
                kind = "landscape+object" if landscape else "object"
        if obj is None and not landscape:
            _log("no usable object: colour-region guess (k-means on Lab colours)")
            per_px = _shapes_and_shading(img, lum)
            obj = [max(0.0, min(1.0, sum(per_px[(r * 8 + y) * W + c * 8 + x]
                                         for y in range(8) for x in range(8)) / 64))
                   for r in range(ROWS) for c in range(COLS)]
            kind = "regions"
            if sum(v >= 0.3 for v in obj) < 3:             # found nothing: scene only
                _log("colour regions found nothing -> scene only")
                obj, kind = None, "scene"
    to8 = lambda vs: [max(0, min(255, round(255 * v))) for v in vs]
    _log(f"layers ready: kind={kind}, object layer={'yes' if obj is not None else 'none'} "
         f"({time.monotonic() - t_start:.1f}s total)")
    return (to8(obj) if obj is not None else None), to8(bg), kind


def knob_state(step: int) -> tuple[int, int, bool]:
    """Knob position -> (background gain 0..255, inverted columns 0..13, wipe_from_right).

    The cycle, turning clockwise (counter-clockwise runs it backwards):
      0-15  positive, background 0 -> 255   (object only -> full scene)
      16-19 negative sweeps in, left to right
      20-27 negative, background 255 -> 0   (negative scene -> negative object)
      28-31 positive sweeps back in, left to right -> step 0 again
    """
    s = step % KNOB_STEPS
    if s <= 15:
        return s * 255 // 15, 0, False
    if s <= 19:
        return 255, (s - 15) * COLS // 5, False
    if s <= 27:
        return (27 - s) * 255 // 8, COLS, False
    return 0, COLS - (s - 27) * COLS // 5, True


def knob_zone(step: int) -> str:
    """Human-readable knob position, for the logs."""
    s = step % KNOB_STEPS
    gain, inv, _ = knob_state(s)
    pct = round(100 * gain / 255)
    if s <= 15:
        return f"background {pct}%"
    if s <= 19:
        return f"turning negative ({inv}/{COLS} columns)"
    if s <= 27:
        return f"negative, background {pct}%"
    return f"turning positive ({COLS - inv}/{COLS} columns)"


def knob_frame(obj: list[int] | None, bg: list[int], step: int) -> list[list[int]]:
    """The frame for a knob position: layers mixed + the negative sweep. Integer-only."""
    gain, inv, from_right = knob_state(step)
    if obj is None:                                   # no object: never fade to black
        gain = NO_OBJECT_MIN_GAIN + gain * (255 - NO_OBJECT_MIN_GAIN) // 255
    o = obj or [0] * (ROWS * COLS)
    out = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            i = r * COLS + c
            v = max(o[i], bg[i] * gain // 255)
            negative = (c >= COLS - inv) if from_right else (c < inv)
            row.append(LEVEL_LUT[255 - v if negative else v])
        out.append(row)
    return out


MODES = ["subject", "gray", "graydither", "threshold", "dither", "edges"]


def image_to_frame(img: Image.Image | str | Path, mode: str = "subject",
                   invert: bool = False) -> list[list[int]]:
    """Squash an image into an 8x13 frame of LED levels (0, 1, 2, 3, 4 or 7).

    mode "subject" (default) decides per picture:
      - landscape + object: the landscape as a dim backdrop (sky 2-3, land 0-1)
        with a confident object on top at 4-7;
      - landscape only: lit sky (4-7) over a dark skyline (0-2);
      - object only: the main object (U2-Net-p mask) plus same-coloured
        siblings, lit and shaded by its own light, background dark;
      - neither: the colour-region guess (also used without onnxruntime).
    Other modes: "gray", "graydither", "threshold", "dither", "edges".
    """
    import time
    t_start = time.monotonic()
    if not isinstance(img, Image.Image):
        img = Image.open(img)
        img.draft("RGB", (512, 512))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size
    big = ImageOps.fit(img, (COLS * 32, ROWS * 32), Image.LANCZOS)  # crop to 13:8
    img = big.resize((COLS * 8, ROWS * 8), Image.LANCZOS)
    _log(f"mode={mode}: {w0}x{h0} centre-cropped to 13:8 -> {big.width}x{big.height} (object model) "
         f"and {img.width}x{img.height} (8x8 pixels per LED)")
    if mode == "subject":
        W, H = img.size
        lum_img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
        lum = lum_img.tobytes()
        px = _lab(img)
        if px is None:
            _log("no colour management (littlecms) in Pillow -> plain brightness")
        sky = _skyline(px, W, H) if px is not None else None
        _log("landscape: " + ("no" if sky is None else "sky scene, skyline not traceable" if sky == "gray"
                               else f"yes, sky covers {100 * sum(sky) / len(sky):.0f}% of the LEDs"))
        done = False
        if sky == "gray" or px is None:           # sky scene without a skyline
            _log("PATH: plain brightness ('gray')")
            return image_to_frame(img, "gray", invert)
        t_obj = time.monotonic()
        mask = _object_mask(big)
        obj_ok = confident = False
        if mask is None:
            _log("object finder unavailable (onnxruntime or u2netp.onnx missing)")
        else:
            area, border, contrast = _object_stats(mask)
            confident = contrast >= 0.85 and 0.03 <= area <= 0.75 and border <= 0.6
            # An unsure but compact mask (a sailboat, a teapot) still beats
            # guessing from colours.
            obj_ok = confident or (contrast >= 0.55 and 0.015 <= area <= 0.75 and border <= 0.6)
            _log(f"object finder ({time.monotonic() - t_obj:.1f}s): area={area:.2f} (need 0.03-0.75), "
                 f"border={border:.2f} (need <=0.6), contrast={contrast:.2f} (confident >=0.85, usable >=0.55) "
                 f"-> confident={confident}, usable={obj_ok}")
        if obj_ok:
            import numpy as np
            obj_vals = _render_object(mask, np.asarray(lum_img, np.float32) / 255,
                                      np.array(px, np.float32).reshape(H, W, 3))
        if isinstance(sky, list):
            terrain = confident and _is_terrain(mask, sky)
            if confident and not terrain:
                # landscape + object: dim scene behind, object on top
                _log("PATH: landscape + object (dim sky 2-3 / land 0-1 backdrop, object on top at 4-7)")
                back = _render_landscape(sky, lum, sky_band=(0.31, 0.52), land_band=(0.0, 0.15))
                vals = [max(b_, o) for b_, o in zip(back, obj_vals)]
            else:                                 # landscape only
                why = "the object is part of the terrain" if terrain else "no confident object"
                _log(f"PATH: landscape only ({why}): sky at levels 4-7, land at 0-2")
                vals = _render_landscape(sky, lum)
            done = True
        elif obj_ok:                              # object only
            _log("PATH: object only (object lit and shaded by its own light, background dark)")
            vals = obj_vals
            done = True
        if not done:                              # neither: colour-region guess
            _log("PATH: colour regions (no landscape, no usable object): k-means on Lab colours")
            per_px = _shapes_and_shading(img, lum)
            vals = [max(0.0, min(1.0, sum(per_px[(r * 8 + y) * W + c * 8 + x]
                                          for y in range(8) for x in range(8)) / 64))
                    for r in range(ROWS) for c in range(COLS)]
        # Never show an empty matrix: if a path found (almost) nothing to light,
        # fall back to plain brightness, which always shows the photo's structure.
        vals = [(1 - v) if invert else v for v in vals]
        lit = sum(v >= 0.3 for v in vals)
        if lit < 3:
            _log(f"only {lit} LEDs lit -> fallback PATH: plain brightness ('gray')")
            return image_to_frame(img, "gray", invert)
        px = _gray([255 * v for v in vals], dither=False)
        _log(f"quantised to perceived-brightness levels: {lit} LEDs >= 30% brightness, "
             f"histogram {{{', '.join(f'{lv}: {px.count(lv)}' for lv in LEVELS)}}} "
             f"({time.monotonic() - t_start:.1f}s total)")
        return [px[r * COLS:(r + 1) * COLS] for r in range(ROWS)]
    g = ImageOps.autocontrast(img.convert("L"), cutoff=2)
    if mode == "edges":
        from PIL import ImageFilter
        g = ImageOps.autocontrast(g.filter(ImageFilter.FIND_EDGES), cutoff=1)
    small = g.resize((COLS, ROWS), Image.LANCZOS)
    small = ImageOps.autocontrast(small, cutoff=1)
    if invert:
        small = ImageOps.invert(small)
    if mode in ("gray", "graydither"):
        px = _gray(list(small.tobytes()), dither=mode == "graydither")
    elif mode == "dither":
        bw = small.convert("1").convert("L")                      # FS dither
        px = [MAX if v else 0 for v in bw.tobytes()]
    else:
        vals = list(small.tobytes())
        t = _otsu(vals)
        px = [MAX if v > t else 0 for v in vals]
    return [px[r * COLS:(r + 1) * COLS] for r in range(ROWS)]


_ASCII = ".-:=+###"


def frame_to_ascii(frame) -> str:
    """For the logs: one character per LED, darkest '.' to brightest '#'."""
    return "\n".join("".join(_ASCII[v] for v in row) for row in frame)
