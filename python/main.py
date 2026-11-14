# UniEye - MPU side
#
# Knob controls (events come from the sketch):
#   MENU     turn = choose what to generate, click = capture, long = history
#   PLAYBACK click = skip back to menu
#   HISTORY  turn = switch saved output, click / long = back to menu

import queue
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Bridge, Frame

import matrix_art   # photo -> 8x13 LED levels (handoff: matrix_sim.image_to_frame)
import muse         # photo -> text with Qwen + vision (handoff: vlm.subject + vlm.micro)

# The arduino:llm brick in app.yaml starts the llama.cpp server with Qwen + its vision
# projector; muse.py talks to it directly (grammars, sampling, thinking off).

ROWS, COLS = 8, 13
SCROLL_FPS = 14          # running text speed (columns per second)
IMAGE_SECONDS = 4.0      # how long the thumbnail is shown
CAPTURE_RES = (640, 480)
MAX_TEXT_CHARS = 320
# DEBUG: keep the latest photo at full capture resolution as data/last_capture.jpg
# (overwritten on every capture). Set to False to discard photos again.
SAVE_DEBUG_PHOTO = True

# One mode: the handoff's micro line ("Balloon — waiting for the moon"). Its name is
# only stored with saved outputs; it is not shown and not given to Qwen.
MODE_NAME = "MUSE"

# DEBUG: print every step the app takes (menu, capture, processing, display, history).
# matrix_art.DEBUG and muse.DEBUG switch the thumbnail and Qwen details on/off separately.
DEBUG = True
_step_t0 = time.monotonic()


def step(msg, restart=False):
    """Debug line with the time since the current capture (or app start) began."""
    global _step_t0
    if restart:
        _step_t0 = time.monotonic()
    if DEBUG:
        print(f"[app +{time.monotonic() - _step_t0:6.1f}s] {msg}", flush=True)


# Object finder for the thumbnail: fetch U2-Net-p once in the background if it's missing.
threading.Thread(target=matrix_art.ensure_saliency_model, daemon=True).start()
# Warm-up: load Qwen + vision while the menu is showing, not on the first click.
threading.Thread(target=muse.warm_up, daemon=True).start()
print("vision model:", muse.MODEL, "| object finder:", matrix_art.object_finder_status())

# ---------------------------------------------------------------- storage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
db = sqlite3.connect(DATA_DIR / "history.db", check_same_thread=False)
db.execute(
    "CREATE TABLE IF NOT EXISTS outputs ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, mode TEXT, "
    "temperature REAL, text TEXT, pixels BLOB)"
)
db.commit()


def save_output(mode, temperature, text, pixels):
    db.execute(
        "INSERT INTO outputs (created, mode, temperature, text, pixels) VALUES (?, ?, ?, ?, ?)",
        (time.time(), mode, temperature, text, bytes(pixels.flatten().tolist())),
    )
    db.commit()


def load_outputs():
    """Newest first."""
    rows = db.execute("SELECT mode, temperature, text, pixels FROM outputs ORDER BY id DESC").fetchall()
    return [
        {
            "mode": m,
            "temperature": t,
            "text": txt,
            "pixels": np.frombuffer(px, dtype=np.uint8).reshape(ROWS, COLS),
        }
        for m, t, txt, px in rows
    ]


# ---------------------------------------------------------------- knob events

events = queue.Queue()
Bridge.provide("knob_turn", lambda delta: events.put(("turn", int(delta))))
Bridge.provide("knob_click", lambda *_: events.put(("click", 0)))
Bridge.provide("knob_long", lambda *_: events.put(("long", 0)))


def wait_event(timeout):
    try:
        return events.get(timeout=max(timeout, 0))
    except queue.Empty:
        return None


def clear_events():
    while not events.empty():
        events.get_nowait()


# ---------------------------------------------------------------- MCU helpers

def show(pixels):
    Bridge.call("draw", Frame(np.asarray(pixels, dtype=np.uint8)).to_board_bytes())


def beep(freq, ms):
    try:
        Bridge.call("beep", int(freq), int(ms))
    except Exception as e:
        print("beep failed:", e)


def read_temperature():
    try:
        return float(Bridge.call("get_temperature"))
    except Exception as e:
        print("temperature read failed:", e)
        return float("nan")


# ---------------------------------------------------------------- 5x7 font

# Classic 5x7 font, column-major, bit 0 = top row. Index 0 = ASCII 32.
FONT = [
    0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x5F,0x00,0x00, 0x00,0x07,0x00,0x07,0x00, 0x14,0x7F,0x14,0x7F,0x14,
    0x24,0x2A,0x7F,0x2A,0x12, 0x23,0x13,0x08,0x64,0x62, 0x36,0x49,0x55,0x22,0x50, 0x00,0x05,0x03,0x00,0x00,
    0x00,0x1C,0x22,0x41,0x00, 0x00,0x41,0x22,0x1C,0x00, 0x08,0x2A,0x1C,0x2A,0x08, 0x08,0x08,0x3E,0x08,0x08,
    0x00,0x50,0x30,0x00,0x00, 0x08,0x08,0x08,0x08,0x08, 0x00,0x60,0x60,0x00,0x00, 0x20,0x10,0x08,0x04,0x02,
    0x3E,0x51,0x49,0x45,0x3E, 0x00,0x42,0x7F,0x40,0x00, 0x42,0x61,0x51,0x49,0x46, 0x21,0x41,0x45,0x4B,0x31,
    0x18,0x14,0x12,0x7F,0x10, 0x27,0x45,0x45,0x45,0x39, 0x3C,0x4A,0x49,0x49,0x30, 0x01,0x71,0x09,0x05,0x03,
    0x36,0x49,0x49,0x49,0x36, 0x06,0x49,0x49,0x29,0x1E, 0x00,0x36,0x36,0x00,0x00, 0x00,0x56,0x36,0x00,0x00,
    0x08,0x14,0x22,0x41,0x00, 0x14,0x14,0x14,0x14,0x14, 0x00,0x41,0x22,0x14,0x08, 0x02,0x01,0x51,0x09,0x06,
    0x32,0x49,0x79,0x41,0x3E, 0x7E,0x11,0x11,0x11,0x7E, 0x7F,0x49,0x49,0x49,0x36, 0x3E,0x41,0x41,0x41,0x22,
    0x7F,0x41,0x41,0x22,0x1C, 0x7F,0x49,0x49,0x49,0x41, 0x7F,0x09,0x09,0x01,0x01, 0x3E,0x41,0x41,0x51,0x32,
    0x7F,0x08,0x08,0x08,0x7F, 0x00,0x41,0x7F,0x41,0x00, 0x20,0x40,0x41,0x3F,0x01, 0x7F,0x08,0x14,0x22,0x41,
    0x7F,0x40,0x40,0x40,0x40, 0x7F,0x02,0x04,0x02,0x7F, 0x7F,0x04,0x08,0x10,0x7F, 0x3E,0x41,0x41,0x41,0x3E,
    0x7F,0x09,0x09,0x09,0x06, 0x3E,0x41,0x51,0x21,0x5E, 0x7F,0x09,0x19,0x29,0x46, 0x46,0x49,0x49,0x49,0x31,
    0x01,0x01,0x7F,0x01,0x01, 0x3F,0x40,0x40,0x40,0x3F, 0x1F,0x20,0x40,0x20,0x1F, 0x7F,0x20,0x18,0x20,0x7F,
    0x63,0x14,0x08,0x14,0x63, 0x03,0x04,0x78,0x04,0x03, 0x61,0x51,0x49,0x45,0x43, 0x00,0x7F,0x41,0x41,0x00,
    0x02,0x04,0x08,0x10,0x20, 0x00,0x41,0x41,0x7F,0x00, 0x04,0x02,0x01,0x02,0x04, 0x40,0x40,0x40,0x40,0x40,
    0x00,0x01,0x02,0x04,0x00, 0x20,0x54,0x54,0x54,0x78, 0x7F,0x48,0x44,0x44,0x38, 0x38,0x44,0x44,0x44,0x20,
    0x38,0x44,0x44,0x48,0x7F, 0x38,0x54,0x54,0x54,0x18, 0x08,0x7E,0x09,0x01,0x02, 0x08,0x14,0x54,0x54,0x3C,
    0x7F,0x08,0x04,0x04,0x78, 0x00,0x44,0x7D,0x40,0x00, 0x20,0x40,0x44,0x3D,0x00, 0x00,0x7F,0x10,0x28,0x44,
    0x00,0x41,0x7F,0x40,0x00, 0x7C,0x04,0x18,0x04,0x78, 0x7C,0x08,0x04,0x04,0x78, 0x38,0x44,0x44,0x44,0x38,
    0x7C,0x14,0x14,0x14,0x08, 0x08,0x14,0x14,0x18,0x7C, 0x7C,0x08,0x04,0x04,0x08, 0x48,0x54,0x54,0x54,0x20,
    0x04,0x3F,0x44,0x40,0x20, 0x3C,0x40,0x40,0x20,0x7C, 0x1C,0x20,0x40,0x20,0x1C, 0x3C,0x40,0x30,0x40,0x3C,
    0x44,0x28,0x10,0x28,0x44, 0x0C,0x50,0x50,0x50,0x3C, 0x44,0x64,0x54,0x4C,0x44, 0x00,0x08,0x36,0x41,0x00,
    0x00,0x00,0x7F,0x00,0x00, 0x00,0x41,0x36,0x08,0x00, 0x02,0x01,0x02,0x04,0x02,
]
DEGREE = [0x00, 0x06, 0x09, 0x09, 0x06]
# Text is drawn one pixel higher than before (glyph rows 0-6 on matrix rows 0-6), so the
# bottom row (bit 7) is free for descenders: these letters use it instead of being squished.
DESCENDERS = {
    "g": [0x98, 0xA4, 0xA4, 0xA4, 0x7C],
    "j": [0x00, 0x80, 0x84, 0x7D, 0x00],
    "p": [0xFC, 0x24, 0x24, 0x24, 0x18],
    "q": [0x18, 0x24, 0x24, 0x24, 0xFC],
    "y": [0x1C, 0xA0, 0xA0, 0xA0, 0x7C],
    ",": [0x00, 0x80, 0x60, 0x00, 0x00],
    ";": [0x00, 0x80, 0x6C, 0x00, 0x00],
}
EM_DASH = [0x08] * 4          # the dash between subject and line: short, 4 columns
DASH_GAP = 1                  # a space next to the dash is drawn as just 1 blank column


def glyph(ch):
    if ch == "°":
        return DEGREE
    if ch == "—":
        return EM_DASH
    if ch in DESCENDERS:
        return DESCENDERS[ch]
    code = ord(ch)
    if not 32 <= code <= 126:
        code = ord("?")
    i = (code - 32) * 5
    return FONT[i:i + 5]


def render_text(text, brightness=7):
    """Text -> 8 x N column bitmap (one blank column between glyphs). Glyph bit r is
    matrix row r: letters sit on rows 0-6, descenders reach row 7."""
    cols = []
    for i, ch in enumerate(text):
        if ch == " " and "—" in text[max(0, i - 1):i + 2]:
            cols += [[0] * ROWS] * DASH_GAP          # tight spacing around the dash
            continue
        for bits in glyph(ch):
            cols.append([brightness if (bits >> r) & 1 else 0 for r in range(ROWS)])
        cols.append([0] * ROWS)
    bmp = np.zeros((ROWS, max(len(cols), 1)), dtype=np.uint8)
    if cols:
        bmp[:, :] = np.array(cols, dtype=np.uint8).T
    return bmp


def clean_text(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)  # thinking without an opening tag
    text = text.replace("<think>", "")
    text = re.sub(r"[*_#`>]+", "", text)                    # markdown leftovers
    # ASCII for the matrix font, except the characters it has extra glyphs for
    text = "".join(ch if ch in "—°" else
                   unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TEXT_CHARS:
        cut = text[:MAX_TEXT_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end > 80 else cut.rstrip() + "..."
    return text or "(no answer)"


# ---------------------------------------------------------------- display routines
# Each routine returns the knob event that interrupted it (or None when it finished).
# `accept` limits which event kinds interrupt; others are ignored.

def scroll_text(text, accept=("turn", "click", "long"), from_left=False, loop=False):
    bmp = render_text(text)
    width = bmp.shape[1]
    padded = np.zeros((ROWS, width + 2 * COLS), dtype=np.uint8)
    padded[:, COLS:COLS + width] = bmp
    start = COLS if from_left else 0
    while True:
        for x in range(start, width + COLS + 1):
            show(padded[:, x:x + COLS])
            ev = sleep_for(1.0 / SCROLL_FPS, accept)
            if ev:
                return ev
        if not loop:
            return None
        start = 0


def hold(pixels, seconds, accept=("turn", "click", "long")):
    show(pixels)
    return sleep_for(seconds, accept)


def sleep_for(seconds, accept):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ev = wait_event(remaining)
        if ev and ev[0] in accept:
            return ev


def spinner_frame(step):
    """A dot (with a fading tail) running around the matrix border."""
    border = [(0, c) for c in range(COLS)] + [(r, COLS - 1) for r in range(1, ROWS)] + \
             [(ROWS - 1, c) for c in range(COLS - 2, -1, -1)] + [(r, 0) for r in range(ROWS - 2, 0, -1)]
    px = np.zeros((ROWS, COLS), dtype=np.uint8)
    for tail, level in enumerate((7, 5, 3, 1)):
        r, c = border[(step - tail) % len(border)]
        px[r, c] = level
    return px


def format_temp(t):
    return "--°C" if t is None or np.isnan(t) else f"{t:.1f}°C"


def show_output(entry, accept):
    """Thumbnail -> running text -> temperature."""
    return (
        hold(entry["pixels"], IMAGE_SECONDS, accept)
        or scroll_text(entry["text"], accept)
        or scroll_text(format_temp(entry["temperature"]), accept)
    )


# ---------------------------------------------------------------- image processing

def to_photo(frame):
    """OpenCV camera frame (BGR numpy) -> RGB PIL image for the pixeliser and the model."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(np.ascontiguousarray(arr[..., 2::-1]))



def capture_frame():
    camera = Camera(resolution=CAPTURE_RES)
    camera.start()
    try:
        frame, got = None, 0
        for _ in range(6):          # let auto-exposure settle, keep the latest frame
            f = camera.capture()
            if f is not None:
                frame = f
                got += 1
            time.sleep(0.05)
        shape = "none" if frame is None else "x".join(map(str, np.asarray(frame).shape))
        step(f"camera: {got}/6 frames read (first ones let auto-exposure settle), kept the last: {shape}")
        return frame
    finally:
        camera.stop()


# ---------------------------------------------------------------- states

# The start screen: a crescent moon (static) with twinkling stars around it.
START_PICTURE = np.array([
    0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 0,
    0, 7, 0, 0, 7, 3, 0, 0, 0, 0, 0, 7, 0,
    7, 0, 0, 4, 7, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 4, 7, 7, 2, 0, 0, 4, 0, 0, 7,
    0, 7, 0, 2, 4, 7, 7, 7, 4, 0, 0, 7, 0,
    0, 0, 0, 0, 0, 3, 4, 4, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,
], dtype=np.uint8).reshape(ROWS, COLS)
MOON_ROWS, MOON_COLS = range(1, 7), range(3, 9)      # the crescent; everything else lit is a star
STARS = [(r, c) for r in range(ROWS) for c in range(COLS)
         if START_PICTURE[r, c] and not (r in MOON_ROWS and c in MOON_COLS)]
# One twinkle, as a share of the star's own brightness: mostly lit, a quick dim and back.
TWINKLE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.3, 0.15, 0.3, 0.6, 1.0]
TWINKLE_S = 0.12             # seconds per animation frame
STAR_PHASE = [(i * 5) % len(TWINKLE) for i in range(len(STARS))]   # fixed, spread-out offsets


def start_frame(t):
    """Frame `t` of the start animation: the picture with each star at its twinkle step."""
    frame = START_PICTURE.copy()
    for (r, c), ph in zip(STARS, STAR_PHASE):
        share = TWINKLE[(t + ph) % len(TWINKLE)]
        if share < 1.0:                     # fully lit stars keep their own level exactly
            level = START_PICTURE[r, c] * share
            frame[r, c] = matrix_art.LEVEL_LUT[int(round(level / 7 * 255))]
    return frame


def menu():
    """The start state: the moon and twinkling stars; click = take a photo, hold = history."""
    step(f"START: moon with {len(STARS)} twinkling stars (click = take a photo, hold = saved outputs)",
         restart=True)
    t = 0
    while True:
        show(start_frame(t))
        t += 1
        ev = wait_event(TWINKLE_S)
        if ev is None:
            continue
        kind, delta = ev
        if kind == "click":
            step("knob click -> capture")
            return "capture"
        if kind == "long":
            step("knob held -> history")
            return "history"
        step(f"knob turned {delta:+d}: nothing to choose in the start state")


def capture_and_generate():
    name = MODE_NAME
    step("===== CAPTURE =====", restart=True)
    step("shutter beep + full-bright matrix as a flash")
    beep(1500, 60)
    show(np.full((ROWS, COLS), 7, dtype=np.uint8))  # "flash"

    try:
        frame = capture_frame()
    except Exception as e:
        print("camera error:", e)
        frame = None
    if frame is None:
        step("no frame from the camera -> CAMERA ERROR")
        beep(300, 400)
        scroll_text("CAMERA ERROR", accept=())
        return

    beep(2400, 40); time.sleep(0.05); beep(3000, 50)    # "picture taken"
    step("picture taken: two rising beeps")
    temperature = read_temperature()
    step(f"temperature from Modulino Thermo: {format_temp(temperature)}")
    photo = to_photo(frame)
    step(f"photo converted BGR -> RGB: {photo.width}x{photo.height}")
    del frame                       # original image is never stored (except the debug copy below)
    if SAVE_DEBUG_PHOTO:
        try:
            photo.save(DATA_DIR / "last_capture.jpg", "JPEG", quality=95)
            print(f"debug photo saved: data/last_capture.jpg ({photo.width}x{photo.height})")
        except Exception as e:
            print("debug photo save failed:", e)
    result = {}

    def layers_worker():
        t0 = time.monotonic()
        try:
            result["layers"] = matrix_art.image_layers(photo)
        except Exception as e:                       # never lose the thumbnail
            print("layer pixelisation failed, using plain brightness:", e)
            flat = [v for row in matrix_art.image_to_frame(photo, "gray") for v in row]
            result["layers"] = (None, [round(255 * matrix_art._LOOK[matrix_art.LEVELS.index(v)])
                                       for v in flat], "gray")
        result["pixel_s"] = time.monotonic() - t0
        step(f"thumbnail worker: layers done in {result['pixel_s']:.1f}s ({result['layers'][2]})")

    def worker():
        # The thumbnail layers run in parallel with Qwen: ~4 s vs ~25 s, so they're
        # ready long before the text and the user can adjust the thumbnail meanwhile.
        step("worker: 1) thumbnail layers started in parallel; 2) photo -> line with Qwen "
             f"(subject, then up to {muse.MICRO_WORDS} words)")
        thumb = threading.Thread(target=layers_worker, daemon=True)
        thumb.start()
        image_url = muse.image_data_url(photo)      # encoded once, shared by every call

        def partial(text_so_far):                   # the line while Qwen is writing it
            result["partial"] = text_so_far

        try:
            result["text"], result["subject"] = muse.micro(image_url, on_partial=partial)
            step("worker: text done")
        except Exception as e:
            step(f"worker: Qwen failed: {e}")
            result["error"] = e
        thumb.join()

    started = time.monotonic()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    clear_events()

    # 1) loading animation until the thumbnail layers are ready
    step("matrix: 'thinking' animation (dot running round the border) until the thumbnail is ready")
    spin = 0
    while "layers" not in result and t.is_alive():
        show(spinner_frame(spin))
        spin += 1
        t.join(0.08)
    clear_events()                  # knob input given before the thumbnail existed is ignored

    # 2) adjust the thumbnail with the knob until it's clicked
    obj, bg, kind = result.get("layers") or (None, [0] * (ROWS * COLS), "none")
    knob = matrix_art.KNOB_START
    pixels = np.array(matrix_art.knob_frame(obj, bg, knob), dtype=np.uint8)
    show(pixels)
    step(f"ADJUST: thumbnail ready ({kind}); turn = adjust object/background, click = done "
         f"(knob {knob}: {matrix_art.knob_zone(knob)})")
    while True:
        ev = wait_event(0.1)
        if ev is None:
            continue
        kind_ev, delta = ev
        if kind_ev == "turn":
            knob = (knob + delta) % matrix_art.KNOB_STEPS
            pixels = np.array(matrix_art.knob_frame(obj, bg, knob), dtype=np.uint8)
            show(pixels)
            step(f"adjust: knob {delta:+d} -> {knob}/{matrix_art.KNOB_STEPS}: {matrix_art.knob_zone(knob)}")
        elif kind_ev == "click":
            step(f"adjust: click -> thumbnail fixed at knob {knob} ({matrix_art.knob_zone(knob)}); "
                 + ("text already ready" if not t.is_alive() else
                    f"text so far: {result.get('partial', '')!r}"))
            beep(1500, 30)
            break
    clear_events()
    used = sorted({int(v) for v in pixels.flatten()})
    print(f"thumbnail ({result.get('pixel_s', 0):.1f}s, {kind}, knob {knob}): {len(used)} of "
          f"{len(matrix_art.LEVELS)} visible brightness levels used {used}\n"
          + matrix_art.frame_to_ascii(pixels.tolist()))

    # 3) slide to the line - shown while Qwen is still writing it - and scroll the
    #    strip with the knob; it is saved once the line is finished
    live_output(result, t, pixels, temperature, name, started)
    del photo                       # the thumbnail and the text are all that's kept


# The output is one strip, scrolled with the knob:
#     [thumbnail] gap [line, growing while Qwen writes it] gap [temperature, once finished]
SCROLL_COLS_PER_STEP = 3     # matrix columns moved per knob step (a letter is 6 columns)
GAP = 6                      # blank columns between the parts
SLIDE_S = 0.03               # seconds per column of the automatic slide to the line


def live_output(result, worker, pixels, temperature, name, started):
    thumb = np.asarray(pixels, dtype=np.uint8)
    text_at = COLS + GAP

    def build():
        done = not worker.is_alive()
        if done and "error" not in result:
            text = clean_text(result["text"])
        else:
            text = clean_text(result.get("partial", "")) if result.get("partial") else ""
        parts = [thumb, np.zeros((ROWS, GAP), np.uint8)]
        if text:
            parts.append(render_text(text))
        temp_at = None
        if done and "error" not in result:
            temp_at = sum(p.shape[1] for p in parts) + GAP
            parts += [np.zeros((ROWS, GAP), np.uint8), render_text(format_temp(temperature))]
        strip = np.concatenate(parts, axis=1)
        if strip.shape[1] < text_at + COLS:          # room to slide the thumbnail fully aside
            strip = np.concatenate([strip, np.zeros((ROWS, text_at + COLS - strip.shape[1]), np.uint8)], axis=1)
        return strip, text, done, temp_at

    def where(p, temp_at):
        if p < text_at - GAP:
            return "thumbnail"
        return "temperature" if temp_at is not None and p + COLS > temp_at else "line"

    # wait for the first part of the line (the subject) before sliding away
    if not result.get("partial") and worker.is_alive():
        step("output: thumbnail stays until the first words of the line arrive")
        while not result.get("partial") and worker.is_alive():
            time.sleep(0.05)
    strip, text, done, temp_at = build()
    if done and "error" in result:
        print("LLM error:", result["error"])
        step("error beep, then 'AI ERROR' (nothing saved)")
        beep(300, 400)
        scroll_text("AI ERROR", accept=())
        clear_events()
        return

    # slide the thumbnail aside so the line comes into view
    step(f"output: sliding the thumbnail aside to the line ({'finished' if done else 'still being written'}): {text!r}")
    for p in range(0, text_at + 1):
        show(strip[:, p:p + COLS])
        time.sleep(SLIDE_S)
    pos, shown_text, saved = text_at, text, False
    step(f"OUTPUT: turn = scroll {SCROLL_COLS_PER_STEP} columns per step, click = done "
         "(once the line is finished)")

    def finish():
        nonlocal saved
        print(f"[{name}] {time.monotonic() - started:.1f}s, {format_temp(temperature)}"
              f"{' (subject: ' + result['subject'] + ')' if 'subject' in result else ''}: {text}")
        save_output(name, temperature, text, thumb)
        count = db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]
        step(f"line finished: saved to data/history.db (adjusted thumbnail + line + temperature), "
             f"{count} outputs stored; temperature added at the end")
        beep(1800, 60); time.sleep(0.08); beep(2400, 90)
        saved = True

    if done:
        finish()
    while True:
        ev = wait_event(0.1)
        if not saved:                                 # pick up new words / the finished line
            strip, text, done, temp_at = build()
            if done and "error" in result:
                print("LLM error:", result["error"])
                step("error while writing the line -> 'AI ERROR' (nothing saved)")
                beep(300, 400)
                scroll_text("AI ERROR", accept=())
                clear_events()
                return
            if text != shown_text:
                shown_text = text
                show(strip[:, pos:pos + COLS])
                step(f"output: line so far {text!r}")
            if done:
                finish()
        if ev is None:
            continue
        kind, delta = ev
        last = strip.shape[1] - COLS
        if kind == "turn":
            new = min(max(pos + delta * SCROLL_COLS_PER_STEP, 0), last)
            if new != pos:
                pos = new
                show(strip[:, pos:pos + COLS])
                step(f"output: knob {delta:+d} -> column {pos}/{last} ({where(pos, temp_at)})")
            else:
                step(f"output: knob {delta:+d}: already at the {'start' if pos == 0 else 'end'}"
                     + ("" if saved else " of the line so far"))
        elif kind == "click":
            if not saved:
                step("output: click ignored - the line is still being written")
                beep(400, 40)
                continue
            step("output: click -> back to the menu")
            clear_events()
            return


def view_saved(entry):
    """One saved output as the same strip as a new output: slide the thumbnail aside
    to the line, then turn = scroll (thumbnail, line, temperature), click = back to the
    thumbnails, long press = back to the start. Returns "click" or "long"."""
    thumb = np.asarray(entry["pixels"], dtype=np.uint8)
    text_at = COLS + GAP
    text_bmp = render_text(entry["text"])
    temp_at = text_at + text_bmp.shape[1] + GAP
    strip = np.concatenate([thumb, np.zeros((ROWS, GAP), np.uint8), text_bmp,
                            np.zeros((ROWS, GAP), np.uint8), render_text(format_temp(entry["temperature"]))], axis=1)
    last = strip.shape[1] - COLS
    step(f"saved output: sliding the thumbnail aside to {entry['text']!r}")
    for p in range(0, min(text_at, last) + 1):
        show(strip[:, p:p + COLS])
        time.sleep(SLIDE_S)
    pos = min(text_at, last)
    step(f"SAVED OUTPUT: turn = scroll {SCROLL_COLS_PER_STEP} columns per step, "
         "click = back to the thumbnails, hold = back to the start")
    while True:
        kind, delta = events.get()          # block until the next knob event
        if kind == "turn":
            new = min(max(pos + delta * SCROLL_COLS_PER_STEP, 0), last)
            if new != pos:
                pos = new
                show(strip[:, pos:pos + COLS])
                where = "thumbnail" if pos < text_at - GAP else "temperature" if pos + COLS > temp_at else "line"
                step(f"saved output: knob {delta:+d} -> column {pos}/{last} ({where})")
            else:
                step(f"saved output: knob {delta:+d}: already at the {'start' if pos == 0 else 'end'}")
        elif kind in ("click", "long"):
            return kind


def history():
    """Turn = switch between the saved thumbnails (newest first), click = open that
    output, long press = back to the start, ready for a new photo."""
    beep(1200, 40); time.sleep(0.06); beep(900, 60)
    outputs = load_outputs()
    step(f"HISTORY: {len(outputs)} saved outputs (newest first); turn = switch, click = open, "
         "hold = back to the start", restart=True)
    if not outputs:
        scroll_text("NO SAVED OUTPUTS", accept=("click", "long"))
        clear_events()
        return
    idx = 0
    show(outputs[idx]["pixels"])
    while True:
        o = outputs[idx]
        step(f"history slot {idx + 1}/{len(outputs)}: [{o['mode']}] {o['text']!r}, {format_temp(o['temperature'])}")
        kind, delta = events.get()          # block until the next knob event
        if kind == "turn":
            new = min(max(idx + (1 if delta > 0 else -1), 0), len(outputs) - 1)
            if new == idx:
                step(f"knob turned {delta:+d}: already at the end of the list")
                beep(400, 40)       # end of list
            else:
                idx = new
                beep(2200, 15)
            show(outputs[idx]["pixels"])
        elif kind == "click":
            step(f"knob click -> open slot {idx + 1}")
            if view_saved(o) == "long":
                kind = "long"
            else:
                step("saved output: click -> back to the thumbnails")
                show(outputs[idx]["pixels"])
        if kind == "long":
            step("knob held -> back to the start")
            beep(900, 40); time.sleep(0.06); beep(1200, 60)
            clear_events()
            return


def loop():
    action = menu()
    if action == "capture":
        capture_and_generate()
    elif action == "history":
        history()


App.run(user_loop=loop)
