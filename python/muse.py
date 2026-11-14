"""Photo -> short text, with Qwen3.5-0.8B + vision on the board's llama-server
(ported from qwen-vision-matrix/vlm.py, handoff CLAUDE.md section 4).

The arduino:llm brick starts the llama.cpp server; this module talks to it
directly over its OpenAI-compatible API, because the handoff's logic needs
options the brick's chat() does not expose: GBNF grammars, per-call temperature
and top_p, repeat_penalty, and thinking switched off.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time

import requests
from PIL import Image

SERVER = "http://llamacpp-models-runner:9999"
# The runner serves the Qwen GGUF together with its mmproj vision projector under this name.
MODEL = "Qwen3.5-0.8B-Q4_0-multimodal"
TIMEOUT = 300             # seconds per request; the first one also loads the model

# DEBUG: print every request, Qwen's output live while it is generated (and any
# "thinking" text the server returns), the server's timings, and each filter decision.
DEBUG = True


def _log(msg: str) -> None:
    if DEBUG:
        print("[qwen] " + msg, flush=True)


def warm_up() -> None:
    """Load Qwen + its vision projector now (one tiny text request), so the first
    capture doesn't also wait ~10 s for the model to load. The request starts with
    the poem system prompt, so the server caches it and the first photo only has to
    read the image (measured: 111 instead of 165 tokens, ~5 s less)."""
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SERVER}/v1/chat/completions", timeout=TIMEOUT, json={
            "model": MODEL, "max_tokens": 1,
            "messages": [{"role": "system", "content": MICRO_SYSTEM},
                         {"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {"enable_thinking": False}})
        _log(f"warm-up: model loaded and ready ({time.monotonic() - t0:.1f}s, HTTP {r.status_code})")
    except Exception as e:
        _log(f"warm-up failed ({e}); the first capture will load the model instead")


# Sent to Qwen at 320 px on the long side (640x480 -> 320x240), scaled down, never cropped:
# ~110 image tokens instead of ~170 at 448 px, measured 26 s instead of 44 s on the UNO Q.
IMAGE_MAX_SIDE = 320


def image_data_url(img: Image.Image, max_side: int = IMAGE_MAX_SIDE) -> str:
    """JPEG data URL, downscaled first: fewer image tokens = much faster on a small CPU."""
    img = img.convert("RGB")
    w0, h0 = img.size
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    _log(f"image for the model: {w0}x{h0} -> {img.width}x{img.height}, JPEG {buf.tell() // 1024} KB")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(image_url: str, prompt: str, max_tokens: int = 200, temperature: float = 0.8,
        grammar: str | None = None, system: str | None = None, top_p: float = 0.9,
        after: tuple[str, str] | None = None, on_text=None) -> str:
    """One chat request: [system], user(image first, then text).

    after=(first_prompt, first_answer) sends `prompt` as a follow-up turn of that
    earlier exchange instead. The image then sits in an unchanged prefix the server
    has already processed: Qwen3.5's recurrent layers can't reuse a cache that only
    partly matches, so a fresh request would re-read the whole image (~40 s on the
    UNO Q), while a follow-up only processes the new words.

    on_text(text_so_far), if given, is called as each piece of the answer arrives.
    """
    messages = [{"role": "system", "content": system}] if system else []
    first = prompt if after is None else after[0]
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},     # image first
        {"type": "text", "text": first},
    ]})
    if after is not None:
        messages.append({"role": "assistant", "content": after[1]})
        messages.append({"role": "user", "content": prompt})
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": 1.1,
        # Always off: with thinking on, the 0.8B model can loop "Wait, ..." until the context runs out.
        "chat_template_kwargs": {"enable_thinking": False},
        # Streamed so the debug log shows the text as Qwen produces it.
        "stream": True,
        "timings_per_token": False,
    }
    if grammar:
        body["grammar"] = grammar  # llama.cpp GBNF: the model can only emit matching text

    _log("---- request ----")
    if system:
        _log(f"system: {system}")
    if after is not None:
        _log(f"turn 1 (image + text, already processed): {after[0]!r}")
        _log(f"turn 1 answer: {after[1]!r}")
        _log(f"turn 2 (follow-up): {prompt!r}")
    else:
        _log(f"user (image + text): {prompt!r}")
    _log(f"sampling: temperature={temperature} top_p={top_p} max_tokens={max_tokens} "
         f"repeat_penalty=1.1 thinking=off grammar={'yes' if grammar else 'no'}")
    if grammar:
        for line in grammar.strip().splitlines():
            _log(f"  grammar | {line}")

    t0 = time.monotonic()
    r = requests.post(f"{SERVER}/v1/chat/completions", json=body, timeout=TIMEOUT, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"llama-server {r.status_code}: {r.text[:300]}")
    text, thinking, timings, first_token = "", "", None, None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        chunk = json.loads(data)
        timings = chunk.get("timings", timings)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            think_piece = delta.get("reasoning_content") or ""
            piece = delta.get("content") or ""
            if think_piece:
                thinking += think_piece
                _log(f"  thinking> {thinking!r}")
            if piece:
                if first_token is None:
                    first_token = time.monotonic() - t0
                    _log(f"  first token after {first_token:.1f}s (reading the image + prompt)")
                text += piece
                _log(f"  output>   {text!r}")
                if on_text is not None:
                    on_text(text)
    total = time.monotonic() - t0
    if thinking:
        _log(f"thinking text (not shown on the matrix): {thinking!r}")
    else:
        _log("thinking text: none (thinking is switched off)")
    if timings:
        _log(f"server: read {timings.get('prompt_n', '?')} new tokens in "
             f"{timings.get('prompt_ms', 0) / 1000:.1f}s, reused {timings.get('cache_n', '?')} cached, "
             f"wrote {timings.get('predicted_n', '?')} tokens in {timings.get('predicted_ms', 0) / 1000:.1f}s "
             f"({timings.get('predicted_per_second', 0):.1f} tok/s)")
    _log(f"raw answer: {text!r}  ({total:.1f}s)")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ------------------------------------------------------------------ micro poem
# The finished line is "<Subject> — " + at most MICRO_WORDS words. The limit is told to
# the model in the prompt AND enforced by the grammar while it writes, so nothing has
# to be cut afterwards (which also lets the line be shown while it is being written).
MICRO_WORDS = 5
DASH = "—"           # em-dash between the subject and the feeling part
# Short and example-free, so the subject question can't copy anything from it;
# both calls share it (and the image), so the server reuses the encoded image.
MICRO_SYSTEM = (
    "You are a poet writing for a tiny glowing screen that fits only a few words. "
    "You see how a moment feels, not how it looks: tenderness, delight, longing, "
    "peace, mischief, loneliness, wonder. You never mention colours."
)
SUBJECT_PROMPT = "What is the main subject of this photo? Answer with one or two simple words."
# Articles are allowed (and stripped later); otherwise a small model fuses them
# into the word ("aboat", "abird").
SUBJECT_GRAMMAR = r'''root ::= ("a " | "an " | "the ")? [a-z]{2,12} (" " [a-z]{2,12})?
'''
# Qwen writes only the feeling part; the app puts "<Subject> — " in front itself
# (asking the model to start with the subject made it echo the instruction:
# "The line starts with the word girl.").
MICRO_PROMPT = (
    "Write two to five plain words, never more than five, that give the {subject} in this "
    "photo a feeling. It must make sense for this photo. No full stop at the end.\n"
    "Examples for other photos: for a bench \"still saving a seat\", "
    "for an old road \"forgetting our names\", for bread \"smelling like home\", "
    "for a kite \"refusing to come down\""
)
# Short words the matrix font can draw, optionally ending with "!" or "?"; no full stop
# or other closing character. The prompt asks for at most five words; the grammar allows
# up to seven, because a hard stop at five made the model glue its last words together
# to fit ("it burns hotter than iteverwax"). Words past the fifth are dropped (they are
# rare, and never shown on the matrix either).
MICRO_GRAMMAR = r'''root ::= word (sep word){1,6} end
sep ::= " " | ", "
word ::= [A-Za-z] [a-z]{0,8}
end ::= ("!" | "?")?
'''
_DANGLING = {"a", "an", "the", "in", "of", "my", "on", "to", "with", "and", "for",
             "at", "is", "its", "by", "from", "into", "under", "over", "as",
             "against", "through", "toward", "towards", "near", "like", "beside",
             "behind", "across", "upon", "onto", "than", "that", "which", "who", "her",
             "his", "their", "our", "your", "this", "these", "those", "very", "so",
             "too", "where", "what", "when", "if", "because", "but", "or", "seems", "feels",
             "just", "being", "can", "will", "would", "could", "feeling", "want", "wants",
             "making", "full", "every", "some", "no"}

# Words the screen poems should not use: colours (the matrix has none) and the
# cliches a small model reaches for first.
_BANNED = re.compile(r"\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|gr[ae]y|"
                     r"gold(en)?|silver|violet|crimson|scarlet|amber|ginger|tan|beige|"
                     r"colou?rs?|colou?rful|rainbow|whisper\w*|danc\w*|shimmer\w*|glow\w*)\b", re.I)
_EXAMPLES = {"still saving a seat", "forgetting our names", "smelling like home",
             "refusing to come down"}
_MOODS = re.compile(r"\b(warmth|joy|nostalgia|calm|longing|mischief|loneliness|wonder|comfort|"
                    r"tenderness|delight|peace)\b", re.I)
_ECHO = re.compile(r"\b(line|starts?|photo|image|picture|subject)\b", re.I)   # prompt words echoed back


def subject(image_url: str, raw: list | None = None) -> str:
    """The photo's main subject in 1-2 plain words ("dog", "coffee cup"), no colours.

    `raw`, if given, receives the model's unfiltered answer (for the follow-up turn).
    """
    _log("STEP 1/2: ask for the main subject (1-2 words, low temperature)")
    out = ask(image_url, SUBJECT_PROMPT, grammar=SUBJECT_GRAMMAR, system=MICRO_SYSTEM,
              temperature=0.2, max_tokens=8)
    if raw is not None:
        raw.append(out)
    words = [w for w in out.split() if w not in ("a", "an", "the") and not _BANNED.search(w)]
    dropped = [w for w in out.split() if w not in words]
    if dropped:
        _log(f"subject: dropped articles/colour words {dropped}")
    while words and words[-1] in _DANGLING:              # "boat and" -> "boat"
        _log(f"subject: dropped dangling last word {words[-1]!r}")
        words.pop()
    subj = " ".join(words[-2:]) or "moment"
    _log(f"subject = {subj!r}" + ("  (nothing usable, fallback)" if not words else ""))
    return subj


def _clean_phrase(phrase: str, subj: str, log: bool = False) -> str:
    """Tidy the feeling part: no repeated subject, lowercase start, no full stop."""
    words = phrase.strip().split()
    # If the model still named the subject itself ("girl smiling", "a small
    # rectangle still quiet"), drop that, with any article in front of it.
    k = 1 if words and words[0].lower() in ("a", "an", "the") else 0
    n = len(subj.split())
    if [w.strip(",.!?").lower() for w in words[k:k + n]] == subj.lower().split() and len(words) > k + n:
        if log:
            _log(f"feeling part repeats the subject: dropped {' '.join(words[:k + n])!r}")
        words = words[k + n:]
    phrase = " ".join(words).lstrip(", ").rstrip(" ,.")
    return phrase[:1].lower() + phrase[1:]


def micro(image_url: str, tries: int = 3, temperature: float = 1.0,
          on_partial=None) -> tuple[str, str]:
    """A tiny mood line about the photo: ("Balloon — waiting for the moon", "balloon").

    1. Ask for the main subject ("balloon"), so the line stays about the photo.
    2. Ask only for the feeling part: two to five plain words. The limit is in the
       prompt, so the model plans for it (the grammar allows a little slack, see
       MICRO_GRAMMAR); it may end with "!" or "?", never with a full stop. Sampled warm
       (temperature ~1.0). The app builds the line itself as "<Subject> — <feeling>".
    3. The first candidate that passes the filters (colour/cliche words, prompt
       echoes, copied examples, mood lists) is used; a rejected one is re-asked, up
       to `tries` times - cheap, because the photo is already read.

    on_partial(line_so_far), if given, receives the line while it is written:
    "Balloon —" as soon as the subject is known, then each new word.
    """
    raw: list = []
    subj = subject(image_url, raw)
    prompt = MICRO_PROMPT.format(subject=subj)
    head = subj[:1].upper() + subj[1:]
    if on_partial is not None:
        on_partial(f"{head} {DASH}")
    best = ""
    for i in range(tries):
        _log(f"STEP 2/2: feeling part, candidate {i + 1}/{tries} "
             f"(warm sampling, follow-up turn, max {MICRO_WORDS} words)")
        stream = None
        if on_partial is not None:
            stream = lambda t: on_partial(
                f"{head} {DASH} {' '.join(_clean_phrase(t, subj).split()[:MICRO_WORDS])}".rstrip())
        # Asked as a follow-up to the subject question: same image prefix, no re-read.
        phrase = ask(image_url, prompt, grammar=MICRO_GRAMMAR, system=MICRO_SYSTEM,
                     temperature=temperature, top_p=0.95, max_tokens=32,
                     after=(SUBJECT_PROMPT, raw[0]), on_text=stream)
        phrase = _clean_phrase(phrase, subj, log=True)
        if len(phrase.split()) > MICRO_WORDS:
            _log(f"more than {MICRO_WORDS} words: dropped {' '.join(phrase.split()[MICRO_WORDS:])!r}")
            phrase = " ".join(phrase.split()[:MICRO_WORDS]).rstrip(",")
        # The line can still stop on a dangling word ("waiting for the")
        while len(phrase.split()) > 2 and phrase.split()[-1].lower().strip("!?") in _DANGLING:
            _log(f"dropped dangling last word {phrase.split()[-1]!r}")
            phrase = " ".join(phrase.split()[:-1])
        low = phrase.lower()
        reasons = [name for name, hit in (
            ("empty", not phrase.strip(" .,!?")), ("colour/cliche word", bool(_BANNED.search(phrase))),
            ("echoes the prompt", bool(_ECHO.search(phrase))),
            ("copies an example", any(e in low for e in _EXAMPLES)),
            ("lists 2+ mood words", len(_MOODS.findall(phrase)) >= 2)) if hit]
        if reasons:
            _log(f"rejected {phrase!r}: {', '.join(reasons)}" + (" -> asking again" if i + 1 < tries else ""))
            if on_partial is not None:
                on_partial(f"{head} {DASH}")       # the shown line starts over
            continue
        best = f"{head} {DASH} {phrase}"
        _log(f"passes the filters ({len(phrase.split())} words): using it")
        break
    final = best or head
    _log(f"final line = {final!r}" + ("" if best else "  (no usable candidate: subject only)"))
    return final, subj


# ------------------------------------------------------------- other modes
FREE_SYSTEM = (
    "You are the voice of a tiny handheld camera. Reply in plain English text only: "
    "no markdown, no emojis, no lists with symbols, at most 50 words."
)


def describe(image_url: str, prompt: str, on_partial=None) -> str:
    """Free-form answer for the longer menu modes (poem, haiku, story, ...)."""
    _log("free-form mode: one request, no grammar")
    return ask(image_url, prompt, system=FREE_SYSTEM, temperature=0.7, max_tokens=120,
               on_text=on_partial)
