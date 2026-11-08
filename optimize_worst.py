#!/usr/bin/env python3
"""optimize_worst - find the least space-efficient videos in a directory and re-encode them.
Yes this is 100% Claude vibecoded

Efficiency is measured as seconds of video per MiB of file (Sec/MiB). A LOW
number means a lot of bytes per second of video, i.e. a good re-encode target.

Re-encode templates use these placeholders:
    $f    full path of the input file WITHOUT its extension
    $ext  the input file's extension, including the dot (".mkv")
The output must be written to "$f.new<ext>" (any extension is fine, as long
as it starts with "$f.new") so the result can be found and verified.

A new file replaces the original only if it is smaller (by --min-saving %) and
its duration matches. Originals are moved to ".optimize_originals/" next to
them (or deleted with --delete-originals). Files that grew are discarded.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

__version__ = "1.1.0"

MIB = 1024 * 1024
VIDEO_EXTS = {".mkv", ".webm", ".mp4", ".m4v", ".avi", ".mov", ".ts",
              ".flv", ".wmv", ".mpg", ".mpeg"}
STATE_NAME = ".optimize_worst.json"
LOG_NAME = ".optimize_worst.log"
BACKUP_DIR = ".optimize_originals"
FFMPEG_QUIET = ["-hide_banner", "-loglevel", "error", "-nostdin", "-nostats",
                "-progress", "pipe:1", "-stats_period", "0.5", "-y"]

# "save_vs_h264" is how much smaller this encoder's output is than an
# unoptimized H.264 encode of the same video (0.80 = ~20% of the H.264 size).
# For other source codecs the script first converts the source back to an
# H.264-equivalent size using CODEC_SIZE_VS_H264 below, then applies the saving.
PRESETS = {
    "cartoon": {
        "save_vs_h264": 0.80,
        "cmd": ('ffmpeg -i "$f$ext" -vf "scale=-2:420:flags=lanczos" '
                '-c:v libsvtav1 -preset 4 -crf 42 '
                '-svtav1-params tune=0:enable-overlays=1 -c:a copy "$f.new$ext"'),
    },
    "film": {  # rough guess, not measured
        "save_vs_h264": 0.50,
        "cmd": ('ffmpeg -i "$f$ext" -c:v libsvtav1 -preset 6 -crf 32 '
                '-svtav1-params tune=0 -c:a copy "$f.new$ext"'),
    },
}

# Typical file size of each codec relative to H.264 at similar quality
# (H.264 = 1.0, lower = more efficient). Rough rules of thumb, not measurements;
# codecs not listed here get no projection.
CODEC_SIZE_VS_H264 = {
    "h264": 1.00, "avc": 1.00,
    "hevc": 0.60, "h265": 0.60,
    "vp9": 0.65,
    "av1": 0.50,
    "vp8": 1.10,
    "vc1": 1.10, "wmv3": 1.20, "wmv2": 1.40, "wmv1": 1.50,
    "mpeg4": 1.50, "msmpeg4v3": 1.60, "msmpeg4v2": 1.60,
    "h263": 1.60, "theora": 1.30,
    "mpeg2video": 2.00, "mpeg1video": 2.50,
}

IS_TTY = sys.stdout.isatty()
COLOR = IS_TTY and "NO_COLOR" not in os.environ


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    path: str
    size: int
    mtime_ns: int
    duration: float = 0.0
    codec: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def mib(self) -> float:
        return self.size / MIB

    @property
    def sec_per_mib(self) -> float:
        return self.duration / self.mib if self.mib else float("inf")


def paint(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


def pct_change(old: int, new: int) -> str:
    return f"{(new - old) / old * 100:+.0f}%"


def substitute(token: str, stem: str, ext: str) -> str:
    return re.sub(r"\$(ext|f)(?![A-Za-z0-9_])",
                  lambda m: ext if m.group(1) == "ext" else stem, token)


def build_argv(template: str, stem: str, ext: str) -> list[str]:
    argv = [substitute(t, stem, ext) for t in shlex.split(template)]
    if argv and os.path.basename(argv[0]) == "ffmpeg":
        argv[1:1] = FFMPEG_QUIET
    return argv


def outputs_for(stem: str) -> list[str]:
    return glob.glob(glob.escape(stem) + ".new.*")


def remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# state (processed files + duration cache)
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"version": 1, "done": {}, "durations": {}}
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if loaded.get("version") == 1:
                self.data = loaded
        except (OSError, ValueError):
            pass

    def _key(self, p: str) -> str:
        return os.path.relpath(p, os.path.dirname(self.path))

    def is_done(self, e: Entry) -> bool:
        rec = self.data["done"].get(self._key(e.path))
        return bool(rec and rec["size"] == e.size and rec["mtime_ns"] == e.mtime_ns)

    def cached_info(self, e: Entry) -> Optional[tuple[float, str]]:
        rec = self.data["durations"].get(self._key(e.path))
        if (rec and rec["size"] == e.size and rec["mtime_ns"] == e.mtime_ns
                and "codec" in rec):
            return rec["duration"], rec["codec"]
        return None

    def set_info(self, e: Entry) -> None:
        with self.lock:
            self.data["durations"][self._key(e.path)] = {
                "size": e.size, "mtime_ns": e.mtime_ns,
                "duration": e.duration, "codec": e.codec}

    def mark_done(self, final_path: str, original_path: str) -> None:
        try:
            st = os.stat(final_path)
        except OSError:
            return
        with self.lock:
            self.data["done"][self._key(final_path)] = {
                "size": st.st_size, "mtime_ns": st.st_mtime_ns}
            if original_path != final_path:
                self.data["done"].pop(self._key(original_path), None)
                self.data["durations"].pop(self._key(original_path), None)

    def save(self) -> None:
        with self.lock:
            try:
                fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=".ow-")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.data, fh)
                os.replace(tmp, self.path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
def scan(root: str, recursive: bool) -> list[Entry]:
    found, stack = [], [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError as exc:
            print(f"warning: cannot read {d}: {exc.strerror}", file=sys.stderr)
            continue
        with it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if recursive:
                            stack.append(e.path)
                        continue
                    if not e.is_file():
                        continue
                    if os.path.splitext(e.name)[1].lower() not in VIDEO_EXTS or ".new." in e.name:
                        continue
                    st = e.stat()
                except OSError:
                    continue
                if st.st_size > 0:
                    found.append(Entry(e.path, st.st_size, st.st_mtime_ns))
    return found


def probe_info(path: str) -> tuple[Optional[float], str]:
    """Return (duration in seconds, first video stream codec name)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration:stream=codec_name",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout or "{}")
        dur = float(data.get("format", {}).get("duration"))
        streams = data.get("streams") or [{}]
        return (dur if dur > 0 else None), str(streams[0].get("codec_name", "")).lower()
    except (ValueError, TypeError, subprocess.SubprocessError, OSError):
        return None, ""


def probe_duration(path: str) -> Optional[float]:
    return probe_info(path)[0]


def measure(entries: list[Entry], state: State) -> list[Entry]:
    todo = []
    for e in entries:
        cached = state.cached_info(e)
        if cached and cached[0]:
            e.duration, e.codec = cached
        else:
            todo.append(e)

    def work(e: Entry) -> Entry:
        dur, e.codec = probe_info(e.path)
        e.duration = dur or 0.0
        return e

    if todo:
        with cf.ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
            for e in pool.map(work, todo):
                if e.duration:
                    state.set_info(e)
        state.save()
    return [e for e in entries if e.duration > 0]


# --------------------------------------------------------------------------- #
# table display
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    entry: Entry
    status: str            # replaced | kept | failed | cancelled
    message: str
    new_size: int = 0
    seconds: float = 0.0


@dataclass
class Row:
    entry: Entry
    index: int
    state: str = "queued"   # queued | running | replaced | kept | failed
    pct: float = 0.0
    est: int = 0            # estimated final size in bytes
    speed: float = 0.0
    eta: float = 0.0
    start: float = 0.0
    end: float = 0.0
    final: int = 0
    msg: str = ""


# key, header, width (0 = flexible), alignment
COLUMNS = [
    ("idx", "#", 2, ">"),
    ("name", "File", 0, "<"),
    ("len", "Length", 7, ">"),
    ("spm", "Sec/MiB", 7, ">"),
    ("size", "Size", 10, ">"),
    ("bar", "Progress", 15, "<"),
    ("est", "Est. size", 11, ">"),
    ("espm", "Est.S/MiB", 10, ">"),
    ("delta", "Δ", 5, ">"),
    ("speed", "Speed", 6, ">"),
    ("elapsed", "Elapsed", 7, ">"),
    ("eta", "ETA", 7, ">"),
]
# before anything runs, the est columns hold the preset's projection
STATIC_KEYS = ["idx", "name", "len", "spm", "size", "est", "espm", "delta"]
STATIC_HEADERS = {"est": "Proj. size", "espm": "Proj.S/MiB", "delta": "Proj."}
DROP_ORDER = ["speed", "eta", "espm", "len", "delta"]
GAP = 2


class Display:
    """One live table: every target is a row, live values update in place."""

    def __init__(self, entries: list[Entry], title: str, live: bool,
                 proj_save: Optional[float] = None):
        self.rows = {e.path: Row(e, i) for i, e in enumerate(entries, 1)}
        self.title = title
        self.proj_save = proj_save
        self.live = live and IS_TTY
        self.lock = threading.Lock()
        self.drawn = 0
        self.t0 = time.monotonic()
        self.cursor_hidden = False

    # ---- state updates (called from worker threads) ----
    def start(self, path: str) -> None:
        with self.lock:
            r = self.rows[path]
            r.state, r.start = "running", time.monotonic()

    def update(self, path: str, pct: float, est: int, speed: float, eta: float) -> None:
        with self.lock:
            r = self.rows[path]
            r.pct, r.est, r.speed, r.eta = min(pct, 1.0), est, speed, eta

    def finish(self, res: Result) -> None:
        with self.lock:
            r = self.rows[res.entry.path]
            r.state = res.status if res.status in ("replaced", "kept", "failed") else "failed"
            r.end, r.final, r.msg = time.monotonic(), res.new_size, res.message
            r.pct = 1.0 if r.state != "failed" else r.pct
            if not self.live:
                self._plain_line(r)

    def _plain_line(self, r: Row) -> None:
        e = r.entry
        n = sum(x.state in ("replaced", "kept", "failed") for x in self.rows.values())
        tag = f"[{n}/{len(self.rows)}]"
        if r.state == "replaced":
            line = f" ✔ {tag} {e.name}  {e.mib:.2f} → {r.final / MIB:.2f} MiB ({pct_change(e.size, r.final)})"
        elif r.state == "kept":
            line = f" ↩ {tag} {e.name}  {e.mib:.2f} → {r.final / MIB:.2f} MiB; original kept"
        else:
            line = f" ✘ {tag} {e.name}  FAILED: {r.msg}"
        print(line, flush=True)

    # ---- projection (preset's saving vs unoptimized H.264) ----
    def proj_size(self, e: Entry) -> Optional[int]:
        rel = CODEC_SIZE_VS_H264.get(e.codec)
        if self.proj_save is None or not rel:
            return None
        # source -> H.264-equivalent size -> this encoder's size
        return max(int(e.size / rel * (1 - self.proj_save)), 1)

    # ---- layout ----
    def _layout(self, keys: list[str], width: int) -> list[tuple]:
        cols = [c for c in COLUMNS if c[0] in keys]
        drop = [k for k in DROP_ORDER if k in keys]
        while True:
            fixed = sum(c[2] for c in cols) + GAP * (len(cols) - 1) + 4
            name_w = width - 1 - fixed
            if name_w >= 18 or not drop:
                break
            gone = drop.pop(0)
            cols = [c for c in cols if c[0] != gone]
        name_w = max(8, min(40, name_w))
        return [(k, h, name_w if w == 0 else w, a) for k, h, w, a in cols]

    # ---- rendering ----
    def _cells(self, r: Row, now: float) -> dict[str, tuple[str, tuple]]:
        e = r.entry
        c: dict[str, tuple[str, tuple]] = {
            "idx": (str(r.index), ("2",)),
            "name": (e.name, ()),
            "len": (fmt_time(e.duration), ("2",)),
            "spm": (f"{e.sec_per_mib:.2f}", ()),
            "size": (f"{e.mib:.2f} MiB", ()),
        }
        dash = ("–", ("2",))
        for k in ("est", "espm", "delta", "speed", "elapsed", "eta"):
            c[k] = dash
        c["bar"] = ("queued", ("2",))

        def bar(pct: float, code: str) -> tuple[str, tuple]:
            n = int(pct * 10 + 0.5)
            return ("━" * n + "─" * (10 - n) + f" {pct * 100:3.0f}%", (code,))

        p = self.proj_size(e)
        if p:
            c["est"] = (f"~{p / MIB:.2f} MiB", ("2",))
            c["espm"] = (f"~{e.duration / (p / MIB):.2f}", ("2",))
            c["delta"] = (pct_change(e.size, p), ("2",))

        if r.state == "queued":
            c["name"] = (e.name, ("2",))
        elif r.state == "running":
            el = now - r.start
            c["name"] = (e.name, ("1",))
            c["bar"] = bar(r.pct, "36")
            c["elapsed"] = (fmt_time(el), ())
            if r.speed > 0:
                c["speed"] = (f"{r.speed:.2f}x", ())
            eta = r.eta
            if eta <= 0 and r.pct > 0.02:
                eta = el * (1 - r.pct) / r.pct
            if eta > 0:
                c["eta"] = (fmt_time(eta), ())
            if r.est > 0:
                c["est"] = (f"{r.est / MIB:.2f} MiB", ("36",))
                c["espm"] = (f"{e.duration / (r.est / MIB):.2f}", ("36",))
                c["delta"] = (pct_change(e.size, r.est), ("32",) if r.est < e.size else ("33",))
            elif not p:
                c["est"] = ("…", ("2",))
        elif r.state in ("replaced", "kept"):
            good = r.state == "replaced"
            code = "32" if good else "33"
            c["bar"] = bar(1.0, code)
            c["est"] = (f"{r.final / MIB:.2f} MiB", (code, "1"))
            c["espm"] = (f"{e.duration / (r.final / MIB):.2f}", (code,))
            c["delta"] = (pct_change(e.size, r.final), (code,))
            c["elapsed"] = (fmt_time(r.end - r.start), ())
            c["name"] = (e.name, ())
        else:  # failed
            c["bar"] = ("failed", ("31", "1"))
            c["name"] = (e.name, ("31",))
            if r.start:
                c["elapsed"] = (fmt_time(r.end - r.start), ())
        return c

    def _icon(self, r: Row) -> str:
        return {"queued": paint("·", "2"), "running": paint("▶", "36", "1"),
                "replaced": paint("✔", "32"), "kept": paint("↩", "33"),
                "failed": paint("✘", "31")}[r.state]

    def _visible(self, budget: int) -> tuple[list[Row], int]:
        rows = list(self.rows.values())
        if len(rows) <= budget:
            return rows, 0
        budget = max(budget - 1, 3)
        running = [r for r in rows if r.state == "running"]
        finished = sorted((r for r in rows if r.state in ("replaced", "kept", "failed")),
                          key=lambda r: -r.index)
        queued = [r for r in rows if r.state == "queued"]
        chosen = (running + finished[: max(budget // 3, 1)] + queued)[:budget]
        return sorted(chosen, key=lambda r: r.index), len(rows) - len(chosen)

    def render(self) -> list[str]:
        size = shutil.get_terminal_size((120, 40))
        width, height = size.columns, size.lines
        keys = [c[0] for c in COLUMNS] if self.live or self._any_started() else STATIC_KEYS
        cols = self._layout(keys, width)
        now = time.monotonic()

        def line(cells: dict[str, tuple[str, tuple]], icon: str) -> str:
            parts = []
            for k, _, w, a in cols:
                text, codes = cells[k]
                parts.append(paint(f"{clip(text, w):{a}{w}}", *codes))
            return "  " + icon + " " + (" " * GAP).join(parts)

        rows = list(self.rows.values())
        n_ok = sum(r.state == "replaced" for r in rows)
        n_kept = sum(r.state == "kept" for r in rows)
        n_fail = [r for r in rows if r.state == "failed"]
        n_run = sum(r.state == "running" for r in rows)
        fail_lines = min(len(n_fail), 3)
        budget = max(height - 8 - fail_lines, 4)
        visible, hidden = self._visible(budget)

        plain_w = sum(w for _, _, w, _ in cols) + GAP * (len(cols) - 1) + 4
        out = [" " + paint(clip(self.title, width - 2), "1")]
        static = not (self.live or self._any_started())
        header = {k: (STATIC_HEADERS.get(k, h) if static else h, ("1", "4"))
                  for k, h, _, _ in cols}
        out.append(line(header, " "))
        out.append("  " + paint("─" * min(plain_w - 2, width - 3), "2"))
        for r in visible:
            out.append(line(self._cells(r, now), self._icon(r)))
        if hidden:
            out.append("  " + paint(f"  … {hidden} more not shown (terminal too short)", "2"))
        out.append("  " + paint("─" * min(plain_w - 2, width - 3), "2"))
        if self.live or self._any_started():
            out.append(self._footer(rows, n_ok, n_kept, len(n_fail), n_run, now, width))
        else:
            out.append(self._proj_footer(rows, width))
        for r in n_fail[-3:]:
            out.append("  " + paint(clip(f"✘ {r.entry.name}: {r.msg}", width - 4), "31"))
        return out

    def _any_started(self) -> bool:
        return any(r.state != "queued" for r in self.rows.values())

    def _footer(self, rows, n_ok, n_kept, n_fail, n_run, now, width) -> str:
        total = len(rows)
        finished = n_ok + n_kept + n_fail
        saved = sum(r.entry.size - r.final for r in rows if r.state == "replaced")
        projected = saved
        for r in rows:
            if r.state == "running":
                guess = r.est or self.proj_size(r.entry) or r.entry.size
                projected += max(r.entry.size - guess, 0)
            elif r.state == "queued":
                projected += max(r.entry.size - (self.proj_size(r.entry) or r.entry.size), 0)
        segs = [
            (f"{finished}/{total} done", ("1",)),
            (f"▶ {n_run} running", ("36",)),
            (f"✔ {n_ok}", ("32",)),
            (f"↩ {n_kept}", ("33",)),
            (f"✘ {n_fail}", ("31",) if n_fail else ("2",)),
            (f"saved {saved / MIB:.1f} MiB", ("32", "1")),
            (f"projected save {projected / MIB:.1f} MiB", ("2",)),
            (f"elapsed {fmt_time(now - self.t0)}", ()),
        ]
        shown, used = [], 4
        for text, codes in segs:
            if used + len(text) + 3 > width - 1:
                break
            shown.append(paint(text, *codes))
            used += len(text) + 3
        return "  " + paint(" │ ", "2").join(shown)

    def _proj_footer(self, rows, width) -> str:
        total = sum(r.entry.size for r in rows)
        known = [r for r in rows if self.proj_size(r.entry)]
        if not known:
            hint = ("unknown source codecs, no projection" if self.proj_save is not None
                    else "no projection (custom command; add --proj-save PCT)")
            return "  " + paint(clip(f"{len(rows)} files · {total / MIB:.1f} MiB · {hint}", width - 4), "2")
        before = sum(r.entry.size for r in known)
        after = sum(self.proj_size(r.entry) for r in known)
        other = sum(r.entry.codec not in ("h264", "avc") for r in known)
        note = f", {other} non-H.264 via codec factors" if other else ""
        text = (f"projected: {before / MIB:.1f} → {after / MIB:.1f} MiB across {len(known)}/{len(rows)} "
                f"files{note}, saves {(before - after) / MIB:.1f} MiB ({pct_change(before, after)})")
        return "  " + paint(clip(text, width - 4), "32", "1")

    # ---- terminal drawing ----
    def draw(self) -> None:
        with self.lock:
            lines = self.render()
        if not self.live:
            return
        buf = ""
        if not self.cursor_hidden:
            buf += "\033[?25l"
            self.cursor_hidden = True
        if self.drawn:
            buf += f"\033[{self.drawn}F\033[J"
        buf += "".join(l + "\n" for l in lines)
        sys.stdout.write(buf)
        sys.stdout.flush()
        self.drawn = len(lines)

    def close(self) -> None:
        if self.live:
            self.draw()
            if self.cursor_hidden:
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                self.cursor_hidden = False

    def print_static(self) -> None:
        with self.lock:
            lines = self.render()
        print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, args, template: str, state: State, display: Display):
        self.args, self.template, self.state, self.display = args, template, state, display
        self.stop = threading.Event()
        self.procs: set[subprocess.Popen] = set()
        self.plock = threading.Lock()
        self.log_lock = threading.Lock()
        self.log_path = os.path.join(args.root, LOG_NAME)

    def kill_all(self) -> None:
        self.stop.set()
        with self.plock:
            for p in list(self.procs):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass

    def _log(self, text: str) -> None:
        with self.log_lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{time.strftime('%F %T')}] {text}\n")
            except OSError:
                pass

    def run(self, e: Entry) -> Result:
        t0 = time.monotonic()
        stem, ext = os.path.splitext(e.path)
        for stale in outputs_for(stem):
            remove_quiet(stale)
        if self.stop.is_set():
            return Result(e, "cancelled", "cancelled")
        self.display.start(e.path)
        try:
            res = self._encode(e, stem, ext)
        except Exception as exc:  # never let one file kill the batch
            res = Result(e, "failed", f"{type(exc).__name__}: {exc}")
            self._log(f"{e.path}: unexpected error {exc!r}")
        finally:
            for leftover in outputs_for(stem):
                if os.path.exists(e.path):  # original intact -> leftover is junk
                    remove_quiet(leftover)
        res.seconds = time.monotonic() - t0
        return res

    def _on_progress(self, e: Entry, stem: str, block: dict, cache: dict) -> None:
        us = block.get("out_time_us") or block.get("out_time_ms") or ""
        if not us.lstrip("-").isdigit():
            return
        out_s = max(int(us), 0) / 1e6
        pct = out_s / e.duration

        size = 0
        ts = block.get("total_size", "")
        if ts.isdigit():
            size = int(ts)
        else:
            if "out" not in cache:
                found = outputs_for(stem)
                cache["out"] = found[0] if found else None
            if cache["out"]:
                try:
                    size = os.path.getsize(cache["out"])
                except OSError:
                    pass
        est = int(size / pct) if size > 0 and pct >= 0.03 else 0

        speed = 0.0
        try:
            speed = float(block.get("speed", "").strip().rstrip("x"))
        except ValueError:
            pass
        eta = (e.duration - out_s) / speed if speed > 0 else 0.0
        self.display.update(e.path, pct, est, speed, max(eta, 0.0))

    def _encode(self, e: Entry, stem: str, ext: str) -> Result:
        argv = build_argv(self.template, stem, ext)
        self._log(f"RUN {shlex.join(argv)}")
        with tempfile.TemporaryFile() as errf:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=errf, text=True, start_new_session=True)
            with self.plock:
                self.procs.add(proc)
            try:
                block: dict[str, str] = {}
                cache: dict = {}
                for line in proc.stdout:
                    key, _, val = line.strip().partition("=")
                    block[key] = val
                    if key == "progress":      # last line of every ffmpeg stats block
                        self._on_progress(e, stem, block, cache)
                        block = {}
                proc.wait()
            finally:
                with self.plock:
                    self.procs.discard(proc)
                if proc.poll() is None:
                    proc.kill()
            errf.seek(0)
            tail = errf.read().decode("utf-8", "replace").strip().splitlines()[-8:]

        if self.stop.is_set():
            return Result(e, "cancelled", "cancelled")
        if proc.returncode != 0:
            self._log(f"FAIL {e.path} rc={proc.returncode}\n" + "\n".join(tail))
            last = tail[-1] if tail else "no error output"
            return Result(e, "failed", f"ffmpeg exited {proc.returncode}: {clip(last, 80)} (see {LOG_NAME})")

        outs = outputs_for(stem)
        if len(outs) != 1:
            return Result(e, "failed", f"expected one output matching '$f.new.*', found {len(outs)}")
        out = outs[0]
        new_size = os.path.getsize(out)
        if new_size == 0:
            return Result(e, "failed", "output is empty")

        new_dur = probe_duration(out)
        if new_dur is None or abs(new_dur - e.duration) > max(2.0, 0.02 * e.duration):
            return Result(e, "failed", f"output duration mismatch ({new_dur} vs {e.duration:.1f}s)")

        if new_size > e.size * (1 - self.args.min_saving / 100):
            self.state.mark_done(e.path, e.path)
            remove_quiet(out)
            return Result(e, "kept", "no real gain, original kept", new_size)

        return self._swap(e, stem, out, new_size)

    def _swap(self, e: Entry, stem: str, out: str, new_size: int) -> Result:
        new_ext = out[len(stem) + len(".new"):]
        final = stem + new_ext
        if final != e.path and os.path.exists(final):
            return Result(e, "failed", f"{os.path.basename(final)} already exists; result left as {os.path.basename(out)}")
        if self.args.delete_originals:
            backup = None
        else:
            bdir = os.path.join(os.path.dirname(e.path), BACKUP_DIR)
            os.makedirs(bdir, exist_ok=True)
            backup = os.path.join(bdir, e.name)
            n = 1
            while os.path.exists(backup):
                backup = os.path.join(bdir, f"{e.name}.{n}")
                n += 1
        try:
            if backup:
                shutil.move(e.path, backup)
            else:
                os.remove(e.path)
            os.replace(out, final)
        except OSError as exc:
            if backup and os.path.exists(backup) and not os.path.exists(e.path):
                shutil.move(backup, e.path)  # roll back
            return Result(e, "failed", f"could not swap files: {exc.strerror}")
        self.state.mark_done(final, e.path)
        return Result(e, "replaced", os.path.basename(final), new_size)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="optimize_worst.py",
        description="Find the least space-efficient videos (lowest Sec/MiB) and re-encode them.",
        epilog='Custom command example:\n  --ffmpeg-cmd \'ffmpeg -i "$f$ext" -c:v libx265 -crf 28 -c:a copy "$f.new$ext"\'',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=".", help="directory to scan (default: .)")
    p.add_argument("--workers", type=int, default=4, help="parallel encodes (default: 4)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ffmpeg-preset", choices=sorted(PRESETS), default="cartoon",
                   help="built-in command (default: cartoon)")
    g.add_argument("--ffmpeg-cmd", metavar="CMD",
                   help="your own ffmpeg command; $f = path without extension, $ext = extension; "
                        "must write to \"$f.new<ext>\"")
    p.add_argument("--proj-save", type=float, metavar="PCT",
                   help="how much smaller (in %%) your custom --ffmpeg-cmd output is than an unoptimized H.264 encode "
                        "(presets carry their own, e.g. cartoon = 80)")
    p.add_argument("--top", type=int, default=20, help="number of worst files to target (default: 20)")
    p.add_argument("-r", "--recursive", action="store_true", help="scan subdirectories too")
    p.add_argument("--min-saving", type=float, default=3.0, metavar="PCT",
                   help="only replace if the new file is at least PCT%% smaller (default: 3)")
    p.add_argument("--delete-originals", action="store_true",
                   help=f"delete replaced originals instead of moving them to {BACKUP_DIR}/")
    p.add_argument("--force", action="store_true", help="also re-process files done in a previous run")
    p.add_argument("-n", "--dry-run", action="store_true", help="only show the ranking")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    if a.workers < 1 or a.top < 1:
        p.error("--workers and --top must be >= 1")
    if a.proj_save is not None and not 0 <= a.proj_save < 100:
        p.error("--proj-save must be between 0 and 100")
    if not 0 <= a.min_saving < 100:
        p.error("--min-saving must be between 0 and 100")
    a.root = os.path.abspath(a.root)
    if not os.path.isdir(a.root):
        p.error(f"not a directory: {a.root}")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    preset = None if args.ffmpeg_cmd else PRESETS[args.ffmpeg_preset]
    template = args.ffmpeg_cmd or preset["cmd"]
    proj_save = preset["save_vs_h264"] if preset else (
        args.proj_save / 100 if args.proj_save is not None else None)
    label ="custom command" if args.ffmpeg_cmd else f"preset: {args.ffmpeg_preset}"
    if "$f.new" not in template:
        print('error: command must write its output to "$f.new<ext>"', file=sys.stderr)
        return 2
    try:
        first = shlex.split(template)[0]
    except (ValueError, IndexError):
        print("error: could not parse the ffmpeg command", file=sys.stderr)
        return 2
    for tool in {"ffprobe", first}:
        if not shutil.which(tool):
            print(f"error: '{tool}' not found in PATH", file=sys.stderr)
            return 2

    state = State(os.path.join(args.root, STATE_NAME))
    print("🔍 Scanning directory and calculating efficiency metrics...")
    entries = scan(args.root, args.recursive)
    if not args.force:
        entries = [e for e in entries if not state.is_done(e)]
    entries = measure(entries, state)
    if not entries:
        print("No candidate video files found.")
        return 0

    entries.sort(key=lambda e: e.sec_per_mib)
    targets = entries[: args.top]
    print()

    if args.dry_run:
        Display(targets, f"📉 THE {len(targets)} LEAST EFFICIENT FILES (targets for re-encoding)",
                live=False, proj_save=proj_save).print_static()
        return 0

    title = (f"📉 RE-ENCODING THE {len(targets)} LEAST EFFICIENT FILES · {label} · "
             f"{args.workers} worker{'s' if args.workers != 1 else ''}")
    display = Display(targets, title, live=True, proj_save=proj_save)

    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to encode without a terminal; pass --yes.", file=sys.stderr)
            return 2
        display.print_static()
        try:
            ans = input(f"\nRe-encode these {len(targets)} files with {args.workers} worker(s)? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 0
        print()
    elif not display.live:
        display.print_static()

    runner = Runner(args, template, state, display)
    t0 = time.monotonic()
    pool = cf.ThreadPoolExecutor(max_workers=args.workers)
    pending = {pool.submit(runner.run, e) for e in targets}
    results: list[Result] = []
    try:
        display.draw()
        while pending:
            finished, pending = cf.wait(pending, timeout=0.25, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                if r.status == "cancelled":
                    continue
                results.append(r)
                display.finish(r)
                state.save()
            display.draw()
    except KeyboardInterrupt:
        runner.kill_all()
        pool.shutdown(wait=True, cancel_futures=True)
        display.close()
        state.save()
        print("\nInterrupted - encoders stopped, partial files removed.", file=sys.stderr)
        return 130
    pool.shutdown()
    display.close()
    if not display.live:
        print()
        display.print_static()

    replaced = [r for r in results if r.status == "replaced"]
    saved = sum(r.entry.size - r.new_size for r in replaced)
    failed = sum(r.status == "failed" for r in results)
    print(f"\nDone in {fmt_time(time.monotonic() - t0)}: {len(replaced)} replaced, "
          f"{sum(r.status == 'kept' for r in results)} kept, {failed} failed. "
          f"Saved {saved / MIB:.1f} MiB.")
    if replaced and not args.delete_originals:
        print(f"Originals are in ./{BACKUP_DIR}/ - delete them once you're happy.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
