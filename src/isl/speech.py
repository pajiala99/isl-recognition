"""Speech output: turn a recognised sign into spoken words.

Three backends, tried in order: the macOS `say` command (preferring an Indian-English
voice, which suits an ISL system), pyttsx3 if installed, and finally a silent fallback
that only prints - so the pipeline never crashes on a machine without a speech engine.

Speaking is non-blocking: recognition must not stall while a word is pronounced.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

# Indian-English voices first - an ISL translator should not sound American
PREFERRED_VOICES = ["Aman", "Rishi", "Veena", "Isha", "Daniel", "Samantha"]


def available_voice() -> str | None:
    """Pick the best installed macOS voice from the preference list."""
    if sys.platform != "darwin" or not shutil.which("say"):
        return None
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                                 timeout=5).stdout
    except Exception:
        return None
    installed = {line.split()[0] for line in listing.splitlines() if line.strip()}
    for v in PREFERRED_VOICES:
        if v in installed:
            return v
    return None


class Speaker:
    """Queued, de-duplicated text-to-speech.

    de-duplication matters: the live demo re-predicts every few frames, so the same
    word would otherwise be repeated several times a second while a sign is held.
    """

    def __init__(self, enabled: bool = True, voice: str | None = None,
                 rate: int = 170, min_gap: float = 1.2):
        self.enabled = enabled
        self.rate = rate
        self.min_gap = min_gap
        self.voice = voice or available_voice()
        self.backend = self._pick_backend()
        self._last_text, self._last_time = "", 0.0
        self._queue: Queue[str | None] = Queue()
        self._engine = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ backends
    def _pick_backend(self) -> str:
        if not self.enabled:
            return "off"
        if sys.platform == "darwin" and shutil.which("say"):
            return "say"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return "print"

    def _worker(self):
        while True:
            try:
                text = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if text is None:
                return
            try:
                self._utter(text)
            except Exception as e:                 # never let speech kill the demo
                print(f"[speech] {text} ({e})")

    def _utter(self, text: str):
        if self.backend == "say":
            cmd = ["say", "-r", str(self.rate)]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd + [text], check=False)
        elif self.backend == "pyttsx3":
            if self._engine is None:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
            self._engine.say(text)
            self._engine.runAndWait()
        else:
            print(f"[speech] {text}")

    # -------------------------------------------------------------------- public
    def say(self, text: str, force: bool = False) -> bool:
        """Speak `text`. Returns False if it was suppressed as a repeat."""
        if not self.enabled or not text:
            return False
        spoken = text.replace("_", " ")
        now = time.time()
        if not force and spoken == self._last_text and now - self._last_time < self.min_gap:
            return False
        self._last_text, self._last_time = spoken, now
        self._queue.put(spoken)
        return True

    def say_sentence(self, words) -> bool:
        """Speak an accumulated sentence, e.g. the words signed so far."""
        sentence = " ".join(w.replace("_", " ") for w in words).strip()
        if not sentence:
            return False
        return self.say(sentence, force=True)

    def describe(self) -> str:
        if self.backend == "say":
            return f"macOS say ({self.voice or 'default voice'})"
        return {"pyttsx3": "pyttsx3", "print": "no engine - printing only",
                "off": "disabled"}[self.backend]

    def close(self):
        self._queue.put(None)
