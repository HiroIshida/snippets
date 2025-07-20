from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time
import tkinter as tk
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


class Msg(Enum):
    STATUS = auto()
    START_TIMER = auto()
    STOP_TIMER = auto()
    ENABLE_PROCEED = auto()
    RESET = auto()
    SHUTDOWN = auto()

    START_DEMO = auto()
    PROCEED_PRESSED = auto()


@dataclass
class Payload:
    tag: Msg
    text: str = ""
    color: str = "black"
    what: str = ""
    font: tuple[str, int, str] | None = None


_UPDATE_MS: Final = 10
_POLL_MS: Final = 50


class _RobotStatusGUI:
    def __init__(self, root: tk.Tk, to_gui: mp.Queue, to_main: mp.Queue):
        self._root = root
        self._to_gui = to_gui
        self._to_main = to_main

        self._make_widgets()
        self._start_time: float | None = None
        self._timer_after: str | None = None
        self._poll_queues()

    def _make_widgets(self) -> None:
        self._root.title("Robot Status Display")
        self._root.geometry("1200x800")
        self._root.configure(bg="black")

        main = tk.Frame(self._root, bg="white")
        main.pack(expand=True, fill="both")

        self._status = tk.Label(
            main, font=("Arial", 170, "bold"), fg="black",
            bg="white", wraplength=1300
        )
        self._status.pack(expand=True, pady=(150, 50))

        self._timer = tk.Label(
            main, font=("Arial", 120), fg="black", bg="white"
        )
        self._timer.pack(pady=20)

        btns = tk.Frame(main, bg="white")
        btns.pack(pady=40)

        self._start_btn = tk.Button(
            btns, text="Start", width=20, height=2, font=("Arial", 40, "bold"),
            command=self._on_start
        )
        self._start_btn.grid(row=0, column=0, padx=50)

        self._proceed_btn = tk.Button(
            btns, text="Proceed", width=20, height=2, font=("Arial", 40),
            state="disabled", command=self._on_proceed
        )
        self._proceed_btn.grid(row=0, column=1, padx=50)

        self._set_status("Ready")

    # ---------- button callbacks ----------
    def _on_start(self) -> None:
        self._to_main.put(Payload(Msg.START_DEMO))
        self._start_btn.config(state="disabled")

    def _on_proceed(self) -> None:
        self._to_main.put(Payload(Msg.PROCEED_PRESSED))
        self._proceed_btn.config(state="disabled")

    # ---------- queue polling ----------
    def _poll_queues(self) -> None:
        try:
            while True:
                msg: Payload = self._to_gui.get_nowait()
                self._handle(msg)
        except _queue.Empty:
            pass
        finally:
            self._root.after(_POLL_MS, self._poll_queues)

    def _handle(self, msg: Payload) -> None:
        if msg.tag is Msg.STATUS:
            self._set_status(msg.text, msg.color, msg.font)
        elif msg.tag is Msg.START_TIMER:
            self._start_timer()
        elif msg.tag is Msg.STOP_TIMER:
            self._stop_timer(msg.what)
        elif msg.tag is Msg.ENABLE_PROCEED:
            self._proceed_btn.config(state="normal")
        elif msg.tag is Msg.RESET:
            self._reset()
        elif msg.tag is Msg.SHUTDOWN:
            self._root.destroy()

    # ---------- helpers ----------
    def _set_status(self, text: str, color: str = "black", font: tuple[str, int, str] | None = None) -> None:
        config_args = {"text": text, "fg": color}
        if font is not None:
            config_args["font"] = font
        self._status.config(**config_args)

    # timer
    def _start_timer(self) -> None:
        self._stop_timer("")
        self._start_time = time.perf_counter()
        self._tick_timer()

    def _tick_timer(self) -> None:
        if self._start_time is None:
            return
        elapsed = time.perf_counter() - self._start_time
        self._timer.config(text=f"Elapsed: {elapsed:6.2f}s")
        self._timer_after = self._root.after(_UPDATE_MS, self._tick_timer)

    def _stop_timer(self, what: str) -> None:
        if self._timer_after:
            self._root.after_cancel(self._timer_after)
            self._timer_after = None
        if self._start_time is not None and what:
            elapsed = time.perf_counter() - self._start_time
            self._timer.config(text=f"{what} in {elapsed:5.2f}s")
        self._timer.config(text="")
        self._start_time = None

    def _reset(self) -> None:
        self._stop_timer("")
        self._set_status("Ready")
        self._timer.config(text="")
        self._start_btn.config(state="normal")
        self._proceed_btn.config(state="disabled")


def _gui_entry(to_gui: mp.Queue, to_main: mp.Queue) -> None:
    root = tk.Tk()
    _RobotStatusGUI(root, to_gui, to_main)
    root.mainloop()


class DisplayBase(ABC):
    @abstractmethod
    def status(self, text: str, *, color: str = "black", font: tuple[str, int, str] | None = None) -> None:
        pass

    @abstractmethod
    def start_timer(self) -> None:
        pass

    @abstractmethod
    def stop_timer(self, what: str = "") -> None:
        pass

    @abstractmethod
    def enable_proceed(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def wait_start(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    def wait_proceed(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> "DisplayBase":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        pass


class Display(DisplayBase):
    def __init__(self) -> None:
        self._to_gui: mp.Queue = mp.Queue()
        self._to_main: mp.Queue = mp.Queue()
        self._proc = mp.Process(
            target=_gui_entry, args=(self._to_gui, self._to_main), daemon=True
        )
        self._proc.start()

    # -------------   commands  -------------
    def status(self, text: str, *, color: str = "black", font: tuple[str, int, str] | None = None) -> None:
        self._to_gui.put(Payload(Msg.STATUS, text=text, color=color, font=font))

    def start_timer(self) -> None:
        self._to_gui.put(Payload(Msg.START_TIMER))

    def stop_timer(self, what: str = "") -> None:
        self._to_gui.put(Payload(Msg.STOP_TIMER, what=what))

    def enable_proceed(self) -> None:
        self._to_gui.put(Payload(Msg.ENABLE_PROCEED))

    def reset(self) -> None:
        self._to_gui.put(Payload(Msg.RESET))

    # -------------   waits  -------------
    def wait_start(self, timeout: float | None = None) -> None:
        self._wait_for(Msg.START_DEMO, timeout)

    def wait_proceed(self, timeout: float | None = None) -> None:
        self._wait_for(Msg.PROCEED_PRESSED, timeout)

    # -------------   housekeeping  -------------
    def shutdown(self) -> None:
        if self._proc.is_alive():
            self._to_gui.put(Payload(Msg.SHUTDOWN))
            self._proc.join()

    # -------------   context mgr  -------------
    def __enter__(self) -> "Display":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # -------------   internal  -------------
    def _wait_for(self, tag: Msg, timeout: float | None) -> None:
        while True:
            msg: Payload = self._to_main.get(timeout=timeout)
            if msg.tag is tag:
                return


class Headless(DisplayBase):
    def __init__(self) -> None:
        self._start_time: float | None = None
        self._proceed_enabled = False
        self._waiting_for_start = False
        self._waiting_for_proceed = False

    # -------------   commands  -------------
    def status(self, text: str, *, color: str = "black", font: tuple[str, int, str] | None = None) -> None:
        print(f"[STATUS] {text}")

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()
        print("[TIMER] Started")

    def stop_timer(self, what: str = "") -> None:
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            if what:
                print(f"[TIMER] {what} in {elapsed:5.2f}s")
            else:
                print(f"[TIMER] Stopped at {elapsed:5.2f}s")
        self._start_time = None

    def enable_proceed(self) -> None:
        self._proceed_enabled = True
        print("[INPUT] Press Enter to proceed...")

    def reset(self) -> None:
        self._start_time = None
        self._proceed_enabled = False
        self._waiting_for_start = False
        self._waiting_for_proceed = False
        print("[SYSTEM] Reset - Ready")

    # -------------   waits  -------------
    def wait_start(self, timeout: float | None = None) -> None:
        self._waiting_for_start = True
        print("[INPUT] Press Enter to start...")
        self._wait_for_input("start", timeout)
        self._waiting_for_start = False

    def wait_proceed(self, timeout: float | None = None) -> None:
        self._waiting_for_proceed = True
        if not self._proceed_enabled:
            print("[INPUT] Waiting for proceed to be enabled...")
            while not self._proceed_enabled:
                time.sleep(0.1)
        self._wait_for_input("proceed", timeout)
        self._waiting_for_proceed = False
        self._proceed_enabled = False

    # -------------   housekeeping  -------------
    def shutdown(self) -> None:
        print("[SYSTEM] Shutting down...")

    # -------------   context mgr  -------------
    def __enter__(self) -> "Headless":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # -------------   internal  -------------
    def _wait_for_input(self, action: str, timeout: float | None) -> None:
        start_time = time.perf_counter()
        while True:
            try:
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    input()
                    return
            except ImportError:
                time.sleep(0.1)
                if timeout and time.perf_counter() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for {action}")

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for {action}")
