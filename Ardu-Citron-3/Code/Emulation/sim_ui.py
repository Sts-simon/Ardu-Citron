#!/usr/bin/env python3
"""Native desktop interface for configuring and running the flight simulator."""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
import threading
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ModuleNotFoundError:  # e.g. minimal Linux/Python installs without python3-tk
    tk = None
    messagebox = scrolledtext = ttk = None

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "aircraft.json"
SIMULATOR = ROOT / "simulator.py"
OUTPUT_DIR = ROOT / "output-ui"
MAX_DURATION_S = 300.0


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} doit être un nombre fini")
    return float(value)


def validate_config(config: Any, duration: Any) -> tuple[dict, float]:
    if not isinstance(config, dict):
        raise ValueError("La configuration doit être un objet JSON")
    for section in ("simulation", "aircraft", "camera", "sensors", "scenarios"):
        if section not in config:
            raise ValueError(f"Section de configuration manquante : {section}")
    if not isinstance(config["simulation"], dict):
        raise ValueError("simulation doit être un objet")
    dt = _finite_number(config["simulation"].get("dt_s"), "simulation.dt_s")
    seconds = _finite_number(duration, "durée")
    if dt <= 0 or dt > 0.1:
        raise ValueError("simulation.dt_s doit être > 0 et ≤ 0,1 s")
    if seconds <= 0 or seconds > MAX_DURATION_S:
        raise ValueError(f"La durée doit être > 0 et ≤ {MAX_DURATION_S:g} s")
    for section in ("aircraft", "camera", "sensors"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} doit être un objet")
    if not isinstance(config["scenarios"], list):
        raise ValueError("scenarios doit être une liste")
    for index, scenario in enumerate(config["scenarios"]):
        if not isinstance(scenario, dict) or scenario.get("type") != "rc_command":
            raise ValueError(f"scenarios[{index}] doit être un scénario rc_command")
        for key in ("time_s", "roll_deg", "pitch_deg", "yaw_rate_dps"):
            _finite_number(scenario.get(key), f"scenarios[{index}].{key}")
    for section, key in (("sensors", "imu_dropout_probability"),
                         ("sensors", "rc_dropout_probability"),
                         ("sensors", "uart_corruption_probability"),
                         ("camera", "drop_probability")):
        value = _finite_number(config[section].get(key), f"{section}.{key}")
        if not 0 <= value <= 1:
            raise ValueError(f"{section}.{key} doit être compris entre 0 et 1")
    config["simulation"]["duration_s"] = seconds
    return config, seconds


class SimulationUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ardu-Citron — Simulation")
        self.root.geometry("1120x780")
        self.root.minsize(760, 560)
        self.root.configure(bg="#101820")
        self._style()
        self._build()
        self.load_defaults()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#101820")
        style.configure("Card.TFrame", background="#182630")
        style.configure("TLabel", background="#182630", foreground="#edf4f6", font=("TkDefaultFont", 10))
        style.configure("Title.TLabel", background="#101820", foreground="#edf4f6", font=("TkDefaultFont", 22, "bold"))
        style.configure("Hint.TLabel", background="#101820", foreground="#a7bbc2", font=("TkDefaultFont", 10))
        style.configure("TButton", padding=(12, 8), font=("TkDefaultFont", 10, "bold"))
        style.configure("Accent.TButton", background="#61d6a5", foreground="#092018")
        style.map("Accent.TButton", background=[("active", "#83e5bb"), ("disabled", "#52786a")])
        style.configure("TEntry", padding=6)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(20, 18, 20, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Simulation Ardu-Citron", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Réglez les paramètres, lancez le modèle 6-DOF et consultez les résultats.",
                  style="Hint.TLabel").pack(anchor="w", pady=(5, 0))

        config_card = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        config_card.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(4, 20))
        config_card.columnconfigure(0, weight=1)
        config_card.rowconfigure(1, weight=1)
        ttk.Label(config_card, text="Paramètres de simulation", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        controls = ttk.Frame(config_card, style="Card.TFrame")
        controls.grid(row=0, column=1, sticky="e", pady=(0, 8))
        ttk.Label(controls, text="Durée (s)").pack(side="left", padx=(0, 6))
        self.duration = tk.StringVar(value="20")
        self.duration_entry = ttk.Entry(controls, textvariable=self.duration, width=8)
        self.duration_entry.pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="Défauts", command=self.load_defaults).pack(side="left")

        self.config_text = scrolledtext.ScrolledText(
            config_card, wrap="none", undo=True, font=("TkFixedFont", 10),
            background="#0d171e", foreground="#edf4f6", insertbackground="#edf4f6",
            selectbackground="#355467", borderwidth=0, padx=10, pady=10)
        self.config_text.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Label(config_card, text="Configuration complète en JSON. Autopilot.enabled reste false par défaut.",
                  foreground="#a7bbc2").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.run_button = ttk.Button(config_card, text="Lancer la simulation", style="Accent.TButton",
                                     command=self.start_simulation)
        self.run_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        results = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        results.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(4, 20))
        results.columnconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)
        ttk.Label(results, text="Exécution et résultats", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.status = tk.StringVar(value="Prêt. Vérifiez la configuration avant de lancer.")
        self.status_label = ttk.Label(results, textvariable=self.status, wraplength=390, justify="left")
        self.status_label.grid(row=1, column=0, sticky="new")
        self.files = ttk.Frame(results, style="Card.TFrame")
        self.files.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.preview = tk.Label(results, bg="#0d171e", fg="#a7bbc2", text="Le graphique apparaîtra ici",
                                anchor="center", height=12)
        self.preview.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.disclaimer = ttk.Label(results, text="SIL uniquement : aucun matériel piloté. Paramètres provisoires, pas une validation de vol.",
                                    foreground="#f4cf76", wraplength=390, justify="left")
        self.disclaimer.grid(row=4, column=0, sticky="w", pady=(12, 0))

    def load_defaults(self) -> None:
        try:
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Configuration", f"Impossible de charger aircraft.json :\n{exc}")
            return
        self.config_text.delete("1.0", "end")
        self.config_text.insert("1.0", json.dumps(config, indent=2, ensure_ascii=False))
        self.duration.set(str(config.get("simulation", {}).get("duration_s", 20)))
        self.status.set("Prêt. Modifiez les paramètres puis lancez la simulation.")

    def start_simulation(self) -> None:
        try:
            config = json.loads(self.config_text.get("1.0", "end"))
            config, duration = validate_config(config, self.duration.get())
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            messagebox.showerror("Paramètres invalides", str(exc))
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config_path = OUTPUT_DIR / "config-ui.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        run_dir = OUTPUT_DIR / "latest"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(SIMULATOR), "--config", str(config_path),
                   "--duration", str(duration), "--out", str(run_dir)]
        self.run_button.configure(state="disabled")
        self.status.set("Simulation en cours…")
        self.preview.configure(image="", text="Simulation en cours…")
        threading.Thread(target=self._run_process, args=(command, run_dir), daemon=True).start()

    def _run_process(self, command: list[str], run_dir: pathlib.Path) -> None:
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=MAX_DURATION_S + 120, check=False)
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or f"Le simulateur a échoué (code {result.returncode})."
                self.root.after(0, self._finish, False, message, run_dir)
            else:
                self.root.after(0, self._finish, True, result.stdout.strip(), run_dir)
        except subprocess.TimeoutExpired:
            self.root.after(0, self._finish, False, "Délai maximal du simulateur dépassé.", run_dir)
        except Exception as exc:
            self.root.after(0, self._finish, False, str(exc), run_dir)

    def _finish(self, succeeded: bool, message: str, run_dir: pathlib.Path) -> None:
        self.run_button.configure(state="normal")
        self.status.set(message or ("Simulation terminée." if succeeded else "Échec de la simulation."))
        for widget in self.files.winfo_children():
            widget.destroy()
        if succeeded:
            for filename in ("flight_log.csv", "summary.json", "flight_summary.png"):
                path = run_dir / filename
                if path.is_file():
                    ttk.Label(self.files, text=str(path), foreground="#61d6a5", wraplength=390).pack(anchor="w", pady=2)
            image_path = run_dir / "flight_summary.png"
            if image_path.is_file():
                try:
                    self._preview_image = tk.PhotoImage(file=str(image_path))
                    max_width = max(240, self.preview.winfo_width())
                    factor = max(1, math.ceil(self._preview_image.width() / max_width))
                    if factor > 1:
                        self._preview_image = self._preview_image.subsample(factor, factor)
                    self.preview.configure(image=self._preview_image, text="")
                except tk.TclError:
                    self.preview.configure(image="", text=f"Courbe créée : {image_path}")


def main() -> int:
    if tk is None:
        raise SystemExit("Tkinter absent. Installez le paquet Tk de votre Python (souvent python3-tk), puis relancez.")
    root = tk.Tk()
    SimulationUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
