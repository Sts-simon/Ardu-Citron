#!/usr/bin/env python3
"""Interface de bureau Tkinter pour configurer et lancer la simulation."""
from __future__ import annotations

import copy
import json
import math
import pathlib
import subprocess
import sys
import threading
from datetime import datetime
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ModuleNotFoundError:  # Certaines distributions Python omettent python3-tk.
    tk = None
    messagebox = scrolledtext = ttk = None

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "aircraft.json"
SIMULATOR = ROOT / "simulator.py"
OUTPUT_DIR = ROOT / "output-ui"
MAX_DURATION_S = 300.0

LABELS = {
    "duration_s": "Durée de la simulation (s)", "dt_s": "Pas de simulation (s)", "seed": "Graine aléatoire",
    "camera_period_s": "Période caméra (s)", "rc_period_s": "Période RC (s)",
    "imu_period_s": "Période IMU (s)", "control_period_s": "Période de contrôle (s)",
    "servo_period_s": "Période servo (s)", "uart_period_s": "Période UART (s)", "output_dir": "Dossier de sortie",
    "mass_kg": "Masse (kg)", "wing_span_m": "Envergure (m)", "wing_area_m2": "Surface alaire (m²)",
    "mean_chord_m": "Corde moyenne (m)", "aspect_ratio": "Allongement", "inertia_kg_m2": "Inertie (kg·m²)",
    "initial_position_ned_m": "Position initiale NED (m)", "initial_velocity_body_m_s": "Vitesse initiale corps (m/s)",
    "initial_attitude_deg": "Attitude initiale (°)", "gravity_m_s2": "Gravité (m/s²)",
    "air_density_kg_m3": "Densité de l’air (kg/m³)", "wind_ned_m_s": "Vent NED (m/s)", "motor_thrust_n": "Poussée (N)",
    "engine_enabled": "Moteur activé dans le modèle", "aero": "Coefficients aérodynamiques", "actuators": "Gouvernes / actionneurs",
    "cl0": "CL à portance nulle", "cl_alpha_per_rad": "Pente CL (1/rad)", "cl_max": "CL maximum",
    "cd0": "CD à portance nulle", "induced_drag_k": "Facteur de traînée induite", "cy_beta_per_rad": "CY β (1/rad)",
    "cl_roll_per_rad": "Cl roll (1/rad)", "cm0": "Cm0", "cm_alpha_per_rad": "Cm α (1/rad)",
    "cm_pitch_rate": "Cm taux de tangage", "cn_beta_per_rad": "Cn β (1/rad)", "cn_yaw_rate": "Cn taux de lacet",
    "control_roll_per_rad": "Gain commande roll", "control_pitch_per_rad": "Gain commande pitch", "control_yaw_per_rad": "Gain commande yaw",
    "max_surface_rad": "Débattement max (rad)", "roll_surface_m2": "Surface roll (m²)",
    "pitch_surface_m2": "Surface pitch (m²)", "yaw_surface_m2": "Surface yaw (m²)",
    "enabled": "Activer le guidage AUTO dans la simulation", "width_px": "Largeur (px)", "height_px": "Hauteur (px)",
    "horizontal_fov_deg": "Champ horizontal (°)", "mount_down_deg": "Inclinaison caméra (°)",
    "marker_size_m": "Taille du marqueur (m)", "marker_ids": "IDs de marqueurs", "marker_dictionary": "Dictionnaire ArUco",
    "camera_to_body": "Rotation caméra → corps", "camera_position_body_m": "Position caméra corps (m)",
    "max_reprojection_error_px": "Erreur reprojection max (px)", "exposure_s": "Exposition (s)",
    "readout_s": "Rolling shutter (s)", "distortion": "Distorsion optique", "noise_sigma_px": "Bruit image (px)",
    "drop_probability": "Probabilité perte caméra", "minimum_visible_fraction": "Fraction visible minimum",
    "gyro_noise_std_dps": "Bruit gyroscope (°/s)", "gyro_bias_dps": "Biais gyroscope (°/s)",
    "gyro_bias_walk_std_dps_sqrt_s": "Dérive biais gyro", "accel_noise_std_g": "Bruit accéléromètre (g)",
    "imu_dropout_probability": "Perte IMU (0–1)", "rc_dropout_probability": "Perte RC (0–1)",
    "uart_corruption_probability": "Corruption UART (0–1)", "use_imu": "Utiliser l’IMU pour la navigation",
    "use_vision": "Utiliser la vision", "use_gps": "Utiliser le GPS", "fix_timeout_s": "Expiration position (s)",
    "process_noise_m_s": "Bruit de propagation (m/s)", "vision_gain": "Gain vision", "gps_max_accuracy_m": "Précision GPS max (m)",
    "north_min_m": "Nord minimum (m)", "north_max_m": "Nord maximum (m)", "east_min_m": "Est minimum (m)",
    "east_max_m": "Est maximum (m)", "margin_m": "Marge de sécurité (m)", "markers": "Repères (JSON)",
    "gps_simulation": "GPS simulé", "period_s": "Période GPS (s)", "position_noise_std_m": "Bruit position GPS (m)",
    "accuracy_m": "Précision déclarée GPS (m)", "minimum_navigation_quality": "Qualité navigation minimum",
    "max_bank_deg": "Inclinaison max (°)", "max_yaw_rate_dps": "Taux de lacet max (°/s)",
    "lookahead_s": "Anticipation (s)", "waypoint_radius_m": "Rayon waypoint (m)",
    "target_altitude_m": "Altitude cible (m)", "altitude_gain_deg_per_m": "Gain altitude (°/m)",
    "scenarios": "Scénarios RC (JSON)", "time_s": "Temps (s)", "roll_deg": "Roll (°)",
    "pitch_deg": "Pitch (°)", "yaw_rate_dps": "Taux de lacet (°/s)", "arm": "Armer",
    "disarm": "Désarmer", "zone": "Zone de vol", "navigation": "Navigation", "camera": "Caméra",
    "sensors": "Capteurs simulés", "aircraft": "Avion", "simulation": "Simulation", "autopilot": "Guidage",
}

TABS = [
    ("Simulation", ("simulation",)),
    ("Avion", ("aircraft",)),
    ("Caméra", ("camera",)),
    ("Capteurs", ("sensors", "gps_simulation")),
    ("Navigation", ("navigation", "zone", "markers")),
    ("Guidage & RC", ("autopilot", "scenarios")),
]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} doit être un nombre fini")
    return float(value)


def validate_config(config: Any) -> dict:
    if not isinstance(config, dict):
        raise ValueError("La configuration doit être un objet JSON")
    for section in ("simulation", "aircraft", "camera", "sensors", "scenarios"):
        if section not in config:
            raise ValueError(f"Section manquante : {section}")
    sim = config["simulation"]
    if not isinstance(sim, dict):
        raise ValueError("simulation doit être un objet")
    duration = _finite_number(sim.get("duration_s"), "simulation.duration_s")
    dt = _finite_number(sim.get("dt_s"), "simulation.dt_s")
    if duration <= 0 or duration > MAX_DURATION_S:
        raise ValueError(f"La durée doit être comprise entre 0 et {MAX_DURATION_S:g} s")
    if dt <= 0 or dt > 0.1:
        raise ValueError("simulation.dt_s doit être > 0 et ≤ 0,1 s")
    for section in ("aircraft", "camera", "sensors", "navigation", "autopilot", "gps_simulation", "zone"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} doit être un objet")
    for section, key in (("sensors", "imu_dropout_probability"),
                         ("sensors", "rc_dropout_probability"),
                         ("sensors", "uart_corruption_probability"),
                         ("camera", "drop_probability")):
        value = _finite_number(config[section].get(key), f"{section}.{key}")
        if not 0 <= value <= 1:
            raise ValueError(f"{section}.{key} doit être compris entre 0 et 1")
    scenarios = config["scenarios"]
    if not isinstance(scenarios, list):
        raise ValueError("scenarios doit être une liste")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or scenario.get("type") != "rc_command":
            raise ValueError(f"scenarios[{index}] doit être un scénario rc_command")
        for key in ("time_s", "roll_deg", "pitch_deg", "yaw_rate_dps"):
            _finite_number(scenario.get(key), f"scenarios[{index}].{key}")
    if not isinstance(config.get("markers", []), list):
        raise ValueError("markers doit être une liste")
    return config


def set_path(config: dict, path: tuple[str, ...], value: Any) -> None:
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


class SimulationUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ardu-Citron — Simulation")
        self.root.geometry("1180x850")
        self.root.minsize(800, 620)
        self.root.configure(bg="#101820")
        self.widgets: list[tuple[tuple[str, ...], Any, Any, str]] = []
        self._running = False
        try:
            self._config_template = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Impossible de charger aircraft.json : {exc}") from exc
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
        style.configure("TEntry", padding=5)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Simulation Ardu-Citron", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Paramètres organisés par catégories · modèle logiciel uniquement",
                  style="Hint.TLabel").pack(anchor="w", pady=(4, 0))

        form = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        form.grid(row=1, column=0, sticky="nsew", padx=(18, 8), pady=(4, 16))
        form.columnconfigure(0, weight=1)
        form.rowconfigure(1, weight=1)
        bar = ttk.Frame(form, style="Card.TFrame")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="Réglages", font=("TkDefaultFont", 13, "bold")).pack(side="left")
        ttk.Button(bar, text="Recharger les valeurs par défaut", command=self.load_defaults).pack(side="right")
        self.tabs = ttk.Notebook(form)
        self.tabs.grid(row=1, column=0, sticky="nsew")
        self._build_tabs()
        ttk.Label(form, text="Les champs complexes (vecteurs, repères, scénarios) se modifient en JSON. AUTO reste désactivé par défaut.",
                  foreground="#a7bbc2", wraplength=760).grid(row=2, column=0, sticky="w", pady=(8, 0))

        results = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        results.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=(4, 16))
        results.columnconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)
        ttk.Label(results, text="Exécution et résultats", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.status = tk.StringVar(value="Prêt à simuler.")
        ttk.Label(results, textvariable=self.status, wraplength=370, justify="left").grid(
            row=1, column=0, sticky="new")
        self.files = ttk.Frame(results, style="Card.TFrame")
        self.files.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.preview = tk.Label(results, bg="#0d171e", fg="#a7bbc2", text="Le graphique apparaîtra ici",
                                anchor="center", height=12)
        self.preview.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(results, text="Aucune sortie matérielle. Les paramètres sont provisoires et ne valident pas un vol.",
                  foreground="#f4cf76", wraplength=370, justify="left").grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.run_button = ttk.Button(self.root, text="Lancer la simulation", style="Accent.TButton",
                                     command=self.start_simulation)
        self.run_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 16))

    def _build_tabs(self) -> None:
        for title, sections in TABS:
            canvas = tk.Canvas(self.tabs, background="#182630", highlightthickness=0)
            scrollbar = ttk.Scrollbar(self.tabs, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.tabs.add(canvas, text=title)
            self.tabs.rowconfigure(0, weight=1)
            self.tabs.columnconfigure(0, weight=1)
            content = ttk.Frame(canvas, style="Card.TFrame", padding=10)
            window = canvas.create_window((0, 0), window=content, anchor="nw")
            content.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda event, c=canvas, w=window: c.itemconfigure(w, width=event.width))
            for section in sections:
                self._add_object(content, (section,))

    def _value_at(self, path: tuple[str, ...]) -> Any:
        value: Any = self._config_template
        for key in path:
            value = value[key]
        return value

    def _add_object(self, parent: ttk.Frame, path: tuple[str, ...]) -> None:
        value = self._value_at(path)
        if isinstance(value, list):
            self._add_field(parent, path, path[-1], value)
            return
        title = LABELS.get(path[-1], path[-1].replace("_", " ").capitalize())
        group = ttk.LabelFrame(parent, text=title, padding=8)
        group.pack(fill="x", expand=True, pady=(2, 8))
        pending: list[tuple[str, Any]] = []

        def flush_scalars() -> None:
            while pending:
                row = ttk.Frame(group, style="Card.TFrame")
                row.pack(fill="x", expand=True)
                for key, item in pending[:2]:
                    self._add_scalar(row, path + (key,), key, item)
                del pending[:2]

        for key, item in value.items():
            item_path = path + (key,)
            if isinstance(item, dict):
                flush_scalars()
                self._add_object(group, item_path)
            elif isinstance(item, (list, tuple)):
                flush_scalars()
                self._add_field(group, item_path, key, item)
            else:
                pending.append((key, item))
        flush_scalars()

    def _add_scalar(self, parent: ttk.Frame, path: tuple[str, ...], key: str, default: Any) -> None:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        label = LABELS.get(key, key.replace("_", " ").capitalize())
        ttk.Label(cell, text=label, font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        if isinstance(default, bool):
            variable = tk.BooleanVar(value=default)
            ttk.Checkbutton(cell, text="Activé", variable=variable).pack(anchor="w", pady=(3, 0))
            kind, control = "bool", variable
        else:
            variable = tk.StringVar(value=str(default))
            ttk.Entry(cell, textvariable=variable).pack(fill="x", expand=True, pady=(3, 0))
            kind, control = "scalar", variable
        self.widgets.append((path, control, default, kind))

    def _add_field(self, parent: ttk.Frame, path: tuple[str, ...], key: str, default: Any) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", expand=True, pady=5)
        label = LABELS.get(key, key.replace("_", " ").capitalize())
        ttk.Label(frame, text=label, font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        widget = scrolledtext.ScrolledText(frame, height=6 if key == "scenarios" else 4,
                                           wrap="none", font=("TkFixedFont", 9),
                                           background="#0d171e", foreground="#edf4f6",
                                           insertbackground="#edf4f6", borderwidth=0)
        widget.pack(fill="x", expand=True, pady=(3, 0))
        widget.insert("1.0", json.dumps(default, indent=2, ensure_ascii=False))
        self.widgets.append((path, widget, default, "json"))

    def _read_form(self) -> dict:
        config = copy.deepcopy(self._config_template)
        for path, control, default, kind in self.widgets:
            if kind == "json":
                try:
                    value = json.loads(control.get("1.0", "end"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{'.'.join(path)} : JSON invalide à la ligne {exc.lineno}") from exc
            elif kind == "bool":
                value = bool(control.get())
            else:
                raw = str(control.get()).strip()
                if isinstance(default, bool):
                    value = raw.lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    value = int(raw)
                elif isinstance(default, float):
                    value = float(raw)
                else:
                    value = raw
            set_path(config, path, value)
        return validate_config(config)

    def load_defaults(self) -> None:
        try:
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Configuration", f"Impossible de charger aircraft.json :\n{exc}")
            return
        self._config_template = config
        for path, control, default, kind in self.widgets:
            value: Any = config
            for key in path:
                value = value[key]
            if kind == "json":
                control.delete("1.0", "end")
                control.insert("1.0", json.dumps(value, indent=2, ensure_ascii=False))
            else:
                control.set(value)
        self.status.set("Valeurs par défaut chargées. Réglez les catégories puis lancez.")

    def start_simulation(self) -> None:
        try:
            config = self._read_form()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror("Paramètres invalides", str(exc))
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = OUTPUT_DIR / datetime.now().strftime("run-%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        duration = config["simulation"]["duration_s"]
        command = [sys.executable, str(SIMULATOR), "--config", str(config_path),
                   "--duration", str(duration), "--out", str(run_dir)]
        self._running = True
        self.run_button.configure(state="disabled")
        self.status.set("Simulation en cours…")
        self.preview.configure(image="", text="Simulation en cours…")
        threading.Thread(target=self._run_process, args=(command, run_dir), daemon=True).start()

    def _run_process(self, command: list[str], run_dir: pathlib.Path) -> None:
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=MAX_DURATION_S + 120, check=False)
            success = result.returncode == 0
            message = result.stdout.strip() if success else (result.stderr.strip() or result.stdout.strip())
            self.root.after(0, self._finish, success, message or "Simulation terminée.", run_dir)
        except subprocess.TimeoutExpired:
            self.root.after(0, self._finish, False, "Délai maximal du simulateur dépassé.", run_dir)
        except Exception as exc:
            self.root.after(0, self._finish, False, str(exc), run_dir)

    def _finish(self, succeeded: bool, message: str, run_dir: pathlib.Path) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.status.set(message or ("Simulation terminée." if succeeded else "Échec de la simulation."))
        for widget in self.files.winfo_children():
            widget.destroy()
        if succeeded:
            for filename in ("flight_log.csv", "summary.json", "flight_summary.png"):
                path = run_dir / filename
                if path.is_file():
                    ttk.Label(self.files, text=str(path), foreground="#61d6a5", wraplength=370).pack(anchor="w", pady=2)
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
