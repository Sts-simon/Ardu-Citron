"""Compact 6-DOF fixed-wing model using NED world and forward/right/down body axes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

G = 9.80665


def rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


@dataclass
class AircraftState:
    position_ned_m: np.ndarray
    velocity_body_m_s: np.ndarray
    attitude_rad: np.ndarray
    rates_body_rad_s: np.ndarray
    servo_pulses_us: np.ndarray
    grounded: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AircraftState":
        return cls(
            np.array(config["initial_position_ned_m"], dtype=float),
            np.array(config["initial_velocity_body_m_s"], dtype=float),
            np.radians(np.array(config["initial_attitude_deg"], dtype=float)),
            np.zeros(3, dtype=float),
            np.full(3, 1500.0, dtype=float),
        )


class Glider6DOF:
    """Rigid body with quasi-steady coefficient aerodynamics, no propeller model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.aircraft = config["aircraft"]
        self.aero = self.aircraft["aero"]
        self.actuator = self.aircraft["actuators"]
        inertia = self.aircraft["inertia_kg_m2"]
        self.mass = float(self.aircraft["mass_kg"])
        self.inertia = np.array([inertia["ixx"], inertia["iyy"], inertia["izz"]], dtype=float)
        self.area = float(self.aircraft["wing_area_m2"])
        self.span = float(self.aircraft["wing_span_m"])
        self.chord = float(self.aircraft["mean_chord_m"])
        self.rho = float(self.aircraft["air_density_kg_m3"])
        self.gravity = float(self.aircraft.get("gravity_m_s2", G))
        self.wind = np.array(self.aircraft["wind_ned_m_s"], dtype=float)
        self.ixz = float(inertia.get("ixz", 0.0))
        self.thrust_n = float(self.aircraft.get("motor_thrust_n", 0.0))
        if self.actuator.get("engine_enabled", False) and self.thrust_n > 0.0:
            self.thrust_n = float(self.aircraft.get("motor_thrust_n", 0.0))
        else:
            self.thrust_n = 0.0
        self.state = AircraftState.from_config(self.aircraft)
        self.last: dict[str, float | np.ndarray] = {}

    def air_data(self) -> tuple[float, float, float, np.ndarray]:
        s = self.state
        relative_ned = rotation_body_to_ned(*s.attitude_rad) @ s.velocity_body_m_s - self.wind
        relative_body = rotation_body_to_ned(*s.attitude_rad).T @ relative_ned
        u, v, w = relative_body
        speed = max(float(np.linalg.norm(relative_body)), 0.01)
        alpha = math.atan2(w, max(u, 0.01))
        beta = math.asin(float(np.clip(v / speed, -0.99, 0.99)))
        return speed, alpha, beta, relative_body

    def derivatives(self, servo_pulses_us: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        s = self.state
        p, q, r = s.rates_body_rad_s
        speed, alpha, beta, relative_body = self.air_data()
        qbar = 0.5 * self.rho * speed * speed

        # Aileron/rudder/pitch linkage is provisional: three independently
        # normalized servo channels, mapped to the aerodynamic coefficient model.
        servo_deflection = np.radians(np.clip((servo_pulses_us - 1500.0) / 50.0, -10.0, 10.0))
        aileron, elevator, rudder = servo_deflection
        a = self.aero
        cl = float(np.clip(a["cl0"] + a["cl_alpha_per_rad"] * alpha + a["control_pitch_per_rad"] * elevator,
                           -a["cl_max"], a["cl_max"]))
        cd = a["cd0"] + a["induced_drag_k"] * cl * cl
        lift = qbar * self.area * cl
        drag = qbar * self.area * cd
        # Body Z points down (NED convention), so lift acts toward -Z.
        # Positive alpha is a downward relative-wind component in body axes.
        # Resolve wind-axis drag opposite the relative flow and lift
        # perpendicular to it into the forward/right/down body frame.
        fx = -drag * math.cos(alpha) + lift * math.sin(alpha) + self.thrust_n
        fy = qbar * self.area * (a["cy_beta_per_rad"] * beta + a["control_yaw_per_rad"] * rudder)
        fz = -drag * math.sin(alpha) - lift * math.cos(alpha)

        phat = p * self.span / (2 * speed)
        qhat = q * self.chord / (2 * speed)
        rhat = r * self.span / (2 * speed)
        roll_moment = qbar * self.area * self.span * (
            a["cl_roll_per_rad"] * aileron - 0.45 * phat + 0.03 * beta)
        pitch_moment = qbar * self.area * self.chord * (
            a["cm0"] + a["cm_alpha_per_rad"] * alpha + a["cm_pitch_rate"] * qhat
            + a["control_pitch_per_rad"] * elevator)
        yaw_moment = qbar * self.area * self.span * (
            a["cn_beta_per_rad"] * beta + a["cn_yaw_rate"] * rhat
            + a["control_yaw_per_rad"] * rudder)

        phi, theta, _ = s.attitude_rad
        gravity_body = np.array([
            -self.gravity * math.sin(theta),
            self.gravity * math.sin(phi) * math.cos(theta),
            self.gravity * math.cos(phi) * math.cos(theta),
        ])
        u, v, w = s.velocity_body_m_s
        udot = fx / self.mass + r * v - q * w + gravity_body[0]
        vdot = fy / self.mass + p * w - r * u + gravity_body[1]
        wdot = fz / self.mass + q * u - p * v + gravity_body[2]

        ix, iy, iz = self.inertia
        inertia_tensor = np.array([[ix, 0.0, -self.ixz], [0.0, iy, 0.0], [-self.ixz, 0.0, iz]])
        omega = s.rates_body_rad_s
        angular_accel = np.linalg.solve(
            inertia_tensor,
            np.array([roll_moment, pitch_moment, yaw_moment]) - np.cross(omega, inertia_tensor @ omega),
        )
        pdot, qdot, rdot = angular_accel

        cos_theta = math.copysign(max(abs(math.cos(theta)), 0.05), math.cos(theta))
        tan_theta = math.sin(theta) / cos_theta
        phidot = p + math.sin(phi) * tan_theta * q + math.cos(phi) * tan_theta * r
        thetadot = math.cos(phi) * q - math.sin(phi) * r
        psidot = (math.sin(phi) * q + math.cos(phi) * r) / cos_theta
        velocity_ned = rotation_body_to_ned(*s.attitude_rad) @ s.velocity_body_m_s

        detail = {
            "airspeed_m_s": speed, "alpha_rad": alpha, "beta_rad": beta,
            "dynamic_pressure_pa": qbar, "lift_n": lift, "drag_n": drag,
            "force_body_n": np.array([fx, fy, fz]),
            "moment_body_nm": np.array([roll_moment, pitch_moment, yaw_moment]),
            "accel_body_m_s2": np.array([fx / self.mass, fy / self.mass, fz / self.mass]),
            "velocity_ned_m_s": velocity_ned,
            "servo_deflection_rad": servo_deflection,
        }
        return (np.array([udot, vdot, wdot]), np.array([pdot, qdot, rdot]),
                np.array([phidot, thetadot, psidot]), velocity_ned, detail)

    def step(self, servo_pulses_us: np.ndarray, dt: float) -> dict:
        if self.state.grounded:
            return self.last
        # RK4 integration of the coupled 6-DOF equations with frozen controls.
        original = self.state

        def evaluate(delta: float):
            return self.derivatives(servo_pulses_us)

        k1raw = evaluate(0.0)
        k1 = (k1raw[3], k1raw[0], k1raw[2], k1raw[1], k1raw[4])
        def displaced(k, factor):
            self.state = AircraftState(
                original.position_ned_m + k[0] * factor,
                original.velocity_body_m_s + k[1] * factor,
                original.attitude_rad + k[2] * factor,
                original.rates_body_rad_s + k[3] * factor,
                servo_pulses_us.copy(),
            )
        displaced(k1, dt * 0.5); k2raw = self.derivatives(servo_pulses_us)
        k2 = (rotation_body_to_ned(*self.state.attitude_rad) @ self.state.velocity_body_m_s,
              k2raw[0], k2raw[2], k2raw[1], k2raw[4])
        displaced(k2, dt * 0.5); k3raw = self.derivatives(servo_pulses_us)
        k3 = (rotation_body_to_ned(*self.state.attitude_rad) @ self.state.velocity_body_m_s,
              k3raw[0], k3raw[2], k3raw[1], k3raw[4])
        displaced(k3, dt); k4raw = self.derivatives(servo_pulses_us)
        k4 = (rotation_body_to_ned(*self.state.attitude_rad) @ self.state.velocity_body_m_s,
              k4raw[0], k4raw[2], k4raw[1], k4raw[4])

        self.state = AircraftState(
            original.position_ned_m + dt / 6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
            original.velocity_body_m_s + dt / 6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
            original.attitude_rad + dt / 6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]),
            original.rates_body_rad_s + dt / 6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]),
            servo_pulses_us.copy(),
        )
        self.state.attitude_rad[0] = (self.state.attitude_rad[0] + math.pi) % (2*math.pi) - math.pi
        self.state.attitude_rad[2] = (self.state.attitude_rad[2] + math.pi) % (2*math.pi) - math.pi
        if self.state.position_ned_m[2] >= 0.0:
            self.state.position_ned_m[2] = 0.0
            self.state.grounded = True

        _, _, _, velocity_ned, detail = self.derivatives(servo_pulses_us)
        detail["velocity_ned_m_s"] = velocity_ned
        self.last = detail
        return detail
