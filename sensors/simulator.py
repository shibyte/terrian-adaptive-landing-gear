"""
sensors/simulator.py
---------------------
Simulated MPU6050 (accelerometer + gyroscope) data generator for the
Single-Leg Terrain-Adaptive Landing Gear project.

Hardware (ESP32 + real MPU6050) isn't ready yet, so this module produces
realistic-looking fake sensor readings for each supported terrain type.

WHAT'S SIMULATED (v2 — passive terrain-adaptive mechanism)
------------------------------------------------------------
This version models the leg as a simple 2-joint (hip + knee) passive
mechanism reacting to a terrain profile, instead of drawing independent
random numbers each call. Every call to `.read()` advances one small time
step ("tick") of a mini physics simulation:

    1. The terrain itself evolves (height + slope) using a random walk
       with terrain-specific roughness ("terrain height/angle effects").
    2. The leg + knee angles are pulled toward a target angle that would
       let the foot conform to the current terrain, using a damped
       spring-mass model ("spring response" + "realistic movement").
    3. Foot contact is only True once the leg/knee have converged close
       enough to the terrain-driven target ("foot contact detection").
    4. Impact spikes at the moment of touchdown (contact False -> True)
       and then decays while the spring settles ("impact and vibration
       changes during contact").
    5. Body pitch/roll drift toward the terrain slope, and accel/gyro
       readings are derived from the actual joint velocities, so the
       raw IMU-style signals are consistent with the leg's motion.

DESIGN NOTE FOR FUTURE HARDWARE SWAP
-------------------------------------
This simulator exposes the same "shape" of reading that the real hardware
will eventually produce: a flat dict with keys like 'accel_x', 'gyro_y',
'pitch', 'leg_angle', 'foot_contact', etc. (plus new keys 'knee_angle',
'terrain_height', 'terrain_slope', 'touchdown_event').

When the ESP32 is ready, create a class such as:

    class ESP32SensorReader:
        def __init__(self, port, baudrate=115200): ...
        def read(self) -> dict: ...   # same keys as MPU6050Simulator.read()

...and the rest of the app (dashboard, ML model, calculations) will keep
working unchanged, because they only depend on the dict shape returned by
`.read()`, not on how the data was produced.
"""

from __future__ import annotations
import numpy as np

TERRAIN_TYPES = ["Flat", "Raised", "Sloped", "Uneven", "Rocky"]

# Leg segment lengths used both by the (rough) kinematics below and by the
# 2D dashboard visualization, so the two stay visually consistent.
LEG_L1 = 1.0  # upper leg (hip -> knee)
LEG_L2 = 0.8  # lower leg (knee -> foot)

# Simulation tick size, in arbitrary "seconds" for the internal spring model.
_DT = 0.2

# Per-terrain physical/behavioural profile. These numbers describe how the
# terrain itself moves (slope/height random walk) and how "difficult" it is
# for the leg to adapt (spring stiffness/damping, roughness, contact
# tolerance, chance of momentarily losing contact). Tune freely -- nothing
# else in the app needs to change.
_TERRAIN_PROFILES = {
    "Flat": dict(
        slope_target=0.0, slope_wander=0.3, height_target=0.0, height_wander=0.05,
        spike_prob=0.0, spike_scale=0.0,
        stiffness=8.0, damping=5.5, roughness=0.3,
        contact_tol=3.0, dropout_prob=0.0,
        accel_noise=0.05, gyro_noise=0.5,
    ),
    "Raised": dict(
        slope_target=0.0, slope_wander=0.5, height_target=1.1, height_wander=0.15,
        spike_prob=0.02, spike_scale=1.5,
        stiffness=6.0, damping=4.2, roughness=0.6,
        contact_tol=4.0, dropout_prob=0.03,
        accel_noise=0.08, gyro_noise=1.0,
    ),
    "Sloped": dict(
        slope_target=14.0, slope_wander=1.0, height_target=0.6, height_wander=0.2,
        spike_prob=0.01, spike_scale=1.2,
        stiffness=5.0, damping=3.5, roughness=0.8,
        contact_tol=5.0, dropout_prob=0.03,
        accel_noise=0.12, gyro_noise=2.0,
    ),
    "Uneven": dict(
        slope_target=0.0, slope_wander=3.0, height_target=0.0, height_wander=0.4,
        spike_prob=0.06, spike_scale=1.8,
        stiffness=4.0, damping=2.5, roughness=1.4,
        contact_tol=6.0, dropout_prob=0.12,
        accel_noise=0.18, gyro_noise=3.0,
    ),
    "Rocky": dict(
        slope_target=0.0, slope_wander=5.0, height_target=0.3, height_wander=0.8,
        spike_prob=0.15, spike_scale=2.5,
        stiffness=3.0, damping=1.6, roughness=2.4,
        contact_tol=8.0, dropout_prob=0.28,
        accel_noise=0.30, gyro_noise=5.0,
    ),
}


class MPU6050Simulator:
    """
    Generates one simulated sensor reading (row) per call to `.read()`.

    Unlike a purely random generator, this keeps internal state (terrain
    slope/height, leg angle, knee angle, and their velocities) so that
    consecutive readings form a smooth, physically-plausible trajectory --
    including a visible settling/adaptation period whenever the terrain
    changes (see `set_terrain`).

    Usage
    -----
        sim = MPU6050Simulator(terrain="Rocky", seed=42)
        row = sim.read()   # -> dict of sensor values
    """

    def __init__(self, terrain: str = "Flat", seed: int | None = None):
        self._rng = np.random.default_rng(seed)

        # Leg/knee state: start in a neutral standing pose so a brand new
        # simulator begins close to a sensible resting position.
        self._leg_angle = 90.0   # degrees; 90 = straight down (neutral)
        self._knee_angle = 170.0  # degrees; 180 = fully extended
        self._leg_vel = 0.0
        self._knee_vel = 0.0

        # Terrain state.
        self._slope = 0.0   # degrees
        self._height = 0.0  # arbitrary units

        # Dynamics bookkeeping.
        self._impact = 0.0
        self._prev_contact = False
        self._t = 0

        self.set_terrain(terrain)

    def set_terrain(self, terrain: str) -> None:
        """
        Switch the active terrain profile. Deliberately does NOT reset the
        leg/knee/terrain state -- this is what makes terrain changes show
        up as a smooth, spring-damped adaptation in `.read()` instead of an
        instant jump, mirroring how a passive mechanical leg would behave.
        """
        if terrain not in _TERRAIN_PROFILES:
            raise ValueError(f"Unknown terrain '{terrain}'. Choose from {TERRAIN_TYPES}")
        self.terrain = terrain
        self._profile = _TERRAIN_PROFILES[terrain]

    # ------------------------------------------------------------------
    # Internal simulation steps
    # ------------------------------------------------------------------
    def _step_terrain(self) -> None:
        """Evolve terrain slope/height one tick using a bounded random walk."""
        p = self._profile
        rng = self._rng

        slope_noise = rng.normal(0, p["slope_wander"])
        if p["spike_prob"] > 0 and rng.random() < p["spike_prob"]:
            slope_noise += rng.choice([-1.0, 1.0]) * p["slope_wander"] * p["spike_scale"] * rng.uniform(1.0, 2.0)

        self._slope += (p["slope_target"] - self._slope) * 0.1 + slope_noise
        self._slope = float(np.clip(self._slope, -35.0, 35.0))

        height_noise = rng.normal(0, p["height_wander"])
        self._height += (p["height_target"] - self._height) * 0.05 + self._slope * 0.01 + height_noise
        self._height = float(np.clip(self._height, -3.0, 5.0))

    def _target_angles(self) -> tuple[float, float]:
        """Compute the leg/knee angles that would let the foot conform to the current terrain."""
        target_leg = 90.0 + self._slope * 1.0
        target_leg = float(np.clip(target_leg, 40.0, 140.0))

        target_knee = 178.0 - abs(self._height) * 18.0 - abs(self._slope) * 0.6
        target_knee = float(np.clip(target_knee, 90.0, 180.0))
        return target_leg, target_knee

    def _step_leg(self, target_leg: float, target_knee: float) -> None:
        """Advance leg/knee angles one tick using a damped spring model (spring response)."""
        p = self._profile

        leg_accel = p["stiffness"] * (target_leg - self._leg_angle) - p["damping"] * self._leg_vel
        self._leg_vel += leg_accel * _DT
        self._leg_angle += self._leg_vel * _DT
        self._leg_angle = float(np.clip(self._leg_angle, 30.0, 150.0))

        knee_accel = p["stiffness"] * (target_knee - self._knee_angle) - p["damping"] * self._knee_vel
        self._knee_vel += knee_accel * _DT
        self._knee_angle += self._knee_vel * _DT
        self._knee_angle = float(np.clip(self._knee_angle, 80.0, 180.0))

    def _step_contact_and_shock(self, target_leg: float, target_knee: float) -> tuple[bool, bool, float, float]:
        """
        Decide foot contact for this tick, and derive impact/vibration from
        how close the leg is to its target and how fast the joints are moving.
        """
        p = self._profile
        rng = self._rng

        angle_err = abs(self._leg_angle - target_leg)
        knee_err = abs(self._knee_angle - target_knee)
        converged = (angle_err < p["contact_tol"]) and (knee_err < p["contact_tol"] * 1.5)
        dropout = p["dropout_prob"] > 0 and rng.random() < p["dropout_prob"]
        foot_contact = bool(converged and not dropout)

        touchdown_event = bool(foot_contact and not self._prev_contact)
        angular_speed = abs(self._leg_vel) + abs(self._knee_vel)

        if touchdown_event:
            self._impact = float(np.clip(angular_speed * 1.8 + p["roughness"] * rng.uniform(1.0, 3.0), 0, 20))
        elif foot_contact:
            # Spring settling: impact decays back toward a small residual.
            self._impact = max(0.0, self._impact * 0.55 + rng.normal(0, 0.15))
        else:
            # Airborne / still adjusting -- no ground shock yet.
            self._impact = max(0.0, rng.normal(0, 0.05))

        impact = float(np.clip(self._impact, 0, 20))
        vibration = float(np.clip(angular_speed * 1.0 + p["roughness"] * rng.uniform(0.1, 0.6), 0, 10))

        self._prev_contact = foot_contact
        return foot_contact, touchdown_event, impact, vibration

    def read(self) -> dict:
        """Advance the simulation by one tick and return the resulting sensor reading."""
        p = self._profile
        rng = self._rng
        self._t += 1

        self._step_terrain()
        target_leg, target_knee = self._target_angles()
        self._step_leg(target_leg, target_knee)
        foot_contact, touchdown_event, impact, vibration = self._step_contact_and_shock(target_leg, target_knee)

        # Body pitch slowly follows terrain slope (damped); roll is mostly
        # noise-driven since a single leg doesn't directly control roll.
        pitch_target = self._slope * 0.4
        pitch = getattr(self, "_pitch", 0.0)
        pitch += (pitch_target - pitch) * 0.15 + rng.normal(0, p["roughness"] * 0.5)
        pitch = float(np.clip(pitch, -30.0, 30.0))
        self._pitch = pitch

        roll = getattr(self, "_roll", 0.0)
        roll += -roll * 0.1 + rng.normal(0, p["roughness"] * 0.4)
        roll = float(np.clip(roll, -30.0, 30.0))
        self._roll = roll

        # Accel/gyro derived from actual joint motion + terrain-scaled noise,
        # so raw IMU-style signals stay consistent with the leg's behaviour.
        accel_x = rng.normal(0, p["accel_noise"]) + self._slope * 0.01
        accel_y = rng.normal(0, p["accel_noise"])
        accel_z = 9.81 + rng.normal(0, p["accel_noise"]) + (impact * 0.05 if foot_contact else 0.0)

        gyro_x = rng.normal(0, p["gyro_noise"]) + roll * 0.05
        gyro_y = rng.normal(0, p["gyro_noise"]) + self._leg_vel * 2.0
        gyro_z = rng.normal(0, p["gyro_noise"]) + self._knee_vel * 1.0

        return {
            "accel_x": round(float(accel_x), 4),
            "accel_y": round(float(accel_y), 4),
            "accel_z": round(float(accel_z), 4),
            "gyro_x": round(float(gyro_x), 4),
            "gyro_y": round(float(gyro_y), 4),
            "gyro_z": round(float(gyro_z), 4),
            "pitch": round(pitch, 3),
            "roll": round(roll, 3),
            "leg_angle": round(self._leg_angle, 3),
            "knee_angle": round(self._knee_angle, 3),
            "foot_contact": foot_contact,
            "touchdown_event": touchdown_event,
            "impact": round(impact, 3),
            "vibration": round(vibration, 3),
            "terrain_height": round(self._height, 3),
            "terrain_slope": round(self._slope, 3),
            "terrain": self.terrain,
        }

    def read_batch(self, n: int) -> list[dict]:
        """Convenience helper: return `n` simulated readings as a list of dicts."""
        return [self.read() for _ in range(n)]