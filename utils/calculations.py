"""
utils/calculations.py
----------------------
Rule-based calculations for the Single-Leg Terrain-Adaptive Landing Gear.

These functions turn raw (simulated, or later real ESP32) sensor readings
into human-useful numbers:

    - stability_score()   -> 0-100, how stable the leg/body currently is
    - landing_risk()      -> "Low" / "Medium" / "High"
    - recommended_action()-> plain-English suggestion for the leg controller
    - leg_geometry()      -> simple 2D coordinates for visualizing the leg

Everything here is intentionally simple and rule-based (no ML) so it is
easy to tune by hand once real hardware data is available. The ML model
(in ml/) is only used for terrain *classification*, not for stability or
risk scoring.

Backward compatibility note: `stability_score`, `landing_risk`, and
`recommended_action` all accept new OPTIONAL keyword arguments (leg_angle,
foot_contact, knee_angle) that let them react to the leg's physical state.
Calling them the old way (without these extra arguments) still works
exactly as before, so existing callers and tests are unaffected.
"""

from __future__ import annotations
import math
import numpy as np


def stability_score(
    pitch: float,
    roll: float,
    vibration: float,
    impact: float,
    leg_angle: float | None = None,
    foot_contact: bool | None = None,
    knee_angle: float | None = None,
) -> float:
    """
    Compute a 0-100 stability score from body pitch/roll, vibration and impact,
    optionally refined by leg angle, foot contact, and knee angle.

    Higher score = more stable.
    - Large pitch/roll angles (tilting) reduce stability.
    - High vibration (noisy accelerometer signal) reduces stability.
    - High impact force reduces stability sharply.
    - A leg angle far from the neutral ~90° stance reduces stability (optional).
    - No foot contact means the foot isn't planted yet -- a meaningful
      stability penalty (optional; only applied if foot_contact is passed).
    - A sharply bent knee (well below ~150°) reduces structural stability (optional).

    Returns
    -------
    float
        Stability score clipped to [0, 100].
    """
    # Tilt penalty: the further from level (0 deg), the worse.
    tilt_penalty = (abs(pitch) + abs(roll)) * 1.5

    # Vibration penalty: shaky readings mean unstable footing.
    vibration_penalty = vibration * 6.0

    # Impact penalty: sudden shocks hurt stability the most.
    impact_penalty = impact * 3.0

    score = 100.0 - tilt_penalty - vibration_penalty - impact_penalty

    if leg_angle is not None:
        # Deviation from neutral upright stance (90°) reduces margin for error.
        score -= abs(leg_angle - 90.0) * 0.3

    if foot_contact is not None and not foot_contact:
        # Foot not planted -- inherently less stable regardless of other factors.
        score -= 25.0

    if knee_angle is not None:
        # Very bent knee (small knee_angle) means less structural stability.
        knee_deviation = max(0.0, 150.0 - knee_angle)
        score -= knee_deviation * 0.15

    return float(np.clip(score, 0, 100))


def landing_risk(stability: float, impact: float, foot_contact: bool | None = None) -> str:
    """
    Classify landing risk from the stability score and impact magnitude,
    optionally escalated when the foot hasn't confirmed ground contact yet.

    Returns one of: "Low", "Medium", "High".
    """
    if foot_contact is False:
        # Without confirmed contact we can't trust the leg has settled,
        # so never report "Low" risk even if the current numbers look good.
        return "High" if stability < 60 else "Medium"

    if stability >= 75 and impact < 8:
        return "Low"
    if stability >= 45 and impact < 14:
        return "Medium"
    return "High"


def recommended_action(
    risk: str,
    leg_angle: float,
    terrain: str,
    foot_contact: bool | None = None,
    knee_angle: float | None = None,
) -> str:
    """
    Suggest a plain-English action for the leg controller based on risk
    level, current leg angle, detected terrain type, and (optionally)
    foot contact / knee angle for more specific guidance.
    """
    if foot_contact is False:
        return (
            f"Foot not yet in contact — continue adjusting leg angle toward "
            f"{leg_angle:.1f}° and knee flex to reach the terrain."
        )

    if risk == "Low":
        return "Maintain current leg and knee position — conditions stable."

    if risk == "Medium":
        if terrain in ("Sloped", "Uneven"):
            return f"Fine-tune leg angle toward {leg_angle + 5:.1f}° to better conform to terrain."
        return "Reduce descent speed and monitor vibration."

    # High risk
    if terrain == "Rocky":
        return "Extend leg for ground clearance, bend knee to absorb shock, and brace for impact."
    if terrain == "Sloped":
        return "Increase leg angle sharply to level the body before contact."
    return "Abort/slow descent — high landing risk detected."


def compute_all(row: dict) -> dict:
    """
    Convenience wrapper: given a dict of sensor readings (one row), compute
    stability, risk and recommended action, and return them merged into a
    new dict alongside the original values.

    Required keys in `row`: 'pitch', 'roll', 'vibration', 'impact',
    'leg_angle', 'terrain'. Optional keys used if present: 'foot_contact',
    'knee_angle'.
    """
    foot_contact = row.get("foot_contact")
    knee_angle = row.get("knee_angle")

    stability = stability_score(
        pitch=row["pitch"],
        roll=row["roll"],
        vibration=row["vibration"],
        impact=row["impact"],
        leg_angle=row.get("leg_angle"),
        foot_contact=foot_contact,
        knee_angle=knee_angle,
    )
    risk = landing_risk(stability, row["impact"], foot_contact=foot_contact)
    action = recommended_action(
        risk, row["leg_angle"], row["terrain"], foot_contact=foot_contact, knee_angle=knee_angle
    )

    return {
        **row,
        "stability_score": round(stability, 2),
        "landing_risk": risk,
        "recommended_action": action,
    }


def leg_geometry(leg_angle: float, knee_angle: float, l1: float = 1.0, l2: float = 0.8) -> dict:
    """
    Compute simple 2D forward-kinematics coordinates for the single-leg
    mechanism, for visualization purposes only (not physically exact).

    The hip (body attachment point) is fixed at (0, 0).

    Parameters
    ----------
    leg_angle : float
        Degrees. 90 = leg pointing straight down (neutral stance).
    knee_angle : float
        Degrees. 180 = fully extended (straight leg); smaller = more bent.
    l1, l2 : float
        Upper leg and lower leg segment lengths (same units as the plot).

    Returns
    -------
    dict with 'hip', 'knee', 'foot' -> each an (x, y) tuple.
    """
    upper_dir = math.radians(leg_angle - 90.0)
    hip = (0.0, 0.0)
    knee = (hip[0] + l1 * math.sin(upper_dir), hip[1] - l1 * math.cos(upper_dir))

    lower_dir = upper_dir + math.radians(180.0 - knee_angle)
    foot = (knee[0] + l2 * math.sin(lower_dir), knee[1] - l2 * math.cos(lower_dir))

    return {"hip": hip, "knee": knee, "foot": foot}