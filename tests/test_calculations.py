import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.calculations import stability_score, landing_risk, recommended_action, compute_all


def test_stability_score_perfect_conditions():
    score = stability_score(pitch=0, roll=0, vibration=0, impact=0)
    assert score == 100.0


def test_stability_score_bad_conditions_is_low():
    score = stability_score(pitch=20, roll=20, vibration=8, impact=15)
    assert score < 30


def test_stability_score_clipped_to_range():
    score_high = stability_score(pitch=0, roll=0, vibration=0, impact=0)
    score_low = stability_score(pitch=100, roll=100, vibration=100, impact=100)
    assert 0 <= score_high <= 100
    assert 0 <= score_low <= 100


def test_landing_risk_levels():
    assert landing_risk(stability=90, impact=1) == "Low"
    assert landing_risk(stability=60, impact=5) == "Medium"
    assert landing_risk(stability=10, impact=18) == "High"


def test_recommended_action_returns_string():
    action = recommended_action("High", leg_angle=90, terrain="Rocky")
    assert isinstance(action, str)
    assert len(action) > 0


def test_compute_all_merges_fields():
    row = {
        "pitch": 1, "roll": 1, "vibration": 0.2, "impact": 0.5,
        "leg_angle": 90, "terrain": "Flat",
    }
    result = compute_all(row)
    assert "stability_score" in result
    assert "landing_risk" in result
    assert "recommended_action" in result
    assert result["terrain"] == "Flat"
