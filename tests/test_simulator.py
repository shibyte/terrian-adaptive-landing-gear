import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.simulator import MPU6050Simulator, TERRAIN_TYPES


def test_all_terrains_supported():
    for terrain in TERRAIN_TYPES:
        sim = MPU6050Simulator(terrain=terrain, seed=1)
        row = sim.read()
        assert row["terrain"] == terrain


def test_read_returns_expected_keys():
    sim = MPU6050Simulator(terrain="Flat", seed=1)
    row = sim.read()
    expected_keys = {
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "pitch", "roll", "leg_angle",
        "foot_contact", "impact", "vibration", "terrain",
    }
    assert expected_keys.issubset(row.keys())


def test_read_batch_length():
    sim = MPU6050Simulator(terrain="Rocky", seed=1)
    rows = sim.read_batch(10)
    assert len(rows) == 10


def test_invalid_terrain_raises():
    try:
        MPU6050Simulator(terrain="Space")
        assert False, "Expected ValueError for invalid terrain"
    except ValueError:
        pass


def test_foot_contact_is_bool():
    sim = MPU6050Simulator(terrain="Uneven", seed=1)
    row = sim.read()
    assert isinstance(row["foot_contact"], bool)
