"""Straight-walking terms of the velocity recipe (2026-10-29).

CPU only: builds cfgs and calls mdp functions on stub envs, never steps an env.
"""
import math

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velocity_env_cfg as vel
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg


# ── Cfg wiring ───────────────────────────────────────────────────────────────

def test_straight_walk_terms_wired():
    cfg = vel.make_microduck_velocity_env_cfg()
    fine = cfg.rewards["track_yaw_fine"]
    assert fine.func is microduck_mdp.track_yaw_rate_fine
    assert fine.weight > 0 and fine.params["std"] == vel.TRACK_YAW_FINE_STD
    drift = cfg.rewards["straight_walk_drift"]
    assert drift.func is microduck_mdp.straight_walk_drift_penalty
    assert drift.weight == 0.0  # starts off, ramped by curriculum


def test_drift_weight_ramp_is_positive_and_late():
    # Self-negating penalty → every stage weight ≥ 0 (negative would reward drift).
    cfg = vel.make_microduck_velocity_env_cfg()
    stages = cfg.curriculum["straight_walk_drift_weight"].params["weight_stages"]
    assert all(s["weight"] >= 0 for s in stages)
    assert stages[0]["weight"] == 0.0
    first_on = min(s["step"] for s in stages if s["weight"] > 0)
    assert first_on >= 1000 * vel.NUM_STEPS_PER_ENV  # only once a gait exists


def test_heading_envs_keep_rate_semantics():
    cfg = vel.make_microduck_velocity_env_cfg()
    tw = cfg.commands["twist"]
    assert tw.heading_command
    assert tw.rel_heading_envs == vel.HEADING_ENVS_FRACTION
    assert 0.0 < tw.rel_heading_envs < 0.5
    assert tw.ranges.heading == (-math.pi, math.pi)
    # Heading envs still emit a yaw RATE clipped to the trained wz range (no obs change).
    assert tw.ranges.ang_vel_z == (-1.0, 1.0)
    assert isinstance(tw, microduck_mdp.VelocityCommandCommandOnlyCfg)


def test_opt_out_and_velstand_unchanged():
    cfg = vel.make_microduck_velocity_env_cfg(straight_walk=False)
    assert "straight_walk_drift" not in cfg.rewards
    assert "straight_walk_drift_weight" not in cfg.curriculum
    assert cfg.commands["twist"].rel_heading_envs == 0.0
    vs = make_microduck_velstand_env_cfg()
    assert "straight_walk_drift" not in vs.rewards
    assert vs.commands["twist"].rel_heading_envs == 0.0


# ── straight_walk_drift_penalty ──────────────────────────────────────────────

class _Data:
    def __init__(self, vy, wz):
        n = len(vy)
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_lin_vel_b[:, 1] = torch.tensor(vy)
        self.root_link_ang_vel_b = torch.zeros(n, 3)
        self.root_link_ang_vel_b[:, 2] = torch.tensor(wz)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Cmds:
    def __init__(self, cmd):
        self.cmd = cmd

    def get_command(self, _name):
        return self.cmd


class _Env:
    def __init__(self, cmd, vy, wz):
        self._asset = _Asset(_Data(vy, wz))
        self.command_manager = _Cmds(torch.tensor(cmd, dtype=torch.float32))
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((len(vy),), 100)

    @property
    def scene(self):
        return {"robot": self._asset}


def _run(env, steps):
    out = None
    for _ in range(steps):
        out = microduck_mdp.straight_walk_drift_penalty(env, tau_s=2.0)
    return out


def test_constant_bias_is_billed():
    env = _Env([[0.3, 0.0, 0.0]], vy=[0.0], wz=[0.1])
    out = _run(env, 1000)  # 20 s ≫ tau: EMA converged to the bias
    assert out[0] < 0
    assert abs(float(out[0]) + 0.1) < 1e-3  # yaw_scale 1.0 · 0.1 rad/s


def test_oscillation_cancels():
    env = _Env([[0.3, 0.0, 0.0]], vy=[0.0], wz=[0.0])
    out = None
    for i in range(1000):  # ±0.3 rad/s sway at the 2-step gait period
        env._asset.data.root_link_ang_vel_b[0, 2] = 0.3 if (i // 10) % 2 == 0 else -0.3
        out = microduck_mdp.straight_walk_drift_penalty(env, tau_s=2.0)
    assert float(out[0]) > -0.02


def test_turning_envs_not_billed_and_ema_reset():
    env = _Env([[0.3, 0.0, 0.8]], vy=[0.0], wz=[0.2])  # badly tracked turn
    out = _run(env, 200)
    assert float(out[0]) == 0.0
    assert float(env._straight_drift_ema.abs().sum()) == 0.0


def test_lateral_drift_is_billed():
    env = _Env([[0.3, 0.0, 0.0]], vy=[0.05], wz=[0.0])
    out = _run(env, 1000)
    assert abs(float(out[0]) + 2.0 * 0.05) < 1e-3


def test_nan_safe():
    env = _Env([[0.3, 0.0, 0.0]], vy=[float("nan")], wz=[float("nan")])
    out = _run(env, 10)
    assert torch.isfinite(out).all()


def test_turn_in_place_clears_heading_flag():
    # A heading env would overwrite wz every step; turn-in-place frames must keep theirs.
    import inspect
    src = inspect.getsource(microduck_mdp.VelocityCommandCommandOnly._resample_command)
    assert "is_heading_env[turn_ids] = False" in src
