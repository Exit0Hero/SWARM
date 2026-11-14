"""Text renderer: exact output on a hand-built map, and read-only behaviour."""

from __future__ import annotations

from swarm.core.env import SwarmEnv
from swarm.core.types import Action, Layer
from swarm.scenarios.config import AgentSpec
from swarm.visualization.text import render_snapshot_text
from tests.helpers import config_for, spec

AIR = spec("air", Layer.AIR)
GND = spec("gnd", Layer.GROUND)


def _env() -> SwarmEnv:
    config, scenario = config_for(
        ["S.~", "B.#"],
        (AIR, GND),
        (AgentSpec("air"), AgentSpec("gnd"), AgentSpec("gnd", initial_energy=1)),
    )
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    env.step({"air_0": int(Action.EAST), "gnd_0": int(Action.EAST), "gnd_1": int(Action.NORTH)})
    return env


def test_exact_render_without_ground_truth() -> None:
    text = render_snapshot_text(_env().snapshot(), legend=False)
    assert text.splitlines() == [
        "step 1 running",
        "x.~",
        "B*#",
        "air_0      air    (1, 1)    energy 9/10 active",
        "gnd_0      ground (1, 1)    energy 9/10 active",
        "gnd_1      ground (0, 0)    energy 0/10 disabled",
    ]


def test_survivors_only_shown_on_request() -> None:
    env = _env()
    shown = render_snapshot_text(env.snapshot(), show_survivors=True, legend=False)
    # The survivor at (0, 0) is under the disabled robot, so agents take precedence there.
    assert shown.splitlines()[1] == "x.~"
    config, scenario = config_for(["SB"], (AIR,), (AgentSpec("air"),))
    lone = SwarmEnv(config)
    lone.reset(0, scenario=scenario)
    assert render_snapshot_text(lone.snapshot(), legend=False).splitlines()[1] == ".A"
    assert (
        render_snapshot_text(lone.snapshot(), show_survivors=True, legend=False).splitlines()[1]
        == "SA"
    )


def test_rendering_does_not_change_state() -> None:
    env = _env()
    before = env.snapshot()
    render_snapshot_text(before, show_survivors=True)
    after = env.snapshot()
    assert before.agents == after.agents
    assert before.step == after.step


def test_hazards_are_drawn_under_agents_and_survivors() -> None:
    config, scenario = config_for(
        ["B.S~"], (AIR,), (AgentSpec("air"),), hazard_cells=((0, 1), (0, 2), (0, 3))
    )
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    hidden = render_snapshot_text(env.snapshot(), legend=False).splitlines()[1]
    shown = render_snapshot_text(env.snapshot(), show_survivors=True, legend=False).splitlines()[1]
    assert hidden == "A!!!"
    assert shown == "A!S!"
