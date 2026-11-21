"""The ``swarm`` command: one entry point over the existing tools.

Every subcommand delegates to an existing module. The older ``python -m swarm.<module>`` commands
keep working unchanged.

Exit codes:

- 0: success;
- 1: the command ran but something failed (an episode crashed, an experiment was incomplete, or
  the 3D check found contacts);
- 2: bad usage, configuration or input.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import swarm
from swarm.examples import NAMES as EXAMPLES
from swarm.examples import resolve_config_text
from swarm.scenarios.config import ConfigError, ScenarioConfig

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


class CliError(Exception):
    """A user-facing error: printed without a traceback, exit code 2."""


def _load(spec: str) -> ScenarioConfig:
    try:
        return ScenarioConfig.from_json(resolve_config_text(spec))
    except (FileNotFoundError, KeyError) as exc:
        raise CliError(str(exc).strip("'\"")) from exc
    except (ConfigError, ValueError) as exc:
        raise CliError(f"config {spec!r}: {exc}") from exc


def _policy_table(models: Path | None) -> dict[str, Any]:
    from swarm.dashboard.service import learned_catalog
    from swarm.policies.registry import POLICIES

    return {**POLICIES, **learned_catalog(models)}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary(result: Any, total: int) -> str:
    m = result.metrics
    found = round((m.recall or 0) * total)
    return (
        f"{result.policy['name']}, seed {result.seed}: {result.end_reason} at step {result.steps}"
        f" | survivors found {found}/{total} (recall {m.recall:.3f})"
        f" | robots disabled {m.agents_disabled} | energy used {m.energy_used}"
    )


# ----------------------------------------------------------------- subcommands


def cmd_validate(args: argparse.Namespace) -> int:
    from swarm.dashboard.service import policy_catalog

    config = _load(args.config)
    print(f"valid config, hash {config.config_hash()}")
    print(
        f"  map {config.height}x{config.width}, {len(config.roster)} robots, "
        f"{config.survivor_count} survivors, time limit {config.time_limit}"
    )
    for block in ("comms", "failures", "perception"):
        print(f"  {block}: {'yes' if getattr(config, block) is not None else 'no'}")
    usable = [
        p["name"] for p in policy_catalog(config, _policy_table(args.models)) if p["available"]
    ]
    print(f"  policies that can run it: {', '.join(usable)}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    from swarm.experiments.runner import run_episode, write_result

    config = _load(args.config)
    table = _policy_table(args.models)
    if args.policy not in table:
        raise CliError(f"unknown policy {args.policy!r}; see `swarm policies`")
    try:
        result = run_episode(config, table[args.policy], args.seed, designation="development")
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    print(_summary(result, config.survivor_count))
    if args.out is not None:
        write_result(result, args.out, args.events)
        print(
            f"result written to {args.out}" + (f", events to {args.events}" if args.events else "")
        )
    return EXIT_OK


def cmd_policies(args: argparse.Namespace) -> int:
    from swarm.dashboard.service import policy_catalog

    config = _load(args.config)
    for p in policy_catalog(config, _policy_table(args.models)):
        tag = " (ORACLE: uses hidden ground truth)" if p.get("oracle") else ""
        state = "" if p["available"] else f"  [unavailable: {p['reason']}]"
        print(f"{p['name']}{tag}{state}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    from swarm.dashboard.service import DashboardError, run_comparison
    from swarm.experiments.experiment import parse_seeds

    config = _load(args.config)
    try:
        seeds = list(parse_seeds(args.seeds))
        result = run_comparison(config, args.policies, seeds, _policy_table(args.models))
    except (DashboardError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    print(f"{len(args.policies)} policies x {len(seeds)} seeds (exploratory, development seeds)")
    print(f"{'policy':<24}{'recall':>18}{'success':>18}{'time to target':>22}")
    for p in args.policies:
        s = result["per_policy"][p]["summaries"]

        def cell(name: str, s: dict[str, Any] = s) -> str:
            v = s.get(name, {})
            return (
                "—"
                if v.get("mean") is None
                else f"{v['mean']:.2f} [{v['ci_low']:.2f},{v['ci_high']:.2f}]"
            )

        print(
            f"{p:<24}{cell('recall'):>18}{cell('mission_success'):>18}"
            f"{cell('time_to_recall_target'):>22}"
        )
    if args.out is not None:
        _write(args.out, json.dumps(result, indent=2))
        print(f"comparison written to {args.out}")
    failed = sum(v["failed"] for v in result["per_policy"].values())
    return EXIT_FAILED if failed else EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the packaged examples end to end and leave everything the dashboard needs."""
    from swarm.dashboard.service import (
        Store,
        export_run,
        record_episode,
        run_comparison,
        scene3d,
    )

    out: Path = args.out_dir
    store = Store(out / "dashboard")
    learned = _policy_table(args.models)
    print(f"SWARM {swarm.__version__} demo -> {out}/")
    t0 = time.perf_counter()
    for name, policy in (("basic", "auction_comm"), ("advanced", "auction_comm")):
        config = _load(name)
        record = record_episode(config, policy, args.seed, learned)
        store.save(record)
        for fmt, suffix in (
            ("json", "run.json"),
            ("events", "events.jsonl"),
            ("csv", "metrics.csv"),
        ):
            _write(out / f"{name}_{suffix}", export_run(record, fmt)[0])
        m, ext = record["metrics"], record["extension_metrics"]
        total = len(record["scenario"]["survivors_ground_truth"])
        print(
            f"\n[{name}] {policy}, seed {args.seed}: {record['end_reason']} at step "
            f"{record['steps']}; survivors found {round(m['recall'] * total)}/{total}; "
            f"robots disabled {m['agents_disabled']}"
        )
        print(
            f"  messages sent {ext.get('messages_sent', 0)}, delivery rate "
            f"{ext.get('message_delivery_rate') or 0:.2f}; perception errors "
            f"{sum(v for k, v in ext.items() if k.startswith('perception_') and 'report' not in k)}"
        )
        if name == "advanced":
            s3 = scene3d(record)
            a = s3["analysis"]
            _write(out / "advanced_scene3d.json", json.dumps(s3))
            if "unavailable" in a:
                print(f"  3D: scene written; MuJoCo analysis skipped ({a['unavailable']})")
            else:
                print(
                    f"  3D (kinematic, {a['engine']}): {a['contacts']} contacts, "
                    f"{a['reports_occluded']}/{a['reports_checked']} reports line-of-sight blocked"
                )
    seeds = list(range(args.seed, args.seed + args.compare_seeds))
    policies = ["frontier", "frontier_shared", "auction_comm"]
    comparison = run_comparison(_load("advanced"), policies, seeds, learned)
    store.save(comparison)
    _write(out / "advanced_comparison.json", json.dumps(comparison, indent=2))
    print(f"\n[comparison] advanced scenario, seeds {seeds[0]}-{seeds[-1]} (exploratory):")
    for p in policies:
        s = comparison["per_policy"][p]["summaries"]
        t = s["time_to_recall_target"]["mean"]
        print(
            f"  {p:<16} success {s['mission_success']['mean']:.2f}  "
            f"time to target (successful episodes) {'—' if t is None else f'{t:.1f}'}"
        )
    print(
        f"\ndone in {time.perf_counter() - t0:.1f}s. "
        f"Next: swarm dashboard --data-dir {out}/dashboard"
    )
    return EXIT_OK


def _delegate(main: Callable[[Sequence[str] | None], int], argv: list[str]) -> int:
    return main(argv)


def cmd_dashboard(args: argparse.Namespace) -> int:
    from swarm.dashboard.server import main

    return _delegate(main, args.rest)


def cmd_sim3d(args: argparse.Namespace) -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise CliError('the 3D analysis needs MuJoCo: pip install "swarm[sim3d]"') from exc
    from swarm.sim3d.mujoco_replay import main

    rest = list(args.rest)
    if "--config" in rest:  # accept example names here too
        k = rest.index("--config") + 1
        if k < len(rest) and rest[k] in EXAMPLES:
            target = Path(tempfile.mkdtemp(prefix="swarm-sim3d-")) / f"{rest[k]}.json"
            _write(target, resolve_config_text(rest[k]))
            rest[k] = str(target)
    return _delegate(main, rest)


def cmd_study(args: argparse.Namespace) -> int:
    from swarm.experiments.study import main

    return _delegate(main, args.rest)


def cmd_analyze(args: argparse.Namespace) -> int:
    from swarm.experiments.study_analysis import main

    return _delegate(main, args.rest)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from swarm.experiments.benchmark import main

    return _delegate(main, args.rest)


# ----------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm",
        description="SWARM: deterministic multi-robot search-and-rescue simulator "
        "(simulation only).",
        epilog="Configs: a JSON path or a packaged example (" + ", ".join(EXAMPLES) + "). "
        "Start with `swarm demo`.",
    )
    parser.add_argument("--version", action="version", version=f"swarm {swarm.__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def config_args(p: argparse.ArgumentParser, default: str | None = "basic") -> None:
        p.add_argument(
            "--config",
            default=default,
            required=default is None,
            help="config JSON path or example name (default: %(default)s)",
        )
        p.add_argument(
            "--models",
            type=Path,
            default=Path("docs/results/phase4/models"),
            help="trained Phase 4 actors, offered as learned:<name> (if present)",
        )

    p = sub.add_parser("demo", help="run the packaged examples end to end (start here)")
    p.add_argument("--out-dir", type=Path, default=Path("swarm-demo"))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--compare-seeds", type=int, default=5, help="seeds in the demo comparison")
    p.add_argument("--models", type=Path, default=Path("docs/results/phase4/models"))
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("run", help="run one episode and print a summary")
    config_args(p)
    p.add_argument("--policy", default="auction_comm")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, help="write the full result JSON here")
    p.add_argument("--events", type=Path, help="also write the event stream (JSON Lines)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="check a config and list the policies that can run it")
    config_args(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("policies", help="list policies and whether they can run a config")
    config_args(p)
    p.set_defaults(func=cmd_policies)

    p = sub.add_parser("compare", help="compare policies on the same seeds (exploratory)")
    config_args(p)
    p.add_argument("--policies", nargs="+", default=["frontier_shared", "auction_comm"])
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_compare)

    for name, func, text in (
        (
            "dashboard",
            cmd_dashboard,
            "launch the web dashboard (same options as python -m swarm.dashboard)",
        ),
        ("sim3d", cmd_sim3d, "3D kinematic co-simulation of one episode (needs the sim3d extra)"),
        ("study", cmd_study, "run a pre-registered study spec"),
        (
            "analyze",
            cmd_analyze,
            "analyze a study run with its pre-registered tests",
        ),
        ("benchmark", cmd_benchmark, "time the simulator and policies (exploratory)"),
    ):
        p = sub.add_parser(name, help=text, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=func)
        DELEGATED[name] = func
    return parser


DELEGATED: dict[str, Callable[[argparse.Namespace], int]] = {}


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if tokens and tokens[0] in DELEGATED:  # pass every remaining option through untouched
        args = argparse.Namespace(command=tokens[0], rest=tokens[1:], func=DELEGATED[tokens[0]])
    else:
        args = parser.parse_args(tokens)
    try:
        code: int = args.func(args)
        return code
    except CliError as exc:
        print(f"swarm {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # output piped into e.g. `head`
        sys.stderr.close()
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
