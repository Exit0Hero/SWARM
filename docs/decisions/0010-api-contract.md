# ADR-0010: API contract

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1 (core), 4 (PettingZoo adapter)

## Context
[docs/09](../09_API_AND_DATA_CONTRACTS.md) gave illustrative interfaces to finalize after repository
inspection. Phase 4 needs a PettingZoo-compatible interface, but the dependency policy keeps
PettingZoo out of the core until then.

## Decision

### Environment
The core environment follows the **shape** of the PettingZoo Parallel API without importing it:

```python
class SwarmEnv:
    possible_agents: tuple[AgentId, ...]  # fixed for the episode, sorted
    agents: tuple[AgentId, ...]  # agents still acting (not disabled)

    def reset(self, seed: int) -> tuple[dict[AgentId, Observation], dict[AgentId, AgentInfo]]: ...

    def step(self, actions: Mapping[AgentId, int]) -> StepResult: ...


@dataclass(frozen=True)
class StepResult:
    observations: dict[AgentId, Observation]
    terminations: dict[AgentId, bool]
    truncations: dict[AgentId, bool]
    infos: dict[AgentId, AgentInfo]
    events: tuple[Event, ...]
```

- `AgentId` is a string such as `"drone_0"` or `"ground_1"`, assigned in roster order.
- `actions` must contain exactly the IDs in `agents`. Missing or extra IDs raise `ValueError`
  ([ADR-0004](0004-action-masks-and-invalid-actions.md)).
- A disabled agent gets `terminations[id] = True` on the step it is disabled and leaves `agents`
  afterwards (the Parallel API convention).
- **No `rewards` in the core** ([ADR-0005](0005-event-driven-simulation.md)).
- Calling `step()` after the episode has ended raises `RuntimeError`.
- Read-only snapshots for evaluation and rendering come from a separate accessor that policies
  never receive ([ADR-0008](0008-observation-and-hidden-state-boundaries.md)).

### Policy

```python
class Policy(Protocol):
    name: str
    version: str
    uses_oracle: bool

    def act(self, observation: Observation, rng: numpy.random.Generator) -> int: ...
```

Policies keep their own per-agent memory. They are constructed per episode by a factory, so no state
leaks between episodes.

### Runner and results
`run_episode(config, policy_factory, seed) -> EpisodeResult` and
`run_experiment(experiment_config) -> ExperimentResult`. Results serialize to versioned JSON
([ADR-0007](0007-configuration-and-serialization.md)) and contain the run metadata from
[docs/11](../11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md) and the metric values with their definitions
([ADR-0009](0009-mission-objectives-and-metrics.md)).

### Phase 4 adapter
A thin `ParallelEnv` wrapper in `swarm.learning` adds Gymnasium spaces and a reward dictionary computed
by a pluggable reward function. It must pass PettingZoo's `parallel_api_test`.

## Alternatives considered
- **Import PettingZoo now:** adds a dependency that Phases 1–3 do not need, and it would tie the
  core's observation type to Gymnasium spaces.
- **Gymnasium single-agent API with a joint action:** cannot express per-agent observations or
  partial observability.
- **AEC (turn-based) API:** conflicts with simultaneous movement ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)).

## Consequences
- Exact type names may be refined during Phase 1 implementation. The *shape* (dicts keyed by agent ID,
  separate termination and truncation, events, no core reward) is fixed.

## Amendment (Phase 1 M2)
`run_experiment(config, policy_factory, seeds, *, bootstrap=None, designation="evaluation")` and
`compare_experiments(baseline, candidate, *, bootstrap=None)` are implemented in the experiments package.
`episode_end` may report `mission_success` ([ADR-0009](0009-mission-objectives-and-metrics.md)), which sets
`terminations` for every acting agent. The reset and step signatures are unchanged.
