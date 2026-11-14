# API and Data Contracts

The interfaces are fixed by [ADR-0010](decisions/0010-api-contract.md). Exact type names may be refined during
Phase 1, but the shape may not change without a new ADR.

## Interfaces
- `ScenarioConfig`: validated, immutable scenario settings ([ADR-0007](decisions/0007-configuration-and-serialization.md)).
- `SwarmEnv.reset(seed)`: start an episode and return per-agent observations and infos.
- `SwarmEnv.step(actions)`: apply a joint action dict keyed by agent ID and return observations, terminations,
  truncations, infos and the ordered event list. The core returns no rewards
  ([ADR-0005](decisions/0005-event-driven-simulation.md)).
- `Policy.act(observation, rng)`: return a valid action for one agent.
- `run_episode(config, policy_factory, seed)` and `run_experiment(experiment_config)`: execute and persist runs.

### As implemented in Phase 1 M1
- `SwarmEnv(config).reset(seed, *, scenario=None)`. The optional `scenario` supplies explicit ground truth for
  tests or fixed benchmark maps.
- `SwarmEnv.step(actions) -> StepResult(observations, terminations, truncations, infos, events)`. Actions may be
  any integer-like value in `0..4`; `bool` is rejected.
- `SwarmEnv.snapshot() -> EnvSnapshot` (ground truth, for the evaluator and renderer only) and
  `SwarmEnv.action_mask(agent_id)`.
- `Policy.act(observation, rng) -> int`, plus `name`, `version`, `uses_oracle`.
  `PolicyFactory = Callable[[AgentId, AgentTypeSpec], Policy]`.
- `run_episode(config, policy_factory, seed, *, designation="evaluation") -> EpisodeResult`.
  `run_experiment` (multiple seeds) is planned for M2.

## Serialization
Versioned JSON for scenario configs and summary results. JSON Lines for event streams. Never use pickle as the only
record of an experiment.

## Compatibility
The core follows the shape of the PettingZoo Parallel API without depending on it. In Phase 4 a thin adapter adds
Gymnasium spaces and rewards, and it must pass PettingZoo's parallel API test. Record the PettingZoo and Gymnasium
versions used.
