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

### As implemented (Phases 1 and 2)
- `SwarmEnv(config).reset(seed, *, scenario=None)`. The optional `scenario` supplies explicit ground truth for
  tests or fixed benchmark maps.
- `SwarmEnv.step(actions) -> StepResult(observations, terminations, truncations, infos, events)`. Actions may be
  any integer-like value in `0..4`; `bool` is rejected.
- `SwarmEnv.snapshot() -> EnvSnapshot` (ground truth, for the evaluator and renderer only) and
  `SwarmEnv.action_mask(agent_id)`.
- `Policy.act(observation, rng) -> int`, plus `name`, `version`, `uses_oracle`.
  `PolicyFactory = Callable[[AgentId, AgentTypeSpec], Policy]`.
- `run_episode(config, policy_factory, seed, *, designation="evaluation") -> EpisodeResult`.
- Team policies (Phase 2, [ADR-0014](decisions/0014-team-policies-and-information-models.md)): `TeamFactory(name, build)`
  builds a `TeamPolicy` from a `MissionBrief`. `act(observations, rngs) -> {agent: action}` covers every active agent;
  `diagnostics()` and `decision_log()` are recorded. Oracle teams get `attach_oracle(OracleView)`. `run_episode` and
  `run_experiment` accept a per-agent `PolicyFactory` or a `TeamFactory`. Results record `policy.information`.
- `run_experiment(config, policy_factory, seeds, *, bootstrap=None, designation="evaluation") -> ExperimentResult`
  and `compare_experiments(baseline, candidate, *, bootstrap=None) -> dict` (Phase 1 M2).
- M2 additions: `Observation.local_hazard`, `ScenarioConfig.hazards / p_false_positive / recall_target`,
  `Scenario.hazards`, the `hazard_exposure` event, and the `mission_success` end reason. Existing fields keep their
  meaning.

## Serialization
Versioned JSON for scenario configs and summary results. JSON Lines for event streams. Never use pickle as the only
record of an experiment.

## Compatibility
The core follows the shape of the PettingZoo Parallel API without depending on it. In Phase 4 a thin adapter adds
Gymnasium spaces and rewards, and it must pass PettingZoo's parallel API test. Record the PettingZoo and Gymnasium
versions used.
