# API and Data Contracts

## Initial interfaces (illustrative)
- `ScenarioConfig`: validated immutable scenario settings.
- `Environment.reset(seed)`: initialize an episode and return initial observations/info.
- `Environment.step(actions)`: apply joint actions and return observations, rewards, termination, truncation, info.
- `Policy.act(observation, agent_id)`: return a valid action.
- `ExperimentRunner.run(config, policy, seeds)`: execute and persist evaluation runs.

The exact signatures should be finalized after repository inspection and aligned with any existing code.

## Serialization
Use versioned JSON or another documented format for scenario configs and summary results. Avoid serializing arbitrary Python objects as the only record of an experiment.

## Compatibility
If adopting Gymnasium/PettingZoo conventions, document version constraints and validate with the relevant API test utilities.
