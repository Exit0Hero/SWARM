# Agent and Action Model

## Heterogeneous agents
Model drone and ground-robot differences as configuration, not scattered conditionals. Example capabilities include movement speed, traversable terrain, sensor footprint, energy cost, communication range, and payload capacity.

## Action contract
A policy receives an observation and returns an action for a specific agent. The environment validates the action and resolves conflicts.

## Invalid actions
Define one consistent strategy: reject with a recorded invalid-action event, map to stay, or mask before policy selection. Do not silently apply inconsistent behavior.

## Resource model
Energy consumption should be parameterized by action and agent type. Return-to-base behavior must have a documented feasibility rule. An agent with insufficient energy must not continue indefinitely.
