# Agent and Action Model

## Heterogeneous agents
Drone and ground-robot differences are configuration (`AgentTypeSpec`), not scattered conditionals: layer,
traversable terrain, sensor radius, energy capacity, per-action energy cost, recharge rate, and later
communication range and payload ([ADR-0002](decisions/0002-two-layer-world-model.md)). In Phase 1 every agent
moves at most one cell per step. Speed differences are a documented limitation
([ADR-0001](decisions/0001-grid-and-discrete-time.md)).

## Action contract
A policy receives one agent's observation and its own RNG stream, and returns an integer action. The environment
validates all actions and resolves conflicts ([ADR-0010](decisions/0010-api-contract.md)).

## Invalid actions
One consistent strategy ([ADR-0004](decisions/0004-action-masks-and-invalid-actions.md)):
- the observation carries an action mask so that policies can avoid invalid actions;
- a masked action that is still submitted becomes `STAY` and is recorded as an `invalid_action` event;
- input outside the contract (unknown action integer, wrong agent IDs) raises an exception.

## Resource model
Energy is an integer, parameterized by action and agent type. Agents recharge on the base cell. An agent that reaches
zero energy away from base is disabled for the rest of the episode. Return-to-base is a policy behaviour. Its
feasibility rule (start returning when remaining energy covers the planned path cost plus a margin) is defined and
tested with the Phase 2 planner.
