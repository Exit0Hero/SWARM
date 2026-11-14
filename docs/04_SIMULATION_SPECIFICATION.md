# 2D Simulation Specification

## World
Represent the initial world as a 2D grid. Each cell may be traversable, blocked, hazardous, or contain a point of interest. Keep the cell model extensible.

## Agents
Each agent has:
- stable ID and type;
- position and heading (if needed);
- movement capability;
- sensor radius and detection model;
- communication range;
- battery/energy capacity and current level;
- status: active, disabled, returning, or completed.

## Hidden state and observations
Survivor locations are hidden from agents until detected. A local observation is derived from sensor range, occlusion rules if implemented, and configurable noise. Do not leak hidden state into decentralized policies.

## Actions
Start with a small discrete action set: stay, move north, move east, move south, move west, scan, communicate, return-to-base. Validate actions centrally.

## Step order
1. Validate proposed actions.
2. Resolve movement and collisions consistently.
3. Apply energy costs.
4. Update hazards and environment dynamics.
5. Generate detections and communication events.
6. Update mission state and termination.
7. Emit metrics/events.

Document tie-breaking rules so seeded runs are reproducible.

## Scenario configuration
Include map dimensions, terrain/obstacles, hazard parameters, base location, survivor count/placement constraints, agent roster, time limit, sensor settings, communication settings, and seed.
