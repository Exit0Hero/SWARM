# Visualization and Dashboard Specification

## MVP views
- 2D map with terrain, obstacles, hazards, base, and survivor markers (ground truth optionally hidden).
- Agent positions, types, status, battery, and sensor radius.
- Search coverage and detected-survivor overlays.
- Communication links and recent messages.
- Mission timeline and key events.
- Run summary and policy comparison.

## Design constraints
Visualization reads simulation snapshots/events and must not mutate environment state. Provide a headless mode for automated experiments. Keep rendering optional so tests and training do not require a display.
