# Testing and Quality Strategy

## Test layers
- Unit: movement, collision, energy, detection, communication, metrics.
- Integration: complete episode, reset reproducibility, policy-environment interaction.
- Property-style: positions remain in bounds; blocked cells remain impassable; energy never becomes negative unless explicitly modelled.
- Regression: fixed-seed scenarios and previously corrected bugs.
- API: environment interface compliance where applicable.

## Quality gates
- Formatting and linting.
- Static typing for public interfaces.
- Test suite passes.
- No unexplained nondeterminism in deterministic mode.
- Documentation updated for changed contracts.

## Test design
Test edge cases: simultaneous movement conflicts, agent failure, zero survivors, all cells blocked, no communication peers, depleted battery, episode time limit, and false detections.
