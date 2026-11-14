"""Navigation: known-map memory, A*, frontiers, energy feasibility, prioritized planning.

Planners are pure functions of a map *as known to the planner* (ADR-0015). Only the oracle
baseline plans on ground truth, and it does so through an explicit ``OracleView`` (ADR-0014).
"""
