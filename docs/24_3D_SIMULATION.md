# 3D simulation guide (Phase 8)

The design is in [ADR-0027](decisions/0027-3d-kinematic-co-simulation.md). Read first: this is
**kinematic co-simulation**. Episodes run in the 2D simulator with the usual policies, then they
are lifted into 3D and checked in MuJoCo. It is not a flight-dynamics model.

```bash
pip install -e ".[sim3d]"     # MuJoCo; optional
python -m swarm.sim3d.mujoco_replay --config configs/phase3/nominal.json \
    --policy auction_comm --seed 1000 --scene-out scene.json
python -m swarm.dashboard      # run an episode, then click "3D view"
```

## What you get
- **Geometry:** structures are 2 m boxes, debris 0.4 m, hazards are floor markers, and the base is
  a pad. One cell is 1 m.
- **Robots:** ground robots move at 0.25 m. Drones are on the pad at the base and fly at 3 m
  elsewhere. Every 3D point is the robot's recorded 2D cell at that step.
- **MuJoCo analysis:**
  - robot–obstacle contacts, which should be 0 for every real episode;
  - for each detection report, whether a straight 3D line of sight from the robot to the cell is
    blocked by terrain (`reports_occluded`).

  The 2D sensor ignores occlusion, so this shows how many reports a line-of-sight sensor would
  have missed. Example (development seed 1000, nominal, `auction_comm`): 6 of 62 reports
  occluded, 0 contacts. That is a single episode, not a study.
- **Dashboard 3D view:** rotate with yaw and pitch. Play, step and scrub as in 2D. It shows
  message links, trails, disabled robots and the ground-truth survivors.

## Not supported
Flight physics and controllers, 3D-aware policies or sensing, and photoreal rendering.
