# Reproducibility and Experiment Logging

Each run should record:
- run ID;
- scenario/configuration hash;
- random seed(s);
- policy name and version;
- code commit;
- Python and dependency versions;
- training/evaluation designation;
- start time and duration;
- metric values and metric schema version;
- artifact/checkpoint references.

Use independent random-number generators for scenario generation, environment stochasticity, and policy stochasticity where possible. Record seed derivation. Avoid relying on process-global random state.
