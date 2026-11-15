"""Multi-agent reinforcement learning (Phase 4, ADR-0022).

NumPy-only modules (importable without the ``learning`` extra):

- :mod:`swarm.learning.sector`: the learned sector-selection agent and team, actor features,
  action masks and the NumPy actor used for inference.
- :mod:`swarm.learning.rewards`: event-based reward components.

Modules that need ``pip install -e .[learning]`` (PyTorch, PettingZoo, Gymnasium):

- :mod:`swarm.learning.pz_env`: PettingZoo ``ParallelEnv`` adapters.
- :mod:`swarm.learning.ppo` and :mod:`swarm.learning.train`: IPPO and MAPPO training.

The core simulator never imports this package.
"""
