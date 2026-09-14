# A Dynamic Obstacle Avoidance Method based on Monocular Camera for UAVs via Knowledge Distillation

Core implementation of the monocular dynamic obstacle avoidance method presented in the paper
*A Dynamic Obstacle Avoidance Method based on Monocular Camera for UAVs via Knowledge Distillation*.

## Modules

| Module | Description |
| --- | --- |
| `obs.py` | obstacle state estimation, tracking and association |
| `ttc_predictor.py` | three-dimensional target tracking and time-to-collision prediction |
| `adptive_area.py` | obstacle area growth estimation |
| `pos_predictor.py` | obstacle position prediction |
| `path_plan.py` | B-spline trajectory optimisation, RRT and repulsion-based local avoidance |
| `thetas_rec.py` | Theta\* global path planning |

## Requirements

```bash
pip install -r requirements.txt
```

## Note

The core code is open-sourced here. The main code of the paper is currently being
compiled in its stable version, and the remaining details will be published in this
repository.
