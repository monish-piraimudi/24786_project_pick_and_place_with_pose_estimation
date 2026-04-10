:::: collapse Summary
## Summary

In this lab you:
- inspected the pick-and-place scene in SOFA
- used that exact scene to generate imitation-learning episodes
- trained a hybrid image-plus-state implicit behavior-cloning policy
- evaluated the learned controller in closed loop
- compared expert and learned rollout performance

The key idea is that one scene drives the whole pipeline:
- scene definition in `modules/pick_place_il.py`
- runtime collection and evaluation in `modules/pick_place_il_runtime.py`
- real RGB observation capture in `modules/emio_camera_observation.py`
- offline training in `train_il_policy.py`

From here, you can experiment with:
- more demonstrations
- wider workspace bounds
- richer low-dimensional state features
- different implicit-policy search hyperparameters
- stronger image encoders

::::
