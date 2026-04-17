:::: collapse Summary
## Summary

In this lab you:
- inspected the pick-and-place scene in SOFA
- used that exact scene to generate imitation-learning episodes
- trained a state-only implicit behavior-cloning policy
- evaluated the learned controller in closed loop
- compared expert and learned rollout performance

The key idea is that one scene drives the whole pipeline:
- scene definition in `modules/pick_place_il.py`
- runtime collection and evaluation in `modules/pick_place_il_runtime.py`
- optional RGB logging and tracker capture in `modules/emio_camera_observation.py`
- offline training in `train_il_policy.py`

The recommended GUI workflow stays simulation-first:
- launch one SOFA scene
- inspect the learned rollout in simulation mode
- use the Emio GUI's Simulation / Robot switch only when you intentionally want hardware execution outside the lab walkthrough; the scene does not connect on launch

From here, you can experiment with:
- more demonstrations
- wider workspace bounds
- different compact state definitions
- different implicit-policy search hyperparameters
- alternative action parameterizations

::::
