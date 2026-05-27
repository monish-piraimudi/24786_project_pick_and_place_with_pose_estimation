:::: collapse Summary
## Summary

In this lab you:
- inspected the pick-and-place scene in SOFA
- used that scene to generate expert demonstrations
- trained a state-based implicit behavior-cloning policy
- evaluated the learned policy in closed loop
- compared expert and learned rollout performance

One of the key limitations of the current approach is that the policy is closely tied to a specific state representation and a fixed workspace configuration. It relies on a hand-crafted 17D geometric state vector and a pre-defined phase structure from the expert controller, rather than learning a fully end-to-end neural policy. This makes the system easier to train and debug, but also means that any change to the task — a new placement target, a different object, or a wider workspace — requires new demonstrations and potentially a new state definition.

From here, there are several directions worth exploring:
- more demonstrations and wider workspace bounds to improve generalisation
- different compact state definitions that capture new task-relevant geometry
- alternative action parameterizations or different implicit-policy search hyperparameters
- adding RGB image observations to the state, enabling the policy to handle tasks where geometry alone is not sufficient — for example, distinguishing objects by appearance or handling partial occlusion

::::
