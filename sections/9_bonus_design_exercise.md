:::: collapse Bonus Exercise: Designing For Different Objects
## Bonus Exercise: Designing For Different Objects

The core lab assumes one cube, one pickup style, and one fixed placement setup. As a final extension, think like a system designer: how would you adapt the same Emio pipeline if the robot had to pick up different kinds of objects instead of just this one cube?

:::: exercise
**Bonus Exercise:**

Imagine that Emio now needs to handle several new object types, such as:
- a taller block
- a fragile object
- a wider or flatter object

For each case, reason through the full system design rather than only the policy.

In your answer, discuss:
1. **Object geometry and contact:** How would the object's size, shape, or contact behavior change the grasping problem?
2. **Perception and state:** What additional information would the system need beyond the current compact state? Would you need object size, orientation, material cues, or something else?
3. **Grasp strategy:** What should change in the approach, hover height, gripper opening, or attach/release logic?
4. **Policy and data:** Would the current demonstrations and dataset still be enough, or would you need new data, new action parameterizations, or different training conditions?
5. **Evaluation:** Which current success metrics would still matter, and what new evaluation criteria would you add for these objects?

Then propose:
- one hardware or end-effector change
- one observation or state change
- one data-collection or training change
- one new evaluation criterion

The goal is not to produce one perfect answer. The goal is to show that you can connect object properties to the design of the scene, the state, the demonstrations, the policy, and the evaluation pipeline.

:::

::::
