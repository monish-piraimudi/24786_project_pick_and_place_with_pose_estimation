:::: collapse How Implicit BC Works
## How Implicit BC Works

::: highlight
##### Overview

Before you continue with the lab, keep one key idea in mind. In explicit behavior cloning, you would usually learn a direct mapping from state to action.

In this lab, you should think about the policy differently. Instead of outputting one action in a single forward pass, the model assigns an energy to each candidate state-action pair $(s, a)$.

Lower energy means "this action looks more like something the expert would do in this state." When you later inspect training and rollout behavior, remember that the controller is searching for a low-energy 4D motor command rather than directly regressing one answer.

:::

### One Concrete Intuition

Suppose the TCP is already close to the cube and the gripper is still open. You should not expect there to be exactly one perfect 4-motor command in that situation. Several nearby motor-angle combinations could move the robot slightly down and inward toward a grasp.

With implicit BC, the model does not have to commit to one action immediately. Instead, you should picture it assigning low energy to a small region of expert-like motor commands, and higher energy to commands that move away from the cube, close the gripper too early, or swing the arm in an implausible direction.

That is the main difference from direct regression. A direct regressor tries to predict one action in one forward pass. Here, the policy scores many candidate actions and then selects a good one by search.

### Formalism

Later in the lab, you can map the code back to the following mathematical picture:

$$
\mathcal{D} = \{(s_i, a_i)\}_{i=1}^{N}, \qquad s_i \in \mathbb{R}^{17}, \qquad a_i \in \mathbb{R}^{4}.
$$

Here, $s_i$ is the compact state observation and $a_i$ is the expert motor-angle action.

The policy learns an energy model

$$
E_{\theta}(s, a).
$$

For each expert pair $(s_i, a_i)$, training compares the positive expert action against $K = 63$ negative actions. In the current implementation, those negatives are a 50/50 mix of:
- global samples drawn uniformly inside the bounded 4D action range
- local samples made by perturbing the expert action with Gaussian noise

All negative actions are clipped back to the action bounds before scoring.

The training logits are the negative energies:

$$
\left[
-E_{\theta}(s_i, a_i),
-E_{\theta}(s_i, a_i^{(1,\mathrm{neg})}),
\ldots,
-E_{\theta}(s_i, a_i^{(63,\mathrm{neg})})
\right].
$$

The loss is cross-entropy with the expert action in the first slot as the correct class. You should read that objective as: push expert actions toward lower energy than sampled non-expert actions in the same state.

At inference time, the controller approximately solves:

$$
\arg\min_{a} E_{\theta}(s, a).
$$

It does not enumerate every possible action. In `modules/imitation_policy.py`, the search is a bounded Cross-Entropy Method (CEM), warm-started from the previous executed action when available. The raw minimum-energy proposal is then passed through exponential moving average (EMA) smoothing before the final command is applied.

:::: quiz
**Question:**
::: question What should you conclude from this section about why the policy in this lab is called "implicit" behavior cloning?
You should conclude that the policy does not directly regress the final 4D motor command from the state alone. Instead, it learns an energy over state-action pairs and then selects an action by search for low energy.
:::
::::
