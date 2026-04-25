<table width="100%">
  <tr>
    <td align="left">
      <img width="200" src="https://github.com/user-attachments/assets/b8556bbd-96dc-4246-8870-7ade261f69c1" />
    </td>
    <td align="center">
      <img width="400" src="https://github.com/user-attachments/assets/f75c32f2-0d8b-49dc-928e-a8c0e1d9fc59" />
    </td>
   <td align="right">
      <img width="200" src="https://github.com/user-attachments/assets/678845b6-e556-4735-81bc-6a94cb78edf7" />
    </td>
  </tr>
</table>

<br>

# Advanced Optimization for Autonomous Pick-and-Place Operations using the Compliance Lab Platform

This repository contains the documentation and implementation of the course project by **Daniel Jung** and **Monish Piraimudi** for the Spring 2026 course:

**24786 Special Topics: Advanced Optimization for Engineering**  
Instructor: **Dr. Frederike Dümbgen**  
*Assistant Professor, Mechanical Engineering, Carnegie Mellon University*

---

## 📌 General Description

The goal of this project is to build an Emio-based **pick-and-place imitation-learning lab** on top of the Compliance Lab Platform.

The current system:
- runs a scripted expert pick-and-place behavior in SOFA
- collects rollout episodes from that same scene
- trains a state-only implicit behavior-cloning policy
- evaluates the learned controller in closed loop
- supports optional camera tracking and RGB logging paths for inspection and future extensions

---

## ⚠️ Problem Statement & Challenges

Key technical challenges include:

- **Consistent Task Definition**  
  Keeping one scene as the source of truth for expert behavior, dataset collection, policy rollout, and evaluation  

- **State Design**  
  Compressing the pick-and-place task into a compact geometric state that is expressive enough for learning while remaining simple enough to train robustly  

- **Implicit Policy Inference**  
  Learning an energy-based state-action model and searching for low-energy motor commands reliably at runtime  

- **Closed-Loop Robustness**  
  Handling different cube start positions, rollout drift, and phase failures during policy execution and evaluation  

---

## 🛠️ Implementation Plan

We design a multi-stage pipeline:

1. **Scene and Expert Layer**
   - Define the Emio pick-and-place task in SOFA
   - Run a scripted expert that performs approach, grasp, lift, place, and retreat phases  

2. **Dataset and Training Layer**
   - Record expert rollouts as episode files
   - Flatten state-action pairs and train an implicit behavior-cloning policy  

3. **Evaluation and Inspection Layer**
   - Roll out the learned policy in closed loop
   - Inspect behavior in SOFA and compare learned performance to the expert baseline  

**Evaluation Metrics:**
- Pick success rate  
- Place success rate  
- Total success rate  
- Final place error  
- Failure phase breakdown  

---

## 📈 Optimization Components

This project integrates several optimization-oriented components:

- **Implicit Behavioral Cloning**  
  Train an energy model over state-action pairs instead of directly regressing actions  

- **Action Search at Inference Time**  
  Use a bounded Cross-Entropy Method (CEM) search to find low-energy motor-angle commands  

- **State and Action Normalization**  
  Reuse training-time normalization statistics and action bounds to keep inference stable  

- **Rollout Smoothing and Control**  
  Apply action smoothing and phase-based rollout structure for more stable closed-loop execution  

---

## 🎯 Learning Outcomes

- **Research Impact**  
  Stronger foundation in imitation learning, energy-based policies, and robot policy evaluation  

- **Career Development**  
  Practical experience with:
  - SOFA scene construction  
  - dataset design for imitation learning  
  - policy training and checkpointing  
  - closed-loop policy evaluation on robotic manipulation tasks  

---

## 🗓️ Timeline

### Before Week 13
- Built the baseline Emio pick-and-place scene  
- Implemented the scripted expert and task evaluator  
- Established the initial collection and simulation workflow  

### Week 13
- Added episode recording, dataset utilities, and rollout manifests  
- Defined the compact state representation used for training and inference  
- Integrated optional camera and tracking hooks for future extensions  

### Week 14
- Trained the state-only implicit behavior-cloning policy  
- Added closed-loop evaluation and SOFA inspection scenes  
- Refined the lab walkthrough, exercises, and final documentation  

### After Week 14
- Finalized the Emio Labs teaching flow and bonus design exercises  
- Continued polishing scene controls, evaluation options, and student-facing documentation  

---

## 📎 Notes

# Emio.imitation_lab

This repository now contains a complete Emio imitation-learning lab for the application [Emio Labs](https://docs-support.compliance-robotics.com/docs/next/Users/EmioLabs/), rather than a blank template. The lab walks students through expert rollout inspection, dataset collection, policy training, evaluation, and interactive policy inspection in SOFA.

## Description of the files

1. `imitation_lab.md`: the main markdown file displayed in the __Emio Labs__ application. It includes the lab overview and pulls in the step-by-step section files from `sections/`.
2. `lab.json`: the json file for the application Emio Labs, with the title, description of the lab, and other info needed by the application:
    ```json
    {
        "name": "imitation lab", // the lab name used by the application
        "filename": "imitation_lab.md", // the name of the markdown file
        "title": "Imitation Lab", // the title shown in Emio Labs
        "description": "Learn the Emio pick-and-place imitation-learning pipeline from expert rollout to policy evaluation." // description shown in the main table of contents
    }
    ```
3. `imitation_lab.py`: the main Python scene entrypoint for __SOFA Robotics__ used to launch the scripted expert pick-and-place scene referenced by the lab markdown.
4. `setLabName.sh`: a legacy helper from the original lab template. It is not part of the core imitation-learning workflow, but remains in the repository.
5. `requirements.txt`: the Python dependency list for this lab, including packages needed for data collection, policy training, evaluation, and supporting utilities.

## Usage

Open the lab in Emio Labs and follow the guided workflow:
1. inspect the scripted expert in SOFA
2. collect expert demonstrations
3. inspect the saved episode format
4. train the implicit behavior-cloning policy
5. evaluate the learned controller
6. inspect expert and policy behavior interactively in SOFA

For direct script usage outside the Emio Labs UI, the main entrypoints are:
- `collect_il_dataset.py`
- `train_il_policy.py`
- `evaluate_il_policy.py`
- `policy_inspect_scene.py`

You can still refer to the Emio Labs authoring documentation [here](https://docs-support.compliance-robotics.com/docs/next/Users/EmioLabs/create-your-lab/) for markdown syntax and platform-specific lab features.
