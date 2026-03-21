<table width="100%">
  <tr>
    <td align="left">
      <img src="https://github.com/user-attachments/assets/55238854-d288-49bb-bbbe-bce856a9a0c8" width="200"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f9af8b64-9151-42c2-81b9-533b7da7ddef" width="400"/>
    </td>
   <td align="right">
      <img src="https://github.com/user-attachments/assets/2a2cf8f1-8af6-4bd8-bb85-231e75b18415" width="200"/>
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

The goal of this project is to implement an autonomous **pick-and-place system** that utilizes onboard cameras and robotic manipulators from the compliance lab platform.

The system will:
- Detect stack and target locations  
- Estimate orientation of objects  
- Count available parts  
- Execute precise placement into recessed targets  

---

## ⚠️ Problem Statement & Challenges

Key technical challenges include:

- **Part Orientation**  
  Robust detection of stack, target, and part orientations for reliable placement  

- **Image Enhancement**  
  Reducing camera noise and improving detection via calibration techniques  

- **Inverse Kinematics**  
  Accurately modeling manipulator motion for precise execution  

---

## 🛠️ Implementation Plan

We design a multi-stage pipeline:

1. **Perception Layer**
   - Detect stack and target
   - Estimate pose and orientation  

2. **Planning Layer**
   - Select objects
   - Compute optimal manipulator sequence  

3. **Execution Layer**
   - Perform autonomous pick-and-place actions  

**Evaluation Metrics:**
- Target population success rate  
- Path efficiency and execution time  

---

## 📈 Optimization Components

This project integrates three major optimization domains:

- **Image Enhancement**  
  Total Variation Denoising via ADMM  

- **Inverse Kinematics Optimization**  
  Improved joint configuration for precise pick and place operations  

- **Pose Estimation & Control**  
  Real-time close loop control with optimization  

---

## 🎯 Learning Outcomes

- **Research Impact**  
  Stronger foundation in optimization for robotics and computer vision  

- **Career Development**  
  Practical experience with:
  - Camera-robot calibration  
  - Autonomous manipulation  
  - Pose estimation systems  

---

## 🗓️ Timeline

### Before Week 13
- Initial stack & target detection  
- Basic pick-and-place pipeline  

### Week 13
- Implement Total Variation Denoising  
- Improve detection and counting  

### Week 14
- Integrate IK and pose optimization  
- Full system testing
- Final report and demonstration  

### After Week 14
- Finalizing implementation for emio-labs software
- Submitted for Emio – Lab Creation Contest 2026

---

## 📎 Notes

# Emio.lab_empty

Build your own lab for the application [Emio Labs](https://docs-support.compliance-robotics.com/docs/next/Users/EmioLabs/). In this repository you will find a template and a good starting point to create your own lab from scratch.

## Description of the files

1. `lab_empty.md`: the markdown file to customize, and which will be displayed in the __Emio Labs__ application. 
2. `lab.json`: the json file for the application Emio Labs, with the title, description of the lab, and other info needed by the application:
    ```json
    {
        "name": "lab_empty", // the name of the lab folder
        "filename": "lab_empty.md", // the name of the markdown file
        "title": "Lab Empty", // the title,... 
        "description": "discover...", //... and description of the lab which will appear in the main table of contents of the Emio Labs application
    }
    ```
3. `lab_empty.py`: the python scene for __SOFA Robotics__, tipically a simulation of the robot Emio that you can launch from the Emio Labs application in exercises sections using buttons.  
4. `setLabName.sh`: the script to set the name of your lab. This will replace all occurences of `"empty"` with your chosen name. Usage is `setLabName.sh myName`. 
5. `requirements.txt`: the file to list the python packages required for your lab. See the README of the `modules/site-packages` directory. Or load the lab_empty in the Emio Labs application and read the "Install Additional Python Packages" section.

## Usage

Download this repository, use the script `setLabName.sh` to update all the files with the name of your lab. Add this lab to the application and you're good to go. 
You can follow [this documentation](https://docs-support.compliance-robotics.com/docs/next/Users/EmioLabs/create-your-lab/) to help you write the markdown file of your lab.
