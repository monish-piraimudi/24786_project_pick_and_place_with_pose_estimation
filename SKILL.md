Use this `SKILL.md` for Codex. It is based on the Python Emio API docs for v25.12: install via `pip install emioapi`, main entrypoint `EmioAPI`, camera classes `EmioCamera` and `MultiprocessEmioCamera`, and motors class `EmioMotors`. The docs also note that motors are position-controlled, use lists of 4 values, are clamped to `[0, π]`, and that the multiprocess camera should not be used inside SOFA scenes. ([docs-support.compliance-robotics.com][1])

````md
---
name: emio-python
description: Use this skill when working with the Python emioapi package for Emio motors, camera access, calibration, and device connection logic.
---

# Emio Python Skill

## When to use this skill
Use this skill for tasks involving the Python `emioapi` package, especially when code needs to:
- connect to an Emio device
- control Emio motors
- read or configure the Emio camera
- calibrate the camera
- choose between `EmioCamera` and `MultiprocessEmioCamera`
- work with tracker positions, point clouds, or image frames
- write safe scripts for SOFA + Emio usage

Install commands documented by the API:
- `pip install emioapi`
- `pip install git+https://github.com/SofaComplianceRobotics/Emio.API.git@main`

Camera calibration tool:
- `python -m emioapi calibrate`

## Core mental model
The package centers around `EmioAPI`, which combines:
- `emio.motors` → `EmioMotors`
- `emio.camera` → `EmioCamera` by default, or `MultiprocessEmioCamera` if `multiprocess_camera=True`

`EmioAPI` is the simplest high-level interface when both motors and camera are needed.

Use the lower-level classes directly when only one subsystem is needed:
- `EmioMotors` for motor-only scripts
- `EmioCamera` for camera work in the same process
- `MultiprocessEmioCamera` for camera work outside SOFA when process isolation is useful

## Hard rules Codex should follow

### 1) Prefer `EmioAPI` for integrated robot control
When a task needs both motors and camera, default to:
```python
from emioapi import EmioAPI

emio = EmioAPI(multiprocess_camera=False)
if emio.connectToEmioDevice():
    ...
    emio.disconnect()
````

Use `EmioMotors` or camera classes directly only when the task is clearly subsystem-specific.

### 2) Motors use lists of 4 values

All motor commands and motor state are lists of length 4, in motor ID order:

* motor 0 = first element
* motor 1 = second
* motor 2 = third
* motor 3 = fourth

Do not generate scalar motor commands when the API expects all 4 motors.

### 3) Never command angles outside `[0, pi]`

The docs warn that Emio motors are clamped between `0` and `PI` radians. If values outside this range are sent, the motor will not move.

Always clamp or validate commanded angles before assigning:

```python
import math

def clamp_angle(x: float) -> float:
    return max(0.0, min(math.pi, x))
```

### 4) Prefer radians, not degrees

Motor angles are documented in radians.
When examples or user intent are in degrees, convert explicitly:

```python
import math
rad = deg * math.pi / 180.0
```

Do not silently mix degrees and radians.

### 5) Use `EmioCamera` inside SOFA scenes

The docs explicitly recommend `EmioCamera` for SOFA scenes.
Do not use `MultiprocessEmioCamera` in SOFA, because multiprocessing clashes with SOFA.

Rule:

* SOFA scene → `EmioCamera`
* non-SOFA standalone camera app → `EmioCamera` or `MultiprocessEmioCamera`
* if unsure and SOFA is mentioned → use `EmioCamera`

### 6) Only use `MultiprocessEmioCamera` when process isolation is desired

`MultiprocessEmioCamera` starts the camera in another process.
That is useful for standalone applications, but Codex should not default to it unless:

* the user explicitly wants multiprocessing
* camera isolation is needed
* the task is not in SOFA

### 7) Open/close resources explicitly

Always structure generated code with explicit open/close behavior.

For camera:

```python
camera = EmioCamera(...)
if camera.open():
    try:
        ...
    finally:
        camera.close()
```

For motors:

```python
motors = EmioMotors()
if motors.open():
    try:
        ...
    finally:
        motors.close()
```

For `EmioAPI`:

```python
emio = EmioAPI(...)
if emio.connectToEmioDevice():
    try:
        ...
    finally:
        emio.disconnect()
```

### 8) Call `update()` before consuming fresh camera outputs

For `EmioCamera`, frame/tracker/point-cloud updates are driven through `camera.update()`.
When generating camera loops, call `update()` before reading:

* `trackers_pos`
* `point_cloud`
* `frame`
* `depth_frame`
* `hsv_frame`
* `mask_frame`

### 9) Do not assume camera parameters hot-reload

The docs state camera parameters are applied when opening the camera, are not saved automatically, and to change them programmatically you need to close and reopen the camera.

Therefore:

* do not generate code that assumes `camera.parameters = ...` immediately reconfigures a running camera
* if parameters change, close then reopen
* if persistence is needed, save parameters manually in user code

### 10) Prefer defensive checks around optional camera outputs

Some camera-derived values may be unavailable depending on settings or startup state.
Use checks before consuming:

* `camera.point_cloud`
* `camera.trackers_pos`
* frames
* calibration status / running status

Avoid assuming point clouds exist unless `compute_point_cloud=True`.

## Key APIs Codex should know

### `EmioAPI`

Use for combined access to motors and camera.

Important members:

* `emio.motors`
* `emio.camera`
* `emio.device_name`
* `emio.camera_serial`

Important methods:

* `EmioAPI.listEmioDevices()`
* `EmioAPI.listUnusedEmioDevices()`
* `EmioAPI.listUsedEmioDevices()`
* `emio.connectToEmioDevice(device_name=None) -> bool`
* `emio.disconnect()`
* `emio.printStatus()`

Preferred pattern:

```python
from emioapi import EmioAPI
import math

emio = EmioAPI(multiprocess_camera=False)

if emio.connectToEmioDevice():
    try:
        emio.printStatus()
        emio.motors.angles = [math.pi / 2] * 4
    finally:
        emio.disconnect()
else:
    print("Failed to connect to Emio device.")
```

### `EmioMotors`

Use for direct motor control.

Important methods/properties:

* `open(device_name=None) -> bool`
* `findAndOpen(device_name=None) -> int`
* `close()`
* `printStatus()`
* `angles`
* `goal_velocity`
* `max_velocity`
* `position_p_gain`
* `position_i_gain`
* `position_d_gain`
* `is_connected`
* `device_name`
* `device_index`
* `moving`
* `moving_status`
* `velocity`
* `velocity_trajectory`
* `position_trajectory`

Conversion helpers documented:

* `lengthToPulse(...)`
* `pulseToLength(...)`
* `pulseToRad(...)`
* `pulseToDeg(...)`
* `relativePos(...)`

Motor-safe example:

```python
from emioapi import EmioMotors
import math

motors = EmioMotors()

if motors.open():
    try:
        target = [math.pi / 3, math.pi / 3, math.pi / 3, math.pi / 3]
        motors.angles = [max(0.0, min(math.pi, a)) for a in target]
        print("Angles:", motors.angles)
    finally:
        motors.close()
```

### `EmioCamera`

Use in-process camera access, especially in SOFA scenes.

Important properties:

* `depth_frame`
* `frame`
* `is_running`
* `track_markers`
* `compute_point_cloud`
* `show_frames`
* `parameters`
* `trackers_pos`
* `point_cloud`
* `hsv_frame`
* `mask_frame`
* `calibration_status`
* `fps`
* `depth_max`
* `depth_min`

Important methods:

* `listCameras()`
* `open(camera_serial=None) -> bool`
* `calibrate()`
* `image_to_simulation(x, y, depth=None)`
* `update()`
* `close()`

Camera loop pattern:

```python
from emioapi import EmioCamera

camera = EmioCamera(
    show=False,
    track_markers=True,
    compute_point_cloud=True,
    configuration="extended",
)

if camera.open():
    try:
        while camera.is_running:
            camera.update()

            positions = camera.trackers_pos
            pc = camera.point_cloud
            frame = camera.frame

            # user logic here
            break
    finally:
        camera.close()
```

### `MultiprocessEmioCamera`

Use only for standalone, non-SOFA camera work when another process is desired.

Important members largely mirror camera access:

* `camera_serial`
* `is_running`
* `track_markers`
* `compute_point_cloud`
* `show_frames`
* `parameters`
* `trackers_pos`
* `point_cloud`
* `hsv_frame`
* `mask_frame`

Important methods:

* `listCameras()`
* `open(camera_serial=None) -> bool`
* `close()`

Do not generate this class in SOFA code.

## Decision guide for Codex

### If the user wants to move the robot

Use `EmioAPI` or `EmioMotors`.
Default to `EmioAPI` unless the camera is irrelevant.

### If the user wants camera tracking in SOFA

Use `EmioCamera`.

### If the user wants a standalone camera process

Use `MultiprocessEmioCamera`.

### If the user wants camera calibration

Mention or use:

```bash
python -m emioapi calibrate
```

Or generate code around `camera.calibrate()` only if the task clearly calls for programmatic calibration flow.

### If the user wants device discovery

Use:

```python
EmioAPI.listEmioDevices()
EmioAPI.listUnusedEmioDevices()
EmioAPI.listUsedEmioDevices()
```

## Code generation preferences

### Prefer small, explicit scripts

Generated scripts should:

* import only needed classes
* open resources explicitly
* validate connection success
* close/disconnect in `finally`
* use radians for motor angles
* keep camera configuration explicit

### Prefer safe defaults

Default assumptions:

* `multiprocess_camera=False`
* `show=False` unless visual debugging is explicitly needed
* `track_markers=False` unless tracking is needed
* `compute_point_cloud=False` unless point cloud is needed

Turn features on only when necessary.

### Avoid these mistakes

Do not:

* send motor angles outside `[0, math.pi]`
* use `MultiprocessEmioCamera` in SOFA
* assume `camera.parameters` persist automatically
* assume `camera.parameters` reconfigure a running camera without reopen
* read stale camera outputs in loops without `update()`
* assume a device or camera exists without checking open/connect return values

## Common patterns Codex can reuse

### Pattern: connect to first available Emio

```python
from emioapi import EmioAPI

emio = EmioAPI(multiprocess_camera=False)
if emio.connectToEmioDevice():
    try:
        print("Connected:", emio.device_name)
    finally:
        emio.disconnect()
```

### Pattern: set all four motors

```python
import math
from emioapi import EmioMotors

motors = EmioMotors()
if motors.open():
    try:
        motors.angles = [math.pi / 2] * 4
    finally:
        motors.close()
```

### Pattern: enable tracking and point cloud

```python
from emioapi import EmioCamera

camera = EmioCamera(show=False, track_markers=True, compute_point_cloud=True)
if camera.open():
    try:
        camera.update()
        print(camera.trackers_pos)
        print(None if camera.point_cloud is None else camera.point_cloud.shape)
    finally:
        camera.close()
```

### Pattern: reopen camera after parameter change

```python
from emioapi import EmioCamera

camera = EmioCamera(track_markers=True)

params = {
    "hue_h": 180,
    "hue_l": 0,
    "sat_h": 255,
    "sat_l": 0,
    "value_h": 255,
    "value_l": 0,
    "erosion_size": 1,
    "area": 50,
}

camera.parameters = params

if camera.open():
    try:
        camera.update()
    finally:
        camera.close()
```

## Output style for Codex

When explaining Emio code:

* be concrete
* mention whether code is for motors, camera, or both
* state whether the code is SOFA-safe
* explicitly call out radians vs degrees
* explicitly call out whether point cloud and marker tracking are enabled

When editing code:

* preserve existing class choice unless it is clearly wrong
* preserve SOFA compatibility if present
* avoid adding multiprocessing unless requested

```

This version is aligned to the Compliance Robotics Python docs, including installation, calibration, the `EmioAPI` composition model, camera/motor APIs, the 4-motor list convention, the `[0, π]` motor constraint, and the SOFA warning for the multiprocess camera. :contentReference[oaicite:1]{index=1}
```

[1]: https://docs-support.compliance-robotics.com/docs/v25.12/Developers/emio-api/ "Emio API | Compliance Robotics | Docs & Support"
