# Route Navigation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a route-keyframe visual navigator for the WZRY farm movement step.

**Architecture:** New focused `navigation/` modules parse a route YAML, localize screenshots against keyframes using ORB/correlation, and run a closed-loop joystick route. `task_engine.py` gets one new `route_navigate` action while preserving existing actions.

**Tech Stack:** Python, OpenCV, NumPy, Pydantic, PyYAML, existing ADB and task engine.

---

### Task 1: Route Config And Localizer

**Files:**
- Create: `navigation/route_config.py`
- Create: `navigation/feature_localizer.py`
- Test: `tests/test_navigation.py`

- [ ] Add route YAML models for joystick settings, template groups, keyframes, and top-level route.
- [ ] Add ORB + BFMatcher + homography localization with a correlation fallback score.
- [ ] Test that the localizer picks the matching keyframe from synthetic textured images.

### Task 2: Closed-Loop Route Navigator

**Files:**
- Create: `navigation/route_navigator.py`
- Test: `tests/test_navigation.py`

- [ ] Add interrupt handling before movement.
- [ ] Add success detection before movement.
- [ ] Add keyframe progress tracking and short joystick movement.
- [ ] Test success-first, interrupt-first, and keyframe-to-move decisions with fake ADB callbacks.

### Task 3: Task Engine Integration

**Files:**
- Modify: `task_engine.py`
- Modify: `task_preflight.py`
- Test: `tests/test_task_engine.py`, `tests/test_task_preflight.py`

- [ ] Add `route_navigate` action and `RouteNavigateSpec`.
- [ ] Execute `RouteNavigator` from task nodes.
- [ ] Include route templates and keyframe images in preflight checks.

### Task 4: WZRY Farm Route Assets And Task Switch

**Files:**
- Create: `routes/wzry_farm.yaml`
- Create: `assets/routes/wzry_farm/*.png`
- Modify: `tasks/wzry_farm.yaml`
- Modify: `README.md`

- [ ] Copy existing 1280x720 farm screenshots into route keyframes.
- [ ] Configure movement actions for spawn, approach, near-statue, interaction.
- [ ] Switch `move_to_farm_button` from `visual_approach` to `route_navigate`.
- [ ] Document how to recapture and replace route keyframes.

### Task 5: Verification And Build

**Files:**
- Build output: `release/MuMuADBVC/MuMuADBVC.exe`

- [ ] Run `python -m pytest -q`.
- [ ] Run a local dry-run route probe if the emulator is reachable.
- [ ] Rebuild with `python build_exe.py`.
- [ ] Confirm release config includes the route YAML and assets.

