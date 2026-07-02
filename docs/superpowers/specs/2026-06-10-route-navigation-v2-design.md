# Route Navigation V2 Design

## Goal

Replace the brittle single-template farm movement logic with a route-based visual navigator that can move the character from the farm spawn area toward the farming interaction area using only screenshots and standard ADB input.

## Current Failure

The old `visual_approach` logic treats the statue template center as a navigation target. This is unstable in a 3D scene: the statue changes scale, moves partly off-screen, is occluded by crops or the character, and small template crops can match unrelated local details. As a result, direction decisions oscillate or fall back to blind search.

## New Approach

Use a Visual Teach-and-Repeat route:

1. Capture a small set of full-screen route keyframes: spawn, approach, near-statue, interaction.
2. On each screenshot, close interrupt popups first.
3. Detect the success button first; if present, stop moving.
4. Localize the current screenshot against route keyframes using ORB feature matches plus downscaled scene correlation.
5. Select the keyframe's configured movement action and take one short joystick step.
6. Re-screenshot and repeat, advancing route progress when a later keyframe matches.

This does not require game memory, private APIs, packet inspection, hooks, or protocol simulation. It remains screenshot-only for state and ADB-only for input.

## Boundaries

- Keep `visual_approach` for compatibility, but the WZRY farm task will use the new `route_navigate` action.
- Python OpenCV is enough for the MVP because OpenCV's feature matching code is native under the Python API.
- C++ migration only makes sense after the route model is validated.

