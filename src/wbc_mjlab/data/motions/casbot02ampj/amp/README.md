# CASBOT02 AMPJ motion data

Place locomotion clips in `WalkandRun/` and fall/get-up clips in `Recovery/`.
Every `.npz` must contain:

- `fps`: positive scalar
- `joint_pos`, `joint_vel`: `[T, 27]`
- `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w`: `[T, 30, 3]`
- `body_quat_w`: `[T, 30, 4]`, quaternion order `wxyz`

The 27 joint columns must use this exact order:

1. `left_leg_pelvic_pitch_joint`
2. `left_leg_pelvic_roll_joint`
3. `left_leg_pelvic_yaw_joint`
4. `left_leg_knee_pitch_joint`
5. `left_leg_ankle_pitch_joint`
6. `left_leg_ankle_roll_joint`
7. `right_leg_pelvic_pitch_joint`
8. `right_leg_pelvic_roll_joint`
9. `right_leg_pelvic_yaw_joint`
10. `right_leg_knee_pitch_joint`
11. `right_leg_ankle_pitch_joint`
12. `right_leg_ankle_roll_joint`
13. `waist_yaw_joint`
14. `left_shoulder_pitch_joint`
15. `left_shoulder_roll_joint`
16. `left_shoulder_yaw_joint`
17. `left_elbow_pitch_joint`
18. `left_wrist_yaw_joint`
19. `left_wrist_pitch_joint`
20. `left_wrist_roll_joint`
21. `right_shoulder_pitch_joint`
22. `right_shoulder_roll_joint`
23. `right_shoulder_yaw_joint`
24. `right_elbow_pitch_joint`
25. `right_wrist_yaw_joint`
26. `right_wrist_pitch_joint`
27. `right_wrist_roll_joint`

The 32 dexterous-hand bodies are removed; the two head bodies remain but their
joints are fixed. Therefore the 30 body columns use the derived model body
order (world excluded): the original source XML order with every finger body
removed. If existing clips contain all 62 source bodies, their body arrays
must be reduced and reordered to this 30-body contract before training.

The exact 30-body column order is:

1. `base_link`
2. `left_leg_pelvic_pitch_link`
3. `left_leg_pelvic_roll_link`
4. `left_leg_pelvic_yaw_link`
5. `left_leg_knee_pitch_link`
6. `left_leg_ankle_pitch_link`
7. `left_leg_ankle_roll_link`
8. `right_leg_pelvic_pitch_link`
9. `right_leg_pelvic_roll_link`
10. `right_leg_pelvic_yaw_link`
11. `right_leg_knee_pitch_link`
12. `right_leg_ankle_pitch_link`
13. `right_leg_ankle_roll_link`
14. `waist_yaw_link`
15. `head_yaw_link`
16. `head_pitch_link`
17. `left_shoulder_pitch_link`
18. `left_shoulder_roll_link`
19. `left_shoulder_yaw_link`
20. `left_elbow_pitch_link`
21. `left_wrist_yaw_link`
22. `left_wrist_pitch_link`
23. `left_wrist_roll_link`
24. `right_shoulder_pitch_link`
25. `right_shoulder_roll_link`
26. `right_shoulder_yaw_link`
27. `right_elbow_pitch_link`
28. `right_wrist_yaw_link`
29. `right_wrist_pitch_link`
30. `right_wrist_roll_link`

Convert all 1000 Hz `.data` files in `WalkandRun/` to 50 Hz `.npz` beside
their sources with:

```bash
PYTHONPATH=. uv run python scripts/casbot02ampj_data_to_npz.py
```

The converter preserves existing `.npz` files by default. To regenerate them,
pass `--overwrite True`; on a machine without CUDA, pass `--device cpu`.

Convert all 30 Hz recovery `.csv` files in `Recovery/` to 50 Hz `.npz` beside
their sources with:

```bash
PYTHONPATH=. uv run python scripts/casbot02ampj_csv_to_npz.py
```

Each source CSV must contain 36 columns: base position (3), base quaternion in
`xyzw` order (4), and 29 joint positions. The converter removes the fixed
`head_yaw_joint` and `head_pitch_joint` columns by name, reorders the remaining
joints to the 27-joint teacher contract, and computes all 30-body states with
forward kinematics. Existing `.npz` files are likewise preserved unless
`--overwrite True` is passed.

Before training, validate both motion directories with:

```bash
PYTHONPATH=. uv run python scripts/verify_casbot02ampj_motion.py
```

Then start the first 40% Recovery / 60% WalkandRun run with:

```bash
./train_casbot02ampj_teacher.sh
```
