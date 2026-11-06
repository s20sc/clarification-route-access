"""A fixed-budget direct retrieval policy and separately scored access outcome."""
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from guarded_rim import FINGERS, arr, screen
from rcb_behavior import up_z


class AccessStop(Exception):
    pass


def perform(s, target, other, released_upper, config, out, dump, clock):
    from bddl import predicates

    robot = s.robot
    p = s.primitives()
    base0, _ = map(arr, robot.get_position_orientation())
    target0, target_quat = map(arr, target.get_position_orientation())
    target_rot = Rotation.from_quat(target_quat)
    other0, other_quat = map(arr, other.get_position_orientation())
    other_rot = Rotation.from_quat(other_quat)
    qi = robot.trunk_control_idx.tolist() + robot.arm_control_idx["left"].tolist() + robot.arm_control_idx["right"].tolist()
    current = lambda: arr(robot.get_joint_positions())[qi]
    tip = robot.eef_link_names["left"]
    base_pose = [v for x in robot.get_position_orientation() for v in x.tolist()]
    stable = 0
    phase = "screen"
    count = 0
    peak_other = peak_turn = 0.
    candidates = []

    def ik(xyz, orn, seed):
        prev = s._curobo_ik
        try:
            s._curobo_ik = lambda *args, **kwargs: None
            return s.kinematics(dict(op="ik", targets={tip: xyz.tolist() + orn.tolist()},
                                     base_pose=base_pose, seed=seed.tolist()))
        finally:
            s._curobo_ik = prev

    with (out / "follow-up-actions.jsonl").open("x") as actions:
        def step(action):
            nonlocal count, peak_other, peak_turn, stable
            if count >= config["max_actions"]:
                raise AccessStop("access action budget")
            a = np.asarray(action, dtype=np.float32)
            assert a.shape == (25,) and np.isfinite(a).all()
            before = clock()
            s._last_action = s.th.as_tensor(a).clone()
            obs, _, terminated, truncated, _ = s.env.step(s._last_action)
            s.ev.obs = s.ev._preprocess_obs(obs)
            s.ev._write_video()
            after = clock()
            assert after[1] - before[1] == 4
            count += 1
            base, base_quat = map(arr, robot.get_position_orientation())
            actual, target_now_quat = map(arr, target.get_position_orientation())
            rest, rest_quat = map(arr, other.get_position_orientation())
            shift = math.dist(rest, other0)
            turn = float((other_rot.inv() * Rotation.from_quat(rest_quat)).magnitude())
            peak_other, peak_turn = max(peak_other, shift), max(peak_turn, turn)
            held = robot._ag_obj_in_hand["left"]
            bounded = (not truncated and up_z(base_quat) >= .95 and math.dist(base, base0) <= .015
                       and shift <= config["other_translation_m"] and turn <= config["other_rotation_rad"]
                       and not any(obj is not None and obj is not target for obj in robot._ag_obj_in_hand.values()))
            target_shift = math.dist(actual, target0)
            target_turn = float((target_rot.inv() * Rotation.from_quat(target_now_quat)).magnitude())
            if phase in ("departure", "hover", "approach"):
                bounded = (bounded and target_shift <= config["other_translation_m"]
                           and target_turn <= config["other_rotation_rad"]
                           and not any(robot._ag_obj_in_hand.values()))
            if phase == "lift":
                bounded = bounded and held is target
            rise = float(actual[2] - target0[2])
            stable = stable + 1 if bounded and held is target and rise >= config["rise_m"] else 0
            actions.write(json.dumps(dict(index=count, phase=phase, action=a.tolist(), clock=after,
                          target_position=actual.tolist(), target_rotation_rad=target_turn,
                          other_position=rest.tolist(), other_rotation_rad=turn,
                          held_target=held is target, rise_m=rise, bounded=bounded, stable=stable,
                          stack_task_terminated=bool(terminated))) + "\n")
            actions.flush()
            if not bounded:
                raise AccessStop("follow-up movement guard")
            return False, False

        def move(commands, indices, action_indices):
            for cmd in commands:
                action = s.hold_action()
                action[action_indices] = s.th.as_tensor(cmd, dtype=action.dtype, device=action.device)
                step(action)
                if float(np.max(np.abs(arr(robot.get_joint_positions())[indices] - cmd))) > .08:
                    raise AccessStop("arm tracking guard")

        s.step = step
        s.episode_done = False  # Access is separately scored after the captured stacking terminal.
        try:
            phase = "departure"
            grip = robot.controller_action_idx["gripper_left"]
            assert np.allclose(arr(s.hold_action())[grip.tolist()], arr(s._gripper_cmd("left", True)), rtol=0, atol=1e-6)
            assert not any(robot._ag_obj_in_hand.values())
            origin, orientation = map(arr, p["get_eef_pose"]("left"))
            center = arr(released_upper.get_position_orientation()[0])
            away = origin[:2] - center[:2]
            away = away / max(np.linalg.norm(away), 1e-9)
            departure = []
            selected = None
            for upward, lateral in config["departure_offsets_m"]:
                endpoint = origin + np.r_[away * lateral, upward]
                solved = ik(endpoint, orientation, current())
                attempt = dict(endpoint_m=endpoint.tolist(), ik=solved)
                departure.append(attempt)
                if solved["ok"]:
                    candidate = screen(s, released_upper, solved["q"], endpoint, orientation, False,
                                       departure_pairs=tuple((finger, released_upper.prim_path) for finger in sorted(FINGERS)))
                    attempt["screen"] = candidate[0]
                    if candidate[0]["passed"]:
                        selected = candidate
                        break
            dump("departure-screen.json", departure)
            if selected is None:
                raise AccessStop("no screened open-hand departure")
            move(*selected[1:])
            p["hold"](10)
            clear_pose, clear_orientation = map(arr, p["get_eef_pose"]("left"))
            clear = screen(s, released_upper, current(), clear_pose, clear_orientation, False)[0]
            dump("departure-end-screen.json", clear)
            if not clear["passed"]:
                raise AccessStop("departure end overlap")
            phase = "screen"
            # Use the same four box-relative rim candidates for either placement.
            center, box_quat = map(arr, target.get_position_orientation())
            yaw = Rotation.from_quat(box_quat).as_euler("xyz")[2]
            axes = Rotation.from_euler("z", yaw)
            top = float(target.aabb[1][2])
            rim = config["rim_radius_m"]
            xy_candidates = [(rim, 0.), (-rim, 0.), (0., rim), (0., -rim)]
            xy_candidates.sort(key=lambda xy: math.dist((center + axes.apply([*xy, 0.]))[:2], base0[:2]))
            for x, y in xy_candidates:
                grasp = center + axes.apply([x, y, 0.])
                grasp[2] = top - .035
                hover = grasp + [0., 0., .14]
                orn = arr(p["top_down_quat"](yaw + math.atan2(y, x) + math.pi / 2))
                above = ik(hover, orn, current())
                at_rim = ik(grasp, orn, np.asarray(above["q"])) if above["ok"] else None
                row = dict(offset_m=[x, y], hover=above, grasp=at_rim)
                candidates.append(row)
                dump("access-candidates.json", candidates)
                if at_rim is None or not at_rim["ok"]:
                    continue
                high = screen(s, target, above["q"], hover, orn, False)
                row["hover_screen"] = high[0]
                dump("access-candidates.json", candidates)
                if not high[0]["passed"]:
                    continue
                phase = "hover"
                move(*high[1:])
                p["hold"](10)
                achieved, achieved_orientation = map(arr, p["get_eef_pose"]("left"))
                row["hover_arrival"] = dict(
                    position_error_m=math.dist(achieved, hover),
                    rotation_error_rad=float((Rotation.from_quat(orn).inv() * Rotation.from_quat(achieved_orientation)).magnitude()),
                )
                dump("access-candidates.json", candidates)
                if row["hover_arrival"]["position_error_m"] > .025 or row["hover_arrival"]["rotation_error_rad"] > .12:
                    continue
                solved = ik(grasp, orn, current())
                row["live_grasp_ik"] = solved
                dump("access-candidates.json", candidates)
                if not solved["ok"]:
                    continue
                low = screen(s, target, solved["q"], grasp, orn, True)
                row["live_grasp_screen"] = low[0]
                dump("access-candidates.json", candidates)
                if not low[0]["passed"]:
                    continue
                phase = "approach"
                move(*low[1:])
                p["hold"](10)
                phase = "grasp"
                p["close_gripper"]("left", 55)
                if robot._ag_obj_in_hand["left"] is not target:
                    raise AccessStop("no attachment to requested box")
                phase = "lift"
                eef, quat = map(arr, p["get_eef_pose"]("left"))
                raised = eef + [0., 0., .14]
                solved = ik(raised, quat, current())
                if not solved["ok"]:
                    raise AccessStop("lift IK")
                on_other = bool(s.env.task._evaluate_predicate(predicates.OnTop, config["requested"],
                                                               config["nonrequested"]))
                support_pair = ("held_cargo", other.prim_path) if on_other else None
                lift = screen(s, target, solved["q"], eef, quat, True, held_cargo=True,
                              departure_pairs=(support_pair,) if support_pair else ())
                dump("access-lift-screen.json", dict(lift[0], initial_requested_on_other=on_other,
                                                     support_pair=support_pair))
                if not lift[0]["passed"]:
                    raise AccessStop("lift screen")
                move(*lift[1:])
                p["hold"](config["stable_actions"])
                if stable < config["stable_actions"]:
                    raise AccessStop("lift not retained")
                ending = "direct access passed"
                break
            else:
                ending = "no admitted direct access candidate"
        except AccessStop as exc:
            ending = str(exc)
    return dict(passed=stable >= config["stable_actions"] and ending == "direct access passed",
                ended=ending, phase=phase, actions=count, stable_actions=stable,
                candidate_count=len(candidates), peak_other_shift_m=peak_other, peak_other_rotation_rad=peak_turn)
