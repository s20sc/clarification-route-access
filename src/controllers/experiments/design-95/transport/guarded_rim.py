"""Screen each arm segment against live scene colliders before issuing it."""
from collections import Counter
import math

import numpy as np
from scipy.spatial.transform import Rotation

import rcb_behavior as runtime

BOX = "storage_container.n.01_1"
FINGERS = {"left_gripper_finger_link1", "left_gripper_finger_link2"}
STEP = .01


def arr(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)


def under(path, root):
    return path == root or path.startswith(root+"/")


def screen(s, obj, goal, destination, orientation, allow_finger_contact, exempt_pair=None, held_cargo=False, all_controlled=False, departure_pairs=()):
    before = s.state_sha(), s.og.sim.current_time_step_index
    robot = s.robot
    k = s._kin()
    world, rotation = map(arr, robot.get_position_orientation())
    base = runtime._T_of(world,rotation)
    q_idx = robot.trunk_control_idx.tolist()+robot.arm_control_idx["left"].tolist()+robot.arm_control_idx["right"].tolist()
    parts=("trunk","arm_left","arm_right") if all_controlled else ("trunk","arm_left")
    a_idx=[i for part in parts for i in robot.controller_action_idx[part].tolist()]
    count=len(a_idx)
    measured = arr(robot.get_joint_positions())[q_idx]
    from omnigibson.controllers.controller_view import ControllerView

    idx = []
    for part in parts:
        group, _ = robot.controllers[part]
        assert not ControllerView.get_use_delta_commands(group)
        idx.extend(ControllerView.get_dof_idx(group).tolist())
    assert idx == q_idx[:count]
    hold = arr(s.hold_action())[a_idx]
    full = robot.get_joint_positions().clone()
    measured_cmd = arr(robot.q_to_action(full))[a_idx]
    assert np.allclose(measured_cmd, measured[:count], rtol=0, atol=1e-6)
    difference = hold - measured_cmd
    command = measured.astype(float)
    target = np.asarray(goal, dtype=np.float32).astype(float)
    target[count:] = measured[count:]
    full[idx] = s.th.as_tensor(target[:count], dtype=full.dtype, device=full.device)
    assert np.allclose(arr(robot.q_to_action(full))[a_idx], target[:count], rtol=0, atol=1e-6)
    names = [k.q_names[i] for i in range(count)]
    joints = [dict(name=name, joint_index=idx[i], action_index=a_idx[i], measured_rad=float(measured[i]),
                   prior_hold_rad=float(hold[i]), prior_error_rad=float(difference[i]))
              for i,name in enumerate(names)]
    initial = k.fk(measured)
    psqi = s.og.sim.psqi
    floor = tuple(o.prim_path for o in s.env.scene.objects if o.category in {"floors","floor"})
    boxes = []
    for name,link in robot.links.items():
        if not link.has_collision_meshes:
            continue
        lo,hi=map(arr,link.aabb)
        if np.prod(hi-lo) <= 1e-10:
            continue
        assert name in initial, name
        boxes.append((name,np.linalg.inv(base @ initial[name]),(lo+hi)/2,(hi-lo)/2))
    assert len(boxes)>10 and FINGERS <= {b[0] for b in boxes}
    if held_cargo:
        lo,hi=map(arr,obj.aabb)
        tip=robot.eef_link_names["left"]
        boxes.append(("held_cargo",np.linalg.inv(base @ initial[tip]),(lo+hi)/2,(hi-lo)/2))

    def query(center, half, attitude):
        found=set()

        def report(hit):
            path=str(hit.rigid_body)
            assert path.startswith("/")
            found.add(path)
            return True

        assert psqi.overlap_box(halfExtent=half.tolist(),pos=center.tolist(),rot=attitude,
                                reportFn=report,anyHit=False)>=len(found)
        return found

    point=arr(obj.aabb_center)
    pos_control=query(point,np.array([.3,.3,.15]),[0.,0.,0.,1.])
    assert any(under(path,obj.prim_path) for path in pos_control)
    assert not query(np.array([70.,70.,7.]),np.full(3,.02),[0.,0.,0.,1.])

    n=max(1,math.ceil(float(np.max(np.abs(target-command)))/STEP))
    commands=np.linspace(command,target,n+1,dtype=np.float32)
    first=None
    contacts=Counter()
    allowed=0
    departure_samples=[]
    departure_by_pair={}
    start_eef=None
    for i in range(n+1):
        fk=k.fk(commands[i])
        eef=(base @ fk[robot.eef_link_names["left"]])[:3,3]
        if start_eef is None:
            start_eef = eef.copy()
        allow = allow_finger_contact and math.dist(eef,destination)<=.07
        events=[]
        for name,inv,center,half in boxes:
            delta=base @ fk[robot.eef_link_names["left"] if name=="held_cargo" else name] @ inv
            rot=Rotation.from_matrix(delta[:3,:3])
            world_center=rot.apply(center)+delta[:3,3]
            for path in sorted(query(world_center,half,rot.as_quat().tolist())):
                if under(path,robot.prim_path):
                    continue
                if held_cargo and under(path,obj.prim_path):
                    continue
                if held_cargo and name=="held_cargo" and allow and any(under(path,f) for f in floor):
                    continue
                if any(name == link and under(path, root) for link, root in departure_pairs):
                    bottom = float(world_center[2] - np.abs(rot.as_matrix()[2]) @ half)
                    history = departure_by_pair.setdefault((name, path), [])
                    contiguous = i == (history[-1]["sample"] + 1 if history else 0)
                    rises = not history or bottom >= history[-1]["finger_bottom_m"] - .0001
                    if contiguous and rises and i < n and math.dist(eef, start_eef) <= .06:
                        sample = dict(sample=i, link=name, body=path, finger_bottom_m=bottom, eef=eef.tolist())
                        history.append(sample)
                        departure_samples.append(sample)
                        continue
                if exempt_pair and name == exempt_pair[0] and under(path,exempt_pair[1]):
                    continue
                if (name=="base_link" or name.startswith(("wheel_","steer_"))) and any(under(path,f) for f in floor):
                    continue
                if allow and name in FINGERS and under(path,obj.prim_path):
                    allowed+=1
                    continue
                events.append((name,path))
        if events and first is None:
            first=dict(sample=i,eef=eef.tolist(),hits=events)
        contacts.update(events)
    assert before==(s.state_sha(),s.og.sim.current_time_step_index)
    return dict(samples=n+1,first=first,allowed_finger_hits=allowed,departure_samples=departure_samples,link_count=len(boxes),
                prior_start_command_error_rad=float(np.max(np.abs(difference))), joints=joints,
                first_command_error_rad=float(np.max(np.abs(commands[0][:count]-measured[:count]))),
                positive_control=sorted(pos_control),
                planned_destination=list(destination), planned_orientation=list(orientation),
                blocked_pairs=[dict(link=k[0],body=k[1],count=v) for k,v in sorted(contacts.items())],
                passed=first is None), commands[:, :count], idx, a_idx


def run(s, obj, candidate, verify, dump, exempt_pair, rest_controller):
    p=s.primitives()
    robot=s.robot
    target=candidate["base_target_xy"]
    yaw=candidate["base_yaw"]
    initial_box=arr(obj.get_position_orientation()[0])
    loc,quat=map(arr,robot.get_position_orientation())
    cy=Rotation.from_quat(quat).as_euler("xyz")[2]
    yaw_error=abs((cy-yaw+math.pi)%(2*math.pi)-math.pi)
    error=math.dist(loc[:2],target)
    verify("box1_dock",dict(arrived=error<=.035,heading=yaw_error<=.05,upright=runtime.up_z(quat.tolist())>=.95,
                            box_unmoved=math.dist(initial_box,arr(obj.get_position_orientation()[0]))<=.02,
                            empty=not any(robot._ag_obj_in_hand.values())), distance_m=error,yaw_error_rad=yaw_error,
                            base_pose=[loc.tolist(),quat.tolist()])
    rest_controller(s,obj,verify,dump,exempt_pair)
    orientation=candidate["arm_target"]["orientation"]
    checks=[]
    q_idx=robot.trunk_control_idx.tolist()+robot.arm_control_idx["left"].tolist()+robot.arm_control_idx["right"].tolist()
    for i,row in enumerate(candidate["arm_target"]["ik"]):
        destination=row["xyz"]
        pos,q=map(arr,robot.get_position_orientation())
        actual=arr(robot.get_joint_positions())[q_idx]
        backup=s._curobo_ik
        try:
            s._curobo_ik=lambda *args,**kwargs: None
            sol=s.kinematics(dict(op="ik",targets={robot.eef_link_names["left"]:destination+orientation},
                                  base_pose=pos.tolist()+q.tolist(),seed=actual.tolist()))
        finally:
            s._curobo_ik=backup
        verify("box1_ik",dict(solved=bool(sol["ok"])),waypoint=i,error=sol["error"])
        observation,commands,control_idx,action_idx=screen(
            s,obj,sol["q"],destination,orientation,allow_finger_contact=i==1,exempt_pair=exempt_pair)
        checks.append(observation)
        dump("arm-screen.json",checks)
        verify("box1_arm_path",dict(clear=observation["passed"]),waypoint=i)
        trace=[]
        for step, command in enumerate(commands):
            action=s.hold_action()
            action[action_idx]=s.th.as_tensor(command,dtype=action.dtype,device=action.device)
            s.step(action)
            observed=arr(robot.get_joint_positions())[control_idx]
            tracking=float(np.max(np.abs(observed-command)))
            trace.append(dict(step=step, max_error_rad=tracking,command_rad=command.tolist(),
                              measured_rad=observed.tolist()))
            if tracking>.08:
                break
        dump(f"arm-tracking-{i}.json",trace)
        verify("box1_tracking",dict(complete=len(trace)==len(commands),
                                    bounded=all(row["max_error_rad"]<=.08 for row in trace)),
               waypoint=i, max_error_rad=max(row["max_error_rad"] for row in trace))
        p["hold"](15)
        eef,actual_orientation=p["get_eef_pose"]("left")
        pos_error=math.dist(eef,destination)
        rot_error=(Rotation.from_quat(orientation).inv()*Rotation.from_quat(actual_orientation)).magnitude()
        verify("box1_arm_arrival",dict(position=pos_error<=.025,orientation=rot_error<=.12,
                                       box_unmoved=math.dist(initial_box,arr(obj.get_position_orientation()[0]))<=.025),
               waypoint=i,position_error_m=pos_error,orientation_error_rad=float(rot_error))
    p["close_gripper"]("left",55)
    verify("box1_attachment",dict(grasped=robot._ag_obj_in_hand["left"] is obj),
           actual=p["object_in_hand"]("left"))
