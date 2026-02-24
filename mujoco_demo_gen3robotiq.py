import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from helper_functions import (
    _get_joint_ids_by_name, find_ee_target, build_gripper_controls, quat_from_mat3,
    set_white_environment_visuals, apply_gripper, show_wrist_window,
    make_R_with_axis_k_down, depth_to_vis
)

try:
    import cv2
except Exception:
    cv2 = None

URDF_PATH = r"gen3_modified.urdf"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_to_rotvec(q_err: np.ndarray) -> np.ndarray:
    v = np.zeros(3, dtype=float)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


def dls_ik_compute_qtarget(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_body_id: int,
    arm_joint_ids: list[int],
    qpos_adrs: list[int],
    dof_adrs: list[int],
    p_des: np.ndarray,
    q_des: np.ndarray,
    rot_w: float,
    damping: float,
    dt_step: float,
    qd_max: np.ndarray,
    iters: int,
) -> np.ndarray:
    q_arm0 = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    for _ in range(iters):
        mujoco.mj_forward(model, data)

        p_cur = data.xpos[ee_body_id].copy()
        R_cur = data.xmat[ee_body_id].reshape(3, 3).copy()
        q_cur = mat3_to_quat(R_cur)

        ep = (p_des - p_cur)
        q_err = quat_mul(q_des, quat_conj(q_cur))
        er = quat_to_rotvec(q_err)
        e = np.concatenate([ep, rot_w * er], axis=0)

        Jp = np.zeros((3, model.nv), dtype=float)
        Jr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, Jp, Jr, ee_body_id)

        Jp7 = Jp[:, dof_adrs]
        Jr7 = Jr[:, dof_adrs]
        J6 = np.vstack([Jp7, rot_w * Jr7])

        JJt = J6 @ J6.T + (damping ** 2) * np.eye(6)
        try:
            x = np.linalg.solve(JJt, e)
        except np.linalg.LinAlgError:
            break
        dq = J6.T @ x

        dq_lim = qd_max * dt_step
        dq = np.clip(dq, -dq_lim, dq_lim)

        for i, adr in enumerate(qpos_adrs):
            data.qpos[adr] += dq[i]

        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 2e-4:
            break

    q_target = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    for i, adr in enumerate(qpos_adrs):
        data.qpos[adr] = q_arm0[i]
    mujoco.mj_forward(model, data)

    return q_target


def main():
    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)
    os.chdir(urdf_dir)

    scene_path = os.path.join(urdf_dir, "robotsuit_cubes.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    wrist_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb"))
    fixed_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_down"))
    if wrist_cam_id < 0 or fixed_cam_id < 0:
        raise RuntimeError("Missing wrist_rgb or fixed_down camera in XML.")

    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    # hold immediately
    data.ctrl[:7] = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    grip_controls, lo_m, hi_m = build_gripper_controls(model)

    bracelet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_bracelet_link")
    if bracelet_body_id < 0:
        raise RuntimeError("Could not find body 'gen3_bracelet_link'.")
    bracelet_body_id = int(bracelet_body_id)

    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)

    renderer = None
    if cv2 is not None:
        # smaller = less chance of falling behind
        renderer = mujoco.Renderer(model, width=960, height=540)

    renderer_depth = None
    if cv2 is not None:
        renderer_depth = mujoco.Renderer(model, width=960, height=540)
        renderer_depth.enable_depth_rendering()

    dt_view = 1.0 / 60.0

    IK_ITERS_NEAR = 12
    IK_ITERS_TRANSFER = 6

    qd_safe = np.deg2rad(30.0)
    qd_max = np.full(7, qd_safe, dtype=float)

    n_sub_full = max(1, int(round(dt_view / model.opt.timestep)))
    n_sub_transfer = max(1, int(0.5 * n_sub_full))

    big_bear_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "big_bear"))
    box_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"))
    if big_bear_id < 0 or box_id < 0:
        raise RuntimeError("Missing big_bear or cardboard_box body.")

    def _set_geom_friction(name: str, slide=3.0, spin=0.03, roll=0.002):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_friction[int(gid), :] = np.array([slide, spin, roll], dtype=float)

    _set_geom_friction("big_bear_col", slide=4.0, spin=0.04, roll=0.003)
    _set_geom_friction("left_fingertip_col", slide=4.0, spin=0.04, roll=0.003)
    _set_geom_friction("right_fingertip_col", slide=4.0, spin=0.04, roll=0.003)

    R0 = data.xmat[ee_body_id].reshape(3, 3).copy()
    R_des, k_up = make_R_with_axis_k_down(R0)
    q_hold = quat_from_mat3(R_des)
    print(f"[INFO] Forcing EE axis to point down (k_up={k_up}).")

    def bezier(p0, p1, p2, p3, s):
        s = float(s)
        return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * (s ** 2) * p2 + (s ** 3) * p3

    g = 1.0
    GRIP_SPEED = 1.2

    z_above = 0.95
    z_pick = 0.76
    z_lift = 1.05
    z_drop_above_box = 0.95
    z_drop_into_box = 0.80

    # render throttle
    render_every_lin = 1
    render_every_bez = 1

    # If you want wrist during transfer, set this True (costs more)
    wrist_during_transfer = True

    next_step_time = time.perf_counter()

    def render_windows(do_wrist: bool, do_fixed: bool):
        if renderer is None:
            return

        MAX_DEPTH_WRIST = 1.0
        MAX_DEPTH_FIXED = 2.5

        if do_wrist:
            # --- wrist pose update (your existing code) ---
            p_b = data.xpos[bracelet_body_id].copy()
            R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
            p_frame = p_b + R_b @ cam_off

            R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()
            tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
            tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

            eye_out = 0.01
            p_frame = p_frame + (R_frame @ (tool_forward * eye_out))

            f_world = _normalize(R_frame @ tool_forward)
            u_world = _normalize(R_frame @ tool_up)

            z_cam_world = _normalize(-f_world)
            y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
            x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
            y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

            R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

            model.cam_pos[wrist_cam_id] = p_frame
            model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

            # --- wrist RGB ---
            renderer.update_scene(data, camera="wrist_rgb")
            wrist_rgb = renderer.render()
            show_wrist_window(wrist_rgb[..., ::-1], title="Wrist Camera (RGB)", w=960, h=540)

            # --- wrist Depth ---
            if renderer_depth is not None:
                renderer_depth.update_scene(data, camera="wrist_rgb")
                wrist_depth = renderer_depth.render()  # float meters
                wrist_depth_vis = depth_to_vis(wrist_depth, max_depth=MAX_DEPTH_WRIST, use_colormap=True)
                show_wrist_window(wrist_depth_vis, title="Wrist Camera (Depth)", w=960, h=540)

        if do_fixed:
            # --- fixed RGB ---
            renderer.update_scene(data, camera="fixed_down")
            fixed_rgb = renderer.render()
            show_wrist_window(fixed_rgb[..., ::-1], title="Fixed Camera (RGB)", w=960, h=540)

            # --- fixed Depth ---
            if renderer_depth is not None:
                renderer_depth.update_scene(data, camera="fixed_down")
                fixed_depth = renderer_depth.render()
                fixed_depth_vis = depth_to_vis(fixed_depth, max_depth=MAX_DEPTH_FIXED, use_colormap=True)
                show_wrist_window(fixed_depth_vis, title="Fixed Camera (Depth)", w=960, h=540)

    def step_realtime():
        nonlocal next_step_time
        next_step_time += dt_view
        now = time.perf_counter()
        # snap if too far behind (prevents runaway lag)
        if now - next_step_time > 0.2:
            next_step_time = now
        sleep_s = next_step_time - now
        if sleep_s > 0:
            time.sleep(sleep_s)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        set_white_environment_visuals(model, viewer)

        try:
            while viewer.is_running():
                mujoco.mj_forward(model, data)

                p_big = data.xpos[big_bear_id].copy()
                p_box = data.xpos[box_id].copy()

                pB0 = np.array([p_big[0], p_big[1], z_above], dtype=float)
                pB1 = np.array([p_big[0], p_big[1], z_pick], dtype=float)
                pB2 = np.array([p_big[0], p_big[1], z_lift], dtype=float)

                pC0 = np.array([p_box[0], p_box[1], z_drop_above_box], dtype=float)
                pC1 = np.array([p_box[0], p_box[1], z_drop_into_box], dtype=float)

                mid_z = 1.10
                mid_y = 0.5 * (p_big[1] + p_box[1])
                c1 = np.array([p_big[0], mid_y, mid_z], dtype=float)
                c2 = np.array([p_box[0], mid_y, mid_z], dtype=float)

                segments = [
                    ("lin", pB0, pB0, 0.4, 0, True),
                    ("lin", pB0, pB1, 1.0, 0, True),

                    ("lin", pB1, pB1, 1.0, -1, True),  # close
                    ("lin", pB1, pB1, 0.5, 0, True),   # settle

                    ("lin", pB1, pB2, 1.2, 0, True),   # lift

                    ("bez", pB2, c1, c2, pC0, 1.8, 0, False),  # transfer

                    ("lin", pC0, pC1, 1.0, 0, True),   # descend into box

                    ("lin", pC1, pC1, 0.8, +1, True),  # open
                    ("lin", pC1, pC1, 0.4, 0, True),   # settle

                    ("lin", pC1, pC0, 1.0, 0, False),  # retreat
                ]

                for seg in segments:
                    if not viewer.is_running():
                        break

                    kind = seg[0]

                    if kind == "lin":
                        _, p0, p1, T, cmd, near_contact = seg
                        iters = IK_ITERS_NEAR if near_contact else IK_ITERS_TRANSFER
                        n_sub = n_sub_full if near_contact else n_sub_transfer
                        render_every = render_every_lin if near_contact else render_every_bez

                        n = max(2, int(np.ceil(float(T) / dt_view)))
                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = (1.0 - s) * p0 + s * p1
                            q_des = q_hold

                            g += float(cmd) * GRIP_SPEED * dt_view
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            q_target = dls_ik_compute_qtarget(
                                model, data, ee_body_id,
                                arm_joint_ids, qpos_adrs, dof_adrs,
                                p_des, q_des,
                                rot_w=2.0,
                                damping=3e-2,
                                dt_step=dt_view,
                                qd_max=qd_max,
                                iters=iters,
                            )
                            data.ctrl[:7] = q_target

                            for _ in range(n_sub):
                                mujoco.mj_step(model, data)

                            viewer.sync()

                            # IMPORTANT: pump GUI events EVERY step (prevents Windows freeze)
                            if cv2 is not None:
                                cv2.waitKey(1)

                            if (renderer is not None) and ((i % render_every) == 0):
                                render_windows(do_wrist=near_contact, do_fixed=True)

                            step_realtime()

                    else:
                        _, p0, p1, p2, p3, T, cmd, _near = seg
                        iters = IK_ITERS_TRANSFER
                        n_sub = n_sub_transfer
                        render_every = render_every_bez

                        n = max(2, int(np.ceil(float(T) / dt_view)))
                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = bezier(p0, p1, p2, p3, s)
                            q_des = q_hold

                            g += float(cmd) * GRIP_SPEED * dt_view
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            q_target = dls_ik_compute_qtarget(
                                model, data, ee_body_id,
                                arm_joint_ids, qpos_adrs, dof_adrs,
                                p_des, q_des,
                                rot_w=2.0,
                                damping=3e-2,
                                dt_step=dt_view,
                                qd_max=qd_max,
                                iters=iters,
                            )
                            data.ctrl[:7] = q_target

                            for _ in range(n_sub):
                                mujoco.mj_step(model, data)

                            viewer.sync()

                            # IMPORTANT: pump GUI events EVERY step
                            if cv2 is not None:
                                cv2.waitKey(1)

                            if (renderer is not None) and ((i % render_every) == 0):
                                # during transfer: fixed cam only, wrist optional
                                render_windows(do_wrist=wrist_during_transfer, do_fixed=True)

                            step_realtime()

        except KeyboardInterrupt:
            pass
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()