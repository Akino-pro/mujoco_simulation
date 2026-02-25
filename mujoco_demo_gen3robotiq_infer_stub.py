import os
import time
import json
import base64
import argparse
import signal
import concurrent.futures

import numpy as np
import mujoco
import mujoco.viewer

from helper_functions import (
    _get_joint_ids_by_name,
    find_ee_target,
    build_gripper_controls,
    set_white_environment_visuals,
    apply_gripper,
    show_wrist_window,
)

try:
    import cv2
except Exception:
    cv2 = None

try:
    import requests
except Exception:
    requests = None

try:
    from websocket import create_connection
except Exception:
    create_connection = None

URDF_PATH = r"gen3_modified.urdf"
_WARNED_NO_REQUESTS = False
_WARNED_NO_WS = False
_STOP_REQUESTED = False


def _handle_sigint(_signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[INFO] Ctrl+C received. Exiting...")


def _resolve_scene_path(urdf_dir: str) -> str:
    env_scene = os.environ.get("MUJOCO_SCENE_XML", "").strip()
    if env_scene:
        scene = env_scene if os.path.isabs(env_scene) else os.path.join(urdf_dir, env_scene)
        if not os.path.exists(scene):
            raise FileNotFoundError(f"MUJOCO_SCENE_XML points to missing file: {scene}")
        print(f"[INFO] Using scene from MUJOCO_SCENE_XML: {scene}")
        return scene

    convex_scene = os.path.join(urdf_dir, "robotsuit_convex.xml")
    cubes_scene = os.path.join(urdf_dir, "robotsuit_cubes.xml")
    if os.path.exists(convex_scene):
        print(f"[INFO] Using convex scene: {convex_scene}")
        return convex_scene
    print(f"[INFO] Using cube-proxy scene: {cubes_scene}")
    return cubes_scene


def _resolve_collision_viz_choice() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--collision-viz",
        type=str,
        default="",
        help="Collision visualization mode: 1/off/hide or 2/on/show.",
    )
    args, _ = parser.parse_known_args()
    raw = str(args.collision_viz).strip().lower()

    if raw in {"1", "off", "hide"}:
        return False
    if raw in {"2", "on", "show"}:
        return True

    print("[SELECT] Collision visualization mode:")
    print("  1) Hide collision boxes/meshes")
    print("  2) Show collision boxes/meshes (current behavior)")
    while True:
        choice = input("Enter 1 or 2 (default 2): ").strip()
        if choice == "":
            return True
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Please enter 1 or 2.")


def _apply_collision_viz_mode(model: mujoco.MjModel, show_collision_viz: bool):
    if show_collision_viz:
        return
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname and "_col" in gname:
            model.geom_rgba[gid, 3] = 0.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def quat_from_mat3(R: np.ndarray) -> np.ndarray:
    return mat3_to_quat(R)


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


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = r / angle
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def dls_ik_compute_qtarget(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_body_id: int,
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

        ep = p_des - p_cur
        q_err = quat_mul(q_des, quat_conj(q_cur))
        er = quat_to_rotvec(q_err)
        e = np.concatenate([ep, rot_w * er], axis=0)

        Jp = np.zeros((3, model.nv), dtype=float)
        Jr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, Jp, Jr, ee_body_id)
        J6 = np.vstack([Jp[:, dof_adrs], rot_w * Jr[:, dof_adrs]])

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


def _resize_and_encode_jpeg_b64(img_rgb: np.ndarray | None, width: int, height: int, quality: int) -> str | None:
    if img_rgb is None or cv2 is None:
        return None
    if img_rgb.shape[1] != width or img_rgb.shape[0] != height:
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    img_bgr = img_rgb[..., ::-1]
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _to_remote_payload(observation: dict, prompt: str, image_w: int, image_h: int, jpeg_quality: int) -> dict:
    return {
        "prompt": prompt,
        "observation/joint_position": np.asarray(observation["observation/joint_position"], dtype=float).tolist(),
        "observation/gripper_position": np.asarray(observation["observation/gripper_position"], dtype=float).tolist(),
        "observation/exterior_image_1_left_b64_jpg": _resize_and_encode_jpeg_b64(
            observation["observation/exterior_image_1_left"], image_w, image_h, jpeg_quality
        ),
        "observation/wrist_image_left_b64_jpg": _resize_and_encode_jpeg_b64(
            observation["observation/wrist_image_left"], image_w, image_h, jpeg_quality
        ),
    }


def _extract_actions(data: dict) -> np.ndarray | None:
    # Prefer actions7 if server provides both; avoids consuming padded dummy dim.
    if "actions7" in data:
        return np.asarray(data["actions7"], dtype=float)
    if "actions" in data:
        return np.asarray(data["actions"], dtype=float)
    return None


def _infer_http(payload: dict, endpoint: str, timeout_s: float) -> np.ndarray | None:
    global _WARNED_NO_REQUESTS
    if requests is None:
        if not _WARNED_NO_REQUESTS:
            print("[WARN] requests is not installed. Cannot use HTTP inference.")
            _WARNED_NO_REQUESTS = True
        return None
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            print(f"[WARN] HTTP {r.status_code} from server: {r.text[:200]}")
            return None
        return _extract_actions(r.json())
    except Exception as e:
        print(f"[WARN] HTTP inference failed: {e}")
        return None


def _infer_ws(payload: dict, endpoint: str, timeout_s: float) -> np.ndarray | None:
    global _WARNED_NO_WS
    if create_connection is None:
        if not _WARNED_NO_WS:
            print("[WARN] websocket-client is not installed. Cannot use websocket inference.")
            _WARNED_NO_WS = True
        return None
    try:
        ws = create_connection(endpoint, timeout=timeout_s)
        try:
            ws.send(json.dumps(payload))
            msg = ws.recv()
        finally:
            ws.close()
        return _extract_actions(json.loads(msg))
    except Exception as e:
        print(f"[WARN] Websocket inference failed: {e}")
        return None


def request_action_chunk_from_model(
    observation: dict,
    *,
    transport: str,
    endpoint: str,
    timeout_s: float,
    prompt: str,
    image_w: int,
    image_h: int,
    jpeg_quality: int,
) -> np.ndarray | None:
    payload = _to_remote_payload(
        observation=observation,
        prompt=prompt,
        image_w=image_w,
        image_h=image_h,
        jpeg_quality=jpeg_quality,
    )
    if payload["observation/exterior_image_1_left_b64_jpg"] is None:
        return None
    if payload["observation/wrist_image_left_b64_jpg"] is None:
        return None

    if transport == "http":
        return _infer_http(payload, endpoint=endpoint, timeout_s=timeout_s)
    if transport == "ws":
        return _infer_ws(payload, endpoint=endpoint, timeout_s=timeout_s)
    raise ValueError(f"Unknown transport '{transport}'")


def _validate_chunk(chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunk, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (7, 8):
        raise ValueError(f"Expected action chunk shape [H,7] or [H,8], got {arr.shape}")
    return arr


def _gripper_cmd_to_discrete(v: float) -> float:
    if v > 0.33:
        return 1.0
    if v < -0.33:
        return -1.0
    return 0.0


def _copy_obs_for_request(observation: dict) -> dict:
    out = {}
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "ws"], default="http")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/infer")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--prompt", type=str, default="pick and place")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--action-mode", choices=["libero_ee_delta", "direct_joint", "delta_joint"], default="libero_ee_delta")
    parser.add_argument("--prefetch-steps", type=int, default=3)
    parser.add_argument("--query-interval-s", type=float, default=0.0)
    parser.add_argument("--joint-delta-scale", type=float, default=0.05)
    parser.add_argument("--ee-pos-scale", type=float, default=1.0)
    parser.add_argument("--ee-rot-scale", type=float, default=1.0)
    parser.add_argument("--ee-z-min", type=float, default=0.60)
    parser.add_argument("--ee-z-max", type=float, default=1.25)
    parser.add_argument("--midway-steps", type=int, default=4)
    args = parser.parse_args()

    show_collision_viz = _resolve_collision_viz_choice()
    print(f"[INFO] Collision visualization: {'ON' if show_collision_viz else 'OFF'}")
    print(f"[INFO] Remote inference: transport={args.transport}, endpoint={args.endpoint}")
    print(f"[INFO] Action mode: {args.action_mode}")
    print(f"[INFO] Mid-way interpolation steps per action: {max(1, int(args.midway_steps))}")

    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)
    os.chdir(urdf_dir)

    scene_path = _resolve_scene_path(urdf_dir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    _apply_collision_viz_mode(model, show_collision_viz)
    data = mujoco.MjData(model)

    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    # Table-facing startup posture (approximate), then clamped to joint limits below.
    q_init = np.array([3.14159, 0.30, 0.0, 1.45, 0.0, 0.90, 0.0], dtype=float)
    for i in range(7):
        data.qpos[qpos_adrs[i]] = float(q_init[i])

    gripper_cmd_state = 0.0
    gripper_pos_01 = 0.0

    grip_controls, lo_m, hi_m = build_gripper_controls(model)
    apply_gripper(model, data, grip_controls, lo_m, hi_m, gripper_pos_01)

    mujoco.mj_forward(model, data)
    data.ctrl[:7] = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)
    hold_q_target = np.array(data.ctrl[:7], dtype=float)

    wrist_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb"))
    fixed_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_down"))
    if wrist_cam_id < 0 or fixed_cam_id < 0:
        raise RuntimeError("Missing wrist_rgb or fixed_down camera in XML.")
    bracelet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_bracelet_link")
    if bracelet_body_id < 0:
        raise RuntimeError("Could not find body 'gen3_bracelet_link'.")
    bracelet_body_id = int(bracelet_body_id)
    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)

    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    renderer = None
    if cv2 is not None:
        renderer = mujoco.Renderer(model, width=960, height=540)

    dt_view = 1.0 / 30.0
    grip_speed = 1.2
    n_sub = max(1, int(round(dt_view / model.opt.timestep)))

    # IK settings (used in libero_ee_delta mode)
    qd_max = np.full(7, np.deg2rad(35.0), dtype=float)
    p_des = data.xpos[ee_body_id].copy()
    q_des = mat3_to_quat(data.xmat[ee_body_id].reshape(3, 3).copy())

    action_chunk = None
    chunk_step = 0
    next_chunk = None
    last_query_t = 0.0
    midway_steps = max(1, int(args.midway_steps))
    interp_start_q = hold_q_target.copy()
    interp_target_q = hold_q_target.copy()
    interp_g_cmd = 0.0
    interp_idx = midway_steps

    print("[INFO] Ready. Holding start pose and asynchronously requesting model chunks.")

    next_step_time = time.perf_counter()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_future = None
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance *= 1.8
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            set_white_environment_visuals(model, viewer)

            try:
                while viewer.is_running() and (not _STOP_REQUESTED):
                    mujoco.mj_forward(model, data)

                    wrist_rgb = None
                    fixed_rgb = None
                    if renderer is not None:
                        # Keep wrist camera pose update consistent with mujoco_demo_gen3robotiq.py.
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

                        renderer.update_scene(data, camera="wrist_rgb")
                        wrist_rgb = renderer.render()
                        show_wrist_window(wrist_rgb[..., ::-1], title="Wrist Camera (RGB)", w=960, h=540)

                        renderer.update_scene(data, camera="fixed_down")
                        fixed_rgb = renderer.render()
                        show_wrist_window(fixed_rgb[..., ::-1], title="Fixed Camera (RGB)", w=960, h=540)

                    joint_obs = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)
                    gripper_obs = np.array([gripper_pos_01], dtype=float)
                    observation = {
                        "observation/exterior_image_1_left": fixed_rgb,
                        "observation/wrist_image_left": wrist_rgb,
                        "observation/joint_position": joint_obs,
                        "observation/gripper_position": gripper_obs,
                    }

                    if pending_future is not None and pending_future.done():
                        try:
                            result = pending_future.result()
                            if result is not None:
                                result = _validate_chunk(result)
                                next_chunk = result
                                print(f"[INFO] Received action chunk with H={result.shape[0]} D={result.shape[1]}")
                        except Exception as e:
                            print(f"[WARN] Inference future failed: {e}")
                        pending_future = None

                    steps_left = 0 if action_chunk is None else max(0, len(action_chunk) - chunk_step)
                    should_query = (
                        pending_future is None
                        and next_chunk is None
                        and (action_chunk is None or steps_left <= args.prefetch_steps)
                        and ((time.perf_counter() - last_query_t) >= args.query_interval_s)
                    )
                    if should_query:
                        obs_for_request = _copy_obs_for_request(observation)
                        pending_future = pool.submit(
                            request_action_chunk_from_model,
                            obs_for_request,
                            transport=args.transport,
                            endpoint=args.endpoint,
                            timeout_s=args.timeout_s,
                            prompt=args.prompt,
                            image_w=args.image_width,
                            image_h=args.image_height,
                            jpeg_quality=args.jpeg_quality,
                        )
                        last_query_t = time.perf_counter()

                    if (action_chunk is None or chunk_step >= len(action_chunk)) and next_chunk is not None:
                        action_chunk = next_chunk
                        next_chunk = None
                        chunk_step = 0

                    if action_chunk is None or chunk_step >= len(action_chunk):
                        # Hold current pose if no chunk is ready yet.
                        data.ctrl[:7] = hold_q_target
                        gripper_cmd_state = 0.0
                        interp_idx = midway_steps
                    else:
                        if interp_idx >= midway_steps:
                            action = action_chunk[chunk_step]
                            chunk_step += 1
                            interp_start_q = hold_q_target.copy()

                            if args.action_mode == "direct_joint":
                                interp_target_q = np.array(action[:7], dtype=float)
                                g_raw = float(action[7]) if action.shape[0] >= 8 else 0.0
                                interp_g_cmd = _gripper_cmd_to_discrete(g_raw)

                            elif args.action_mode == "delta_joint":
                                q_now = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)
                                interp_target_q = q_now + args.joint_delta_scale * action[:7]
                                g_raw = float(action[7]) if action.shape[0] >= 8 else 0.0
                                interp_g_cmd = _gripper_cmd_to_discrete(g_raw)

                            else:
                                # libero_ee_delta: interpret action[0:3]=dpos, action[3:6]=drotvec,
                                # and gripper from action[6] (if D=7) or action[7] (if D=8 and non-padded).
                                a = np.zeros(7, dtype=float)
                                a[: min(7, action.shape[0])] = action[: min(7, action.shape[0])]

                                p_des = p_des + a[:3]
                                p_des[2] = float(np.clip(p_des[2], args.ee_z_min, args.ee_z_max))

                                dq = rotvec_to_quat(a[3:6])
                                q_des = _normalize(quat_mul(dq, q_des))

                                interp_target_q = dls_ik_compute_qtarget(
                                    model,
                                    data,
                                    ee_body_id,
                                    qpos_adrs,
                                    dof_adrs,
                                    p_des,
                                    q_des,
                                    rot_w=1.8,
                                    damping=3e-2,
                                    dt_step=dt_view,
                                    qd_max=qd_max,
                                    iters=10,
                                )

                                g_raw = float(action[6])
                                if action.shape[0] >= 8 and abs(float(action[7])) > 1e-6:
                                    g_raw = float(action[7])
                                interp_g_cmd = _gripper_cmd_to_discrete(g_raw)

                            interp_idx = 0

                        alpha = float(interp_idx + 1) / float(midway_steps)
                        data.ctrl[:7] = (1.0 - alpha) * interp_start_q + alpha * interp_target_q
                        gripper_cmd_state = interp_g_cmd
                        interp_idx += 1
                        if interp_idx >= midway_steps:
                            hold_q_target = interp_target_q.copy()

                    gripper_pos_01 += gripper_cmd_state * grip_speed * dt_view
                    gripper_pos_01 = float(np.clip(gripper_pos_01, 0.0, 1.0))
                    apply_gripper(model, data, grip_controls, lo_m, hi_m, gripper_pos_01)

                    for _ in range(n_sub):
                        mujoco.mj_step(model, data)

                    viewer.sync()
                    if cv2 is not None:
                        cv2.waitKey(1)

                    next_step_time += dt_view
                    now = time.perf_counter()
                    if now - next_step_time > 0.2:
                        next_step_time = now
                    sleep_s = next_step_time - now
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt received. Exiting...")
            finally:
                if cv2 is not None:
                    cv2.destroyAllWindows()
    finally:
        if pending_future is not None and (not pending_future.done()):
            pending_future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
