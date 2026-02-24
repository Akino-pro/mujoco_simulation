import os
import time
import re
import socket
import struct
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from helper_functions import (
    _get_joint_ids_by_name, find_ee_target, build_gripper_controls, quat_from_mat3,
    set_white_environment_visuals, ik_step_dls, apply_gripper,
    make_R_with_axis_k_down
)

try:
    import cv2  # only used for JPEG encoding
except Exception:
    cv2 = None

URDF_PATH = r"robot/gen3_modified.urdf"
SCENE_XML = r"A.xml"

HOST = "127.0.0.1"
PORT_RGB = 5005

RENDER_W, RENDER_H = 480, 480
WRIST_REFRESH_EVERY = 2
JPEG_QUALITY = 80


def _extract_tag_inner(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(rf"<{tag}\b[^>]*/>", text, flags=re.DOTALL)
    if m2:
        return ""
    return ""


def _strip_comments(s: str) -> str:
    return re.sub(r"<!--.*?-->", "", s or "", flags=re.DOTALL)


def _ensure_nonempty_asset_children(asset_inner: str) -> str:
    keepalive = '<material name="__asset_keepalive" rgba="1 1 1 1"/>\n'
    s = (asset_inner or "").strip()
    if not s or not _strip_comments(s).strip():
        return keepalive
    return s + "\n"


def _ensure_nonempty_body_children(worldbody_inner: str) -> str:
    keepalive = '<geom name="__body_keepalive" type="sphere" size="0.0001" rgba="0 0 0 0"/>\n'
    s = (worldbody_inner or "").strip()
    if not s or not _strip_comments(s).strip():
        return keepalive
    return s + "\n"


def _normalize(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-9:
        return v
    return v / nrm


def lookat_quat_from_tool_axes(
    p_tool: np.ndarray,
    p_target: np.ndarray,
    tool_forward_local: np.ndarray,
    tool_up_local: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
) -> np.ndarray:
    # desired forward direction in world
    f = p_target - p_tool
    f = _normalize(f)

    # choose an up direction in world that is orthogonal to f
    u = world_up - np.dot(world_up, f) * f
    u = _normalize(u)

    # right-handed basis
    r = _normalize(np.cross(u, f))
    u = _normalize(np.cross(f, r))

    # local basis from tool axes
    tf = _normalize(tool_forward_local)
    tu = _normalize(tool_up_local)
    tr = _normalize(np.cross(tu, tf))  # local right

    # map local basis -> world basis
    R_world = np.column_stack([r, u, f]) @ np.linalg.inv(np.column_stack([tr, tu, tf]))
    return quat_from_mat3(R_world)


class FrameSender:
    """Length-prefixed TCP sender. Sends bytes payloads to localhost receiver."""
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        print(f"[INFO] Connected to wrist receiver at {host}:{port}")

    def send(self, payload: bytes):
        header = struct.pack("!I", len(payload))  # 4-byte big-endian length
        self.sock.sendall(header + payload)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    if cv2 is None:
        raise RuntimeError("opencv-python is required for JPEG encoding (cv2.imencode).")

    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)


    # keep your pipeline step
    robot_model = mujoco.MjModel.from_xml_path(urdf_abs)
    robot_mjcf_path = os.path.join(urdf_dir, "_gen3_from_urdf.xml")
    mujoco.mj_saveLastXML(robot_mjcf_path, robot_model)

    robot_text = Path(robot_mjcf_path).read_text(encoding="utf-8", errors="ignore")
    _ = _ensure_nonempty_asset_children(_extract_tag_inner(robot_text, "asset"))
    _ = _ensure_nonempty_body_children(_extract_tag_inner(robot_text, "worldbody"))

    scene_abs = os.path.abspath(SCENE_XML)
    model = mujoco.MjModel.from_xml_path(scene_abs)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # camera id
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb")
    if wrist_cam_id < 0:
        raise RuntimeError("Camera 'wrist_rgb' not found in model XML")
    wrist_cam_id = int(wrist_cam_id)

    # tomato look-at site id
    tomato_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tomato_aim")
    if tomato_site_id < 0:
        raise RuntimeError("Site 'tomato_aim' not found. Did you add it to robotsuit_cubes.xml?")
    tomato_site_id = int(tomato_site_id)

    # Arm joints + EEF body
    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    grip_controls, lo_m, hi_m = build_gripper_controls(model)

    bracelet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_bracelet_link")
    if bracelet_body_id < 0:
        raise RuntimeError("Could not find body 'gen3_bracelet_link'")
    bracelet_body_id = int(bracelet_body_id)

    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)

    renderer = mujoco.Renderer(model, width=RENDER_W, height=RENDER_H)

    # [新增] 专门用于提取深度图的渲染器
    renderer_depth = mujoco.Renderer(model, width=RENDER_W, height=RENDER_H)
    renderer_depth.enable_depth_rendering()

    sender_rgb = FrameSender(HOST, PORT_RGB)

    # IK timing
    dt = 1.0 / 60.0
    IK_ITERS = 10
    qd_safe = np.deg2rad(10.0)
    qd_max = np.full(7, qd_safe, dtype=float)
    dt_inner = dt / IK_ITERS

    # Path / motion
    p_green = np.array([0.45, 0.18, 0.825], dtype=float)
    p_blue  = np.array([0.45, -0.18, 0.825], dtype=float)
    z_above = 0.98
    z_pick  = 0.86

    R0 = data.xmat[ee_body_id].reshape(3, 3).copy()
    R_des, k_up = make_R_with_axis_k_down(R0)
    q_hold = quat_from_mat3(R_des)
    print(f"[INFO] EE axis pointing up at start was body axis index {k_up} (0=x,1=y,2=z). Forcing it to point down.")

    def bezier(p0, p1, p2, p3, s):
        s = float(s)
        return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * (s ** 2) * p2 + (s ** 3) * p3

    pA0 = np.array([p_green[0], p_green[1], z_above], dtype=float)
    pA1 = np.array([p_green[0], p_green[1], z_pick], dtype=float)
    pA2 = np.array([p_green[0], p_green[1], z_above], dtype=float)

    pB0 = np.array([p_blue[0], p_blue[1], z_above], dtype=float)
    pB1 = np.array([p_blue[0], p_blue[1], z_pick], dtype=float)
    pB2 = np.array([p_blue[0], p_blue[1], z_above], dtype=float)

    mid_z = 1.10
    c1 = np.array([p_green[0], 0.00, mid_z], dtype=float)
    c2 = np.array([p_blue[0],  0.00, mid_z], dtype=float)

    segments = [
        ("lin", pA0, pA0, 1.0, 0),
        ("lin", pA0, pA1, 1.2, 0),
        ("lin", pA1, pA1, 0.8, -1),
        ("lin", pA1, pA2, 1.2, 0),
        ("bez", pA2, c1, c2, pB0, 2.4, 0),
        ("lin", pB0, pB1, 1.2, 0),
        ("lin", pB1, pB1, 0.8, +1),
        ("lin", pB1, pB2, 1.0, 0),
        ("bez", pB2, c2, c1, pA0, 2.4, 0),
    ]

    g = 0.8
    GRIP_SPEED = 0.9

    tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
    tool_up      = np.array([0.0, -1.0,  0.0], dtype=float)
    eye_out = 0.01
    eye_up  = 0.0

    frame_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        set_white_environment_visuals(model, viewer)

        next_step_time = time.perf_counter()

        try:
            while viewer.is_running():
                for seg in segments:
                    kind = seg[0]

                    if kind == "lin":
                        _, p0, p1, T, cmd = seg
                        n = max(2, int(np.ceil(float(T) / dt)))

                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = (1.0 - s) * p0 + s * p1

                            p_target = data.site_xpos[tomato_site_id].copy()
                            p_tool = data.xpos[ee_body_id].copy()

                            q_des = lookat_quat_from_tool_axes(
                                p_tool=p_tool,
                                p_target=p_target,
                                tool_forward_local=tool_forward,
                                tool_up_local=tool_up
                            )

                            g += float(cmd) * GRIP_SPEED * dt
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            for _ in range(IK_ITERS):
                                err = ik_step_dls(
                                    model, data, ee_body_id,
                                    arm_joint_ids, qpos_adrs, dof_adrs,
                                    p_des, q_des,
                                    rot_w=2.0,
                                    damping=3e-2,
                                    dt_step=dt_inner,
                                    qd_max=qd_max,
                                )
                                if err < 1e-4:
                                    break

                            # ---- wrist cam pose ----
                            p_b = data.xpos[bracelet_body_id].copy()
                            R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                            p_frame = p_b + R_b @ cam_off

                            R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()
                            p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (R_frame @ (tool_up * eye_up))

                            f_world = _normalize(R_frame @ tool_forward)
                            u_world = _normalize(R_frame @ tool_up)

                            z_cam_world = _normalize(-f_world)
                            y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
                            x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
                            y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                            R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

                            model.cam_pos[wrist_cam_id]  = p_frame
                            model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

                            viewer.sync()

                            # ---- send wrist RGB frame ----
                            frame_counter += 1
                            if (frame_counter % WRIST_REFRESH_EVERY) == 0:
                                # 1. 渲染 RGB
                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                if wrist_rgb.dtype != np.uint8:
                                    wrist_rgb = np.clip(wrist_rgb, 0, 255).astype(np.uint8)
                                wrist_bgr = wrist_rgb[:, :, ::-1]

                                # 2. [新增] 渲染 深度图 (Depth)
                                renderer_depth.update_scene(data, camera="wrist_rgb")
                                depth_map = renderer_depth.render()  # 返回的是真实物理距离（米）的浮点数组

                                # 将深度距离映射为 0-255 的可视图像。
                                # 这里设置 MAX_DEPTH=1.5 米，意味着距离相机 1.5 米外的物体都会显示为最大颜色。
                                # 扫描西红柿时距离通常很近，你可以根据实际效果微调这个 1.5 的值。
                                MAX_DEPTH = 1.5
                                depth_norm = np.clip((depth_map / MAX_DEPTH) * 255.0, 0, 255).astype(np.uint8)

                                # 给深度图加上伪彩色（JET色带），视觉效果会非常专业（红-黄-绿-蓝）
                                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                                # 3. [新增] 将 RGB 和 深度图 左右拼接起来
                                combined_img = np.hstack((wrist_bgr, depth_color))

                                # 4. 压缩并发送这张长图
                                ok, buf = cv2.imencode(
                                    ".jpg", combined_img,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                                )
                                if ok:
                                    try:
                                        sender_rgb.send(buf.tobytes())
                                    except BrokenPipeError:
                                        print("[WARN] RGB receiver disconnected (BrokenPipe). Start MAC2.py first.")
                                        return

                            next_step_time += dt
                            sleep_s = next_step_time - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)

                    else:  # "bez"
                        _, p0, p1, p2, p3, T, cmd = seg
                        n = max(2, int(np.ceil(float(T) / dt)))

                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = bezier(p0, p1, p2, p3, s)

                            p_target = data.site_xpos[tomato_site_id].copy()
                            p_tool = data.xpos[ee_body_id].copy()

                            q_des = lookat_quat_from_tool_axes(
                                p_tool=p_tool,
                                p_target=p_target,
                                tool_forward_local=tool_forward,
                                tool_up_local=tool_up
                            )

                            g += float(cmd) * GRIP_SPEED * dt
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            for _ in range(IK_ITERS):
                                err = ik_step_dls(
                                    model, data, ee_body_id,
                                    arm_joint_ids, qpos_adrs, dof_adrs,
                                    p_des, q_des,
                                    rot_w=2.0,
                                    damping=3e-2,
                                    dt_step=dt_inner,
                                    qd_max=qd_max,
                                )
                                if err < 1e-4:
                                    break

                            p_b = data.xpos[bracelet_body_id].copy()
                            R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                            p_frame = p_b + R_b @ cam_off

                            R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()
                            p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (R_frame @ (tool_up * eye_up))

                            f_world = _normalize(R_frame @ tool_forward)
                            u_world = _normalize(R_frame @ tool_up)

                            z_cam_world = _normalize(-f_world)
                            y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
                            x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
                            y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                            R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

                            model.cam_pos[wrist_cam_id]  = p_frame
                            model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

                            viewer.sync()

                            frame_counter += 1
                            if (frame_counter % WRIST_REFRESH_EVERY) == 0:
                                # 1. 渲染 RGB
                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                if wrist_rgb.dtype != np.uint8:
                                    wrist_rgb = np.clip(wrist_rgb, 0, 255).astype(np.uint8)
                                wrist_bgr = wrist_rgb[:, :, ::-1]

                                # 2. [新增] 渲染 深度图 (Depth)
                                renderer_depth.update_scene(data, camera="wrist_rgb")
                                depth_map = renderer_depth.render()  # 返回的是真实物理距离（米）的浮点数组

                                # 将深度距离映射为 0-255 的可视图像。
                                # 这里设置 MAX_DEPTH=1.5 米，意味着距离相机 1.5 米外的物体都会显示为最大颜色。
                                # 扫描西红柿时距离通常很近，你可以根据实际效果微调这个 1.5 的值。
                                MAX_DEPTH = 1.5
                                depth_norm = np.clip((depth_map / MAX_DEPTH) * 255.0, 0, 255).astype(np.uint8)

                                # 给深度图加上伪彩色（JET色带），视觉效果会非常专业（红-黄-绿-蓝）
                                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                                # 3. [新增] 将 RGB 和 深度图 左右拼接起来
                                combined_img = np.hstack((wrist_bgr, depth_color))

                                # 4. 压缩并发送这张长图
                                ok, buf = cv2.imencode(
                                    ".jpg", combined_img,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                                )
                                if ok:
                                    try:
                                        sender_rgb.send(buf.tobytes())
                                    except BrokenPipeError:
                                        print("[WARN] RGB receiver disconnected (BrokenPipe). Start MAC2.py first.")
                                        return

                            next_step_time += dt
                            sleep_s = next_step_time - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                renderer.close()
            except Exception:
                pass
            sender_rgb.close()


if __name__ == "__main__":
    main()