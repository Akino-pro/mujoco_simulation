import cv2
import mujoco
import numpy as np


# -----------------------------
# Quaternion / rotation helpers
# -----------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # [w,x,y,z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_normalize(q0 + s * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - s) * theta) / sin_theta
    b = np.sin(s * theta) / sin_theta
    return quat_normalize(a * q0 + b * q1)


def mat3_from_xmat(xmat9: np.ndarray) -> np.ndarray:
    return np.array(xmat9, dtype=float).reshape(3, 3)


def quat_from_mat3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def rotvec_from_quat(q_err: np.ndarray) -> np.ndarray:
    q_err = quat_normalize(q_err)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    v = q_err[1:]
    sin_half = np.linalg.norm(v)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=float)
    axis = v / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return axis * angle


# -----------------------------
# MuJoCo helpers
# -----------------------------
def _get_joint_ids_by_name(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    joint_ids: list[int] = []
    missing: list[str] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
        else:
            joint_ids.append(int(jid))
    if missing:
        all_joint_names = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm:
                all_joint_names.append(nm)
        raise RuntimeError(
            "Missing joints:\n"
            f"  {missing}\n\n"
            "Example available joints:\n"
            "  " + ", ".join(all_joint_names[:120]) + (" ..." if len(all_joint_names) > 120 else "")
        )
    return joint_ids


def find_ee_target(model: mujoco.MjModel) -> tuple[int, str]:
    candidates = [
        "gen3_robotiq_85_base_link",
        "gen3_end_effector_link",
        "gen3_bracelet_link",
    ]
    for nm in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return int(bid), nm
    last_body = model.nbody - 1
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, last_body) or f"body_{last_body}"
    return int(last_body), nm


def clamp_to_joint_limits(model: mujoco.MjModel, joint_ids: list[int], q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[i] = float(np.clip(q[i], lo, hi))
    return q


def _get_body_id_any(model: mujoco.MjModel, candidates: list[str]) -> tuple[int, str] | None:
    for nm in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return int(bid), nm
    return None


def _get_viewport_and_ctx(viewer):
    """
    Different mujoco.viewer versions expose these differently.
    Returns (width, height, ctx_object).
    """
    # ctx
    ctx = getattr(viewer, "ctx", None)
    if ctx is None:
        ctx = getattr(viewer, "context", None)

    # viewport rect (width/height)
    vp = getattr(viewer, "viewport", None)
    if vp is not None and hasattr(vp, "width") and hasattr(vp, "height"):
        return int(vp.width), int(vp.height), ctx

    # common fallbacks
    rect = getattr(viewer, "_rect", None)
    if rect is not None and hasattr(rect, "width") and hasattr(rect, "height"):
        return int(rect.width), int(rect.height), ctx

    # last resort: try window size fields (if any)
    w = getattr(viewer, "width", None)
    h = getattr(viewer, "height", None)
    if w is not None and h is not None:
        return int(w), int(h), ctx

    raise RuntimeError("Could not access viewer viewport size. Tell me your mujoco.__version__ and viewer attributes.")


# -----------------------------
# IK (damped least squares) + safe speed limit
# -----------------------------
def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = data.xpos[body_id].copy()
    R = mat3_from_xmat(data.xmat[body_id].copy())
    q = quat_from_mat3(R)
    return p, q


def jacobian_6d_body(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return np.vstack([jacp, jacr])


def clamp_dq_by_speed(dq: np.ndarray, qd_max: np.ndarray, dt_step: float) -> np.ndarray:
    dq = np.asarray(dq, dtype=float)
    qd_max = np.asarray(qd_max, dtype=float)
    dq_max = qd_max * float(dt_step)
    return np.clip(dq, -dq_max, dq_max)


def ik_step_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    arm_joint_ids: list[int],
    qpos_adrs: list[int],
    dof_adrs: list[int],
    p_des: np.ndarray,
    q_des: np.ndarray,
    pos_w: float = 1.0,
    rot_w: float = 0.6,
    damping: float = 2e-2,
    step_scale: float = 0.9,
    dt_step: float = 1.0 / 600.0,
    qd_max: np.ndarray | None = None,   # rad/s, len=7
) -> float:
    p_cur, q_cur = get_body_pose(model, data, body_id)

    e_pos = (p_des - p_cur) * pos_w
    q_err = quat_mul(q_des, quat_conj(q_cur))
    e_rot = rotvec_from_quat(q_err) * rot_w
    e6 = np.hstack([e_pos, e_rot])
    err = float(np.linalg.norm(e6))

    J = jacobian_6d_body(model, data, body_id)
    J_arm = J[:, dof_adrs]  # 6 x 7

    A = (J_arm @ J_arm.T) + (damping ** 2) * np.eye(6)
    y = np.linalg.solve(A, e6)
    dq = J_arm.T @ y

    dq = step_scale * dq
    if qd_max is not None:
        dq = clamp_dq_by_speed(dq, qd_max, dt_step)

    q = np.array([data.qpos[adr] for adr in qpos_adrs], dtype=float)
    q_new = q + dq
    q_new = clamp_to_joint_limits(model, arm_joint_ids, q_new)

    for adr, val in zip(qpos_adrs, q_new):
        data.qpos[adr] = float(val)

    mujoco.mj_forward(model, data)
    return err


# -----------------------------
# Gripper explicit mimic control
# -----------------------------
def build_gripper_controls(model: mujoco.MjModel):
    MASTER_CANDIDATES = [
        "gen3_robotiq_85_left_knuckle_joint",
        "robotiq_85_left_knuckle_joint",
        "left_knuckle_joint",
    ]

    MIMICS = [
        ("gen3_robotiq_85_right_knuckle_joint",       -1.0),
        ("gen3_robotiq_85_left_inner_knuckle_joint",  +1.0),
        ("gen3_robotiq_85_right_inner_knuckle_joint", -1.0),
        ("gen3_robotiq_85_left_finger_tip_joint",     -1.0),
        ("gen3_robotiq_85_right_finger_tip_joint",    +1.0),
    ]

    def _info(jname: str):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            return None
        jid = int(jid)
        adr = int(model.jnt_qposadr[jid])
        if model.jnt_limited[jid]:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        else:
            lo, hi = None, None
        return adr, lo, hi

    master_name = None
    master_info = None
    for nm in MASTER_CANDIDATES:
        info = _info(nm)
        if info is not None:
            master_name = nm
            master_info = info
            break

    if master_info is None:
        print("[WARN] Could not find a gripper master joint. Gripper disabled.")
        return [], 0.0, 0.8

    adr_m, lo_m, hi_m = master_info
    if lo_m is None or hi_m is None:
        lo_m, hi_m = 0.0, 0.8

    controls = [(adr_m, lo_m, hi_m, 1.0, master_name)]
    for jname, mult in MIMICS:
        info = _info(jname)
        if info is None:
            print(f"[WARN] Missing gripper mimic joint '{jname}' (skipping)")
            continue
        adr, lo, hi = info
        controls.append((adr, lo, hi, float(mult), jname))

    print(f"[INFO] Gripper master '{master_name}' range=[{lo_m:.3f}, {hi_m:.3f}]")
    return controls, lo_m, hi_m


def apply_gripper(model: mujoco.MjModel, data: mujoco.MjData, controls, lo_m, hi_m, g01: float):
    if not controls:
        return
    q_master = lo_m + g01 * (hi_m - lo_m)
    for adr, lo, hi, mult, _name in controls:
        q = float(mult * q_master)
        if lo is not None and hi is not None:
            lo2, hi2 = (lo, hi) if hi >= lo else (hi, lo)
            q = float(np.clip(q, lo2, hi2))
        data.qpos[adr] = q
    mujoco.mj_forward(model, data)


def set_white_environment_visuals(model: mujoco.MjModel, viewer=None):
    try:
        model.vis.headlight.ambient[:]  = [0.85, 0.85, 0.85]
        model.vis.headlight.diffuse[:]  = [0.90, 0.90, 0.90]
        model.vis.headlight.specular[:] = [0.20, 0.20, 0.20]
    except Exception:
        pass
    try:
        rgba = model.vis.rgba
        for field in ["haze", "fog", "skybox"]:
            if hasattr(rgba, field):
                getattr(rgba, field)[:] = [1.0, 1.0, 1.0, 1.0]
    except Exception:
        pass
    if viewer is not None:
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKYBOX] = True
        except Exception:
            pass


# -----------------------------
# Wrist camera rendering (offscreen)
# -----------------------------
def _axis_vec(key: str) -> np.ndarray:
    m = {
        "+x": np.array([1.0, 0.0, 0.0], dtype=float),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+y": np.array([0.0, 1.0, 0.0], dtype=float),
        "-y": np.array([0.0, -1.0, 0.0], dtype=float),
        "+z": np.array([0.0, 0.0, 1.0], dtype=float),
        "-z": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    return m.get(key, m["+z"])


def _azim_elev_from_dir(d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    azim = np.degrees(np.arctan2(d[1], d[0]))
    elev = np.degrees(np.arctan2(d[2], np.sqrt(d[0]*d[0] + d[1]*d[1])))
    return float(azim), float(elev)


def render_wrist_camera_rgb(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    local_forward_axis: str = "+z",  # optical frame looks along +Z
    fovy_deg: float = 60.0,
) -> np.ndarray:
    p = data.xpos[body_id].copy()
    R = data.xmat[body_id].reshape(3, 3).copy()

    fwd_world = R @ _axis_vec(local_forward_axis)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fovy = float(fovy_deg)

    cam.lookat[:] = p + fwd_world * 0.05
    cam.distance = 0.05

    azim, elev = _azim_elev_from_dir(fwd_world)
    cam.azimuth = azim
    cam.elevation = elev

    renderer.update_scene(data, camera=cam)
    return renderer.render()  # RGB uint8


def draw_pip_rgb_to_viewer(viewer, rgb: np.ndarray, x: int, y: int, w: int, h: int):
    """
    Draw an RGB uint8 image into the main viewer as a picture-in-picture overlay.
    Coordinates: x,y = lower-left corner in window pixels.
    """
    # resize
    if rgb.shape[1] != w or rgb.shape[0] != h:
        if cv2 is not None:
            img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        else:
            # crude fallback: nearest via slicing
            ys = np.linspace(0, rgb.shape[0] - 1, h).astype(int)
            xs = np.linspace(0, rgb.shape[1] - 1, w).astype(int)
            img = rgb[ys][:, xs]
    else:
        img = rgb

    # MuJoCo expects bottom-up pixel buffer
    img = np.flipud(img)

    vp = mujoco.MjrRect(int(x), int(y), int(w), int(h))

    win_w, win_h, ctx = _get_viewport_and_ctx(viewer)
    if ctx is None:
        raise RuntimeError("Viewer has no rendering context (viewer.ctx). Tell me your mujoco version.")

    mujoco.mjr_drawPixels(img, None, vp, ctx)

def draw_pip_border(viewer, x: int, y: int, w: int, h: int, thickness: int = 3):
    """
    Draw a red border rectangle around the PiP area.
    Coordinates: x,y lower-left corner in window pixels.
    """
    ctx = getattr(viewer, "ctx", None) or getattr(viewer, "context", None)
    if ctx is None:
        return

    # Many mujoco builds expose mjr_rectangle; use it if available.
    if hasattr(mujoco, "mjr_rectangle"):
        # Draw 4 thin rectangles: bottom, top, left, right
        r, g, b, a = 1.0, 0.0, 0.0, 1.0

        # bottom
        mujoco.mjr_rectangle(mujoco.MjrRect(x, y, w, thickness), r, g, b, a, ctx)
        # top
        mujoco.mjr_rectangle(mujoco.MjrRect(x, y + h - thickness, w, thickness), r, g, b, a, ctx)
        # left
        mujoco.mjr_rectangle(mujoco.MjrRect(x, y, thickness, h), r, g, b, a, ctx)
        # right
        mujoco.mjr_rectangle(mujoco.MjrRect(x + w - thickness, y, thickness, h), r, g, b, a, ctx)
        return

    # Fallback: if mjr_rectangle is missing, do nothing (tell me your mujoco.__version__)
    # and I’ll switch to an OpenGL-line overlay approach.


def show_wrist_window(img_rgb: np.ndarray, title: str = "Wrist Camera", w: int = 480, h: int = 360):
    
    if cv2 is None:
        return

    # resize
    frame = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, frame)
    cv2.waitKey(1)


def find_any_object_id(model: mujoco.MjModel, names: list[str]):
    """
    Try to find an id among BODY, SITE, GEOM.
    Returns (objtype, id, name) or None.
    """
    for nm in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return mujoco.mjtObj.mjOBJ_BODY, int(bid), nm

        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid >= 0:
            return mujoco.mjtObj.mjOBJ_SITE, int(sid), nm

        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid >= 0:
            return mujoco.mjtObj.mjOBJ_GEOM, int(gid), nm

    return None


def get_object_pose(model: mujoco.MjModel, data: mujoco.MjData, objtype, objid: int):
    """
    Returns world position + rotation matrix for BODY/SITE/GEOM.
    """
    if objtype == mujoco.mjtObj.mjOBJ_BODY:
        p = data.xpos[objid].copy()
        R = data.xmat[objid].reshape(3, 3).copy()
        return p, R
    if objtype == mujoco.mjtObj.mjOBJ_SITE:
        p = data.site_xpos[objid].copy()
        R = data.site_xmat[objid].reshape(3, 3).copy()
        return p, R
    if objtype == mujoco.mjtObj.mjOBJ_GEOM:
        p = data.geom_xpos[objid].copy()
        R = data.geom_xmat[objid].reshape(3, 3).copy()
        return p, R
    raise ValueError("Unsupported objtype")

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def make_R_with_axis_k_down(R0: np.ndarray, prefer_x_world=np.array([1.0, 0.0, 0.0])):
    """
    R0: current world-from-body rotation (3x3).
    Finds which body axis is currently most aligned with world +Z (pointing up),
    and constructs a desired world-from-body rotation where that SAME body axis
    points to world -Z (down). Keeps a stable "yaw" by aligning another body axis
    with prefer_x_world as much as possible.
    Returns (R_des, k_up) where k_up is 0/1/2 for body x/y/z.
    """
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    body_axes_world = [R0[:, 0], R0[:, 1], R0[:, 2]]  # body x,y,z in world
    dots = [float(np.dot(a, world_up)) for a in body_axes_world]
    k_up = int(np.argmax(dots))  # which body axis points most "up"

    # We want that same axis to point DOWN in the desired pose
    down = np.array([0.0, 0.0, -1.0], dtype=float)

    # Choose another column to align with prefer_x_world as much as possible
    cols = [None, None, None]
    cols[k_up] = down

    # pick j != k_up
    j = 0 if k_up != 0 else 1

    # project prefer_x_world into plane orthogonal to 'down'
    xproj = prefer_x_world - np.dot(prefer_x_world, down) * down
    xproj = _normalize(xproj)

    cols[j] = xproj

    # remaining column index
    k_rem = [0, 1, 2]
    k_rem.remove(k_up)
    k_rem.remove(j)
    r = k_rem[0]

    # build right-handed frame: col_r = col_k_up x col_j or col_j x col_k_up depending on ordering
    # We need det(R)=+1. We'll try both and pick the one that gives +1.
    cand1 = np.cross(cols[k_up], cols[j])
    cand2 = np.cross(cols[j], cols[k_up])

    # choose candidate that gives positive determinant
    cols1 = cols.copy()
    cols1[r] = _normalize(cand1)
    R1 = np.column_stack(cols1)
    if np.linalg.det(R1) > 0:
        return R1, k_up

    cols2 = cols.copy()
    cols2[r] = _normalize(cand2)
    R2 = np.column_stack(cols2)
    return R2, k_up


def build_wrist_free_camera(
    p_cam: np.ndarray,
    R_cam: np.ndarray,
    fwd_axis: np.ndarray = np.array([0.0, 0.0, -1.0]),
    up_axis:  np.ndarray = np.array([0.0, 1.0,  0.0]),
    look_ahead: float = 0.22,
    down_bias: float = 0.10,
    cam_back: float = 0.10,
    distance: float | None = None,
):
    """
    Version compatible with MuJoCo builds where MjvCamera has NO 'fovy' attribute.

    We keep claws visible by:
    - placing camera slightly behind p_cam along forward axis (cam_back)
    - looking ahead (look_ahead) and slightly downward (down_bias)
    - controlling zoom via cam.distance (distance); if None, computed from geometry
    """
    fwd_world = R_cam @ np.asarray(fwd_axis, dtype=float)
    up_world  = R_cam @ np.asarray(up_axis,  dtype=float)

    fwd_world = _normalize(fwd_world)
    up_world  = _normalize(up_world)

    # camera position (virtual) used only to compute azim/elev + distance
    p = p_cam - fwd_world * float(cam_back)

    # look target: ahead + slight downward bias to keep gripper near bottom
    lookat = p_cam + fwd_world * float(look_ahead) - up_world * float(down_bias)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat

    # Use distance to control framing (acts like zoom)
    if distance is None:
        cam.distance = float(np.linalg.norm(lookat - p))
    else:
        cam.distance = float(distance)

    # azimuth/elevation from direction (lookat - p)
    d = lookat - p
    d = d / (np.linalg.norm(d) + 1e-12)
    cam.azimuth   = float(np.degrees(np.arctan2(d[1], d[0])))
    cam.elevation = float(np.degrees(np.arctan2(d[2], np.sqrt(d[0]*d[0] + d[1]*d[1]))))

    return cam

def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)

def make_free_camera_from_frame_pose(
    p_frame: np.ndarray,
    R_frame: np.ndarray,
    optical_forward: np.ndarray = np.array([0.0, 0.0, 1.0]),  # +Z forward in optical frame (ROS)
    optical_up: np.ndarray      = np.array([0.0, -1.0, 0.0]), # -Y is up in optical frame (since +Y is down)
    look_ahead: float = 0.35,
    cam_back: float = 0.06,
    keep_claws_down_bias: float = 0.03,
    distance: float | None = None,
):
    """
    Create an mjCAMERA_FREE that matches a URDF optical frame pose.

    - Uses p_frame, R_frame directly (no arbitrary aiming).
    - Optical frame: x right, y down, z forward => forward = +Z, "up" = -Y.
    - keep_claws_down_bias: small shift in the image 'down' direction to keep claws at bottom.
      (physically: camera is above gripper, so claws appear lower; this just ensures it.)
    - No cam.fovy used (works on builds where MjvCamera has no fovy attr).
    """
    fwd = _norm(R_frame @ np.asarray(optical_forward, dtype=float))  # world forward
    up  = _norm(R_frame @ np.asarray(optical_up, dtype=float))       # world up
    right = _norm(np.cross(fwd, up))

    # camera position used for azim/elev inference
    p_cam = p_frame - fwd * float(cam_back)

    # "image down" direction in world is -up (because up points up)
    img_down = -up

    lookat = p_frame + fwd * float(look_ahead) + img_down * float(keep_claws_down_bias)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat

    if distance is None:
        cam.distance = float(np.linalg.norm(lookat - p_cam))
    else:
        cam.distance = float(distance)

    d = _norm(lookat - p_cam)
    cam.azimuth   = float(np.degrees(np.arctan2(d[1], d[0])))
    cam.elevation = float(np.degrees(np.arctan2(d[2], np.sqrt(d[0]*d[0] + d[1]*d[1]))))

    return cam

import numpy as np
import mujoco

def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)

def build_camera_from_pos_and_gripper_dir(
    p_cam: np.ndarray,
    p_ee: np.ndarray,
    R_ee: np.ndarray,
    ee_forward_axis: np.ndarray = np.array([0.0, 0.0, -1.0]),  # which EE axis is "pointing"
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    look_ahead: float = 0.35,
    cam_back: float = 0.06,
    claws_down_bias: float = 0.04,
    distance: float | None = None,
):
    """
    Camera location = p_cam (from URDF camera frame).
    Camera direction = gripper direction (from EE frame axis).
    Ensures "up" is stable using world_up.
    Adds a small 'claws_down_bias' so fingers appear in lower part of image.

    Works on MuJoCo builds where MjvCamera has no 'fovy'.
    """
    # Forward direction from EE orientation
    fwd = _norm(R_ee @ np.asarray(ee_forward_axis, dtype=float))

    # Construct a consistent camera up vector (avoid roll)
    up = world_up - np.dot(world_up, fwd) * fwd
    up = _norm(up)
    right = _norm(np.cross(fwd, up))
    up = _norm(np.cross(right, fwd))  # re-orthonormalize

    # camera position used to compute azim/elev
    p_virtual = np.asarray(p_cam, dtype=float) - fwd * float(cam_back)

    # "image down" is -up
    img_down = -up

    # Look at where gripper points + small downward bias to include claws
    lookat = np.asarray(p_cam, dtype=float) + fwd * float(look_ahead) + img_down * float(claws_down_bias)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat

    if distance is None:
        cam.distance = float(np.linalg.norm(lookat - p_virtual))
    else:
        cam.distance = float(distance)

    d = _norm(lookat - p_virtual)
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    cam.elevation = float(np.degrees(np.arctan2(d[2], np.sqrt(d[0]*d[0] + d[1]*d[1]))))
    return cam

def geom_local_min_z(model: mujoco.MjModel, gid: int) -> float:
    """
    Return local-frame min z of this geom (before xmat/xpos transform).
    Uses analytic bounds for primitive geoms; falls back to -rbound for mesh.
    """
    gtype = int(model.geom_type[gid])
    sz = model.geom_size[gid]  # (3,) always

    # MuJoCo geom types: sphere=2, capsule=3, cylinder=4, box=5, mesh=6 (common)
    # But safest is to handle box/cyl/sphere/capsule/ellipsoid; mesh fallback.
    # We'll use numeric constants from mujoco.mjtGeom if available.
    try:
        sphere   = mujoco.mjtGeom.mjGEOM_SPHERE
        capsule  = mujoco.mjtGeom.mjGEOM_CAPSULE
        cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
        box      = mujoco.mjtGeom.mjGEOM_BOX
        ellip    = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        mesh     = mujoco.mjtGeom.mjGEOM_MESH
    except Exception:
        sphere, capsule, cylinder, box, ellip, mesh = 2, 3, 4, 5, 7, 6  # fallback guesses

    if gtype == sphere:
        return -float(sz[0])
    if gtype == capsule:
        # radius=sz[0], half-length=sz[1] along local z
        return -(float(sz[0]) + float(sz[1]))
    if gtype == cylinder:
        # radius=sz[0], half-length=sz[1] along local z
        return -float(sz[1])
    if gtype == box:
        # half-sizes: sz[0],sz[1],sz[2]
        return -float(sz[2])
    if gtype == ellip:
        return -float(sz[2])
    if gtype == mesh:
        # we don't have mesh AABB here -> use rbound as fallback
        return -float(model.geom_rbound[gid])

    # unknown -> safe fallback
    return -float(model.geom_rbound[gid])


def geom_world_min_z(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> float:
    """
    Compute the *actual* lowest world Z of a geom.
    - For primitives: analytic bottom in geom local frame, then transform to world.
    - For meshes: transform mesh vertices to world and take min(z).
      Handles both MuJoCo Python binding conventions:
        A) mesh_vertadr/vertnum are in VERTEX units
        B) mesh_vertadr/vertnum are in FLOAT units (xyzxyz...)
      Also handles model.mesh_vert being 1D or 2D.
    """
    import numpy as np
    import mujoco

    gtype = int(model.geom_type[gid])

    # --- mjtGeom constants ---
    GEOM_SPHERE   = mujoco.mjtGeom.mjGEOM_SPHERE
    GEOM_CAPSULE  = mujoco.mjtGeom.mjGEOM_CAPSULE
    GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
    GEOM_BOX      = mujoco.mjtGeom.mjGEOM_BOX
    GEOM_ELLIP    = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    GEOM_MESH     = mujoco.mjtGeom.mjGEOM_MESH

    # -------------------------
    # Mesh: transform vertices
    # -------------------------
    if gtype == GEOM_MESH:
        mid = int(model.geom_dataid[gid])  # mesh id
        if mid >= 0:
            adr = int(model.mesh_vertadr[mid])
            num = int(model.mesh_vertnum[mid])

            mv = model.mesh_vert

            # Case 1: mv is (N,3)
            if hasattr(mv, "ndim") and mv.ndim == 2:
                v = mv[adr: adr + num, :].astype(np.float64)

            else:
                # mv is flat 1D: [x0,y0,z0,x1,y1,z1,...]
                mv = np.asarray(mv)

                # Try interpretation A: adr/num are vertex counts
                startA = 3 * adr
                endA   = 3 * (adr + num)
                okA = (0 <= startA < mv.size) and (0 < endA <= mv.size) and ((endA - startA) == 3 * num)

                if okA:
                    flat = mv[startA:endA]
                    v = flat.reshape(num, 3).astype(np.float64)
                else:
                    # Interpretation B: adr/num are float counts
                    startB = adr
                    endB   = adr + num
                    if not (0 <= startB < mv.size and 0 < endB <= mv.size):
                        # fallback (shouldn't happen)
                        cz = float(data.geom_xpos[gid][2])
                        return cz - float(model.geom_rbound[gid])

                    flat = mv[startB:endB]
                    if flat.size % 3 != 0:
                        # fallback (corrupt / unexpected)
                        cz = float(data.geom_xpos[gid][2])
                        return cz - float(model.geom_rbound[gid])

                    v = flat.reshape(flat.size // 3, 3).astype(np.float64)

            # Apply geom mesh scale
            scale = np.array(model.geom_size[gid], dtype=np.float64)
            v = v * scale

            R = data.geom_xmat[gid].reshape(3, 3)
            p = data.geom_xpos[gid]

            vw = (v @ R.T) + p
            return float(vw[:, 2].min())

        # mesh id missing → fallback
        cz = float(data.geom_xpos[gid][2])
        return cz - float(model.geom_rbound[gid])

    # -------------------------
    # Primitives: local-bottom
    # -------------------------
    sz = np.array(model.geom_size[gid], dtype=np.float64)

    if gtype == GEOM_SPHERE:
        local = np.array([0.0, 0.0, -float(sz[0])])
    elif gtype == GEOM_CAPSULE:
        local = np.array([0.0, 0.0, -(float(sz[0]) + float(sz[1]))])
    elif gtype == GEOM_CYLINDER:
        local = np.array([0.0, 0.0, -float(sz[1])])
    elif gtype == GEOM_BOX:
        local = np.array([0.0, 0.0, -float(sz[2])])
    elif gtype == GEOM_ELLIP:
        local = np.array([0.0, 0.0, -float(sz[2])])
    else:
        cz = float(data.geom_xpos[gid][2])
        return cz - float(model.geom_rbound[gid])

    R = data.geom_xmat[gid].reshape(3, 3)
    p = data.geom_xpos[gid]
    world = p + R @ local
    return float(world[2])

import numpy as np
import mujoco

import numpy as np
import mujoco

def draw_body_frame_in_viewer(viewer, data, body_id: int, length=0.10, radius=0.004):
    """
    Draw XYZ axes of a MuJoCo body frame as 3 capsules in the viewer overlay.
    X=red, Y=green, Z=blue.
    """
    scn = viewer.user_scn
    scn.ngeom = 0  # clear overlay geoms each frame

    p = np.array(data.xpos[body_id], dtype=np.float64)
    R = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)

    px = p + R[:, 0] * float(length)
    py = p + R[:, 1] * float(length)
    pz = p + R[:, 2] * float(length)

    def _add_capsule(p0, p1, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).ravel(),
            np.array(rgba, dtype=np.float32),
        )

        # Your mujoco build expects: (geom, type, width, from_, to)
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(p0, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
        )

        scn.ngeom += 1

    _add_capsule(p, px, [1, 0, 0, 1])  # X
    _add_capsule(p, py, [0, 1, 0, 1])  # Y
    _add_capsule(p, pz, [0, 0, 1, 1])  # Z

def show_wrist_window2(frame, title="Wrist Camera", w=480, h=480, *, fallback_save_path=None):
    """
    Cross-platform safe OpenCV imshow.
    - Sanitizes dtype/shape
    - Ensures contiguous memory
    - Pumps events via waitKey(1)
    - NEVER raises: on failure prints warning and optionally saves a PNG
    Returns: True if displayed, False otherwise
    """
    try:
        import numpy as np
        import cv2
    except Exception:
        return False

    try:
        # ---- Validate ----
        if frame is None:
            print("[WARN] show_wrist_window: frame is None")
            return False
        if not hasattr(frame, "shape"):
            print(f"[WARN] show_wrist_window: frame type={type(frame)} (not ndarray-like)")
            return False
        if frame.size == 0:
            print("[WARN] show_wrist_window: empty frame")
            return False

        img = frame

        # ---- Convert to uint8 BGR ----
        if img.dtype != np.uint8:
            # common: float32 in [0,1] from render()
            if np.issubdtype(img.dtype, np.floating):
                mx = float(np.nanmax(img)) if img.size else 0.0
                if mx <= 1.5:
                    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # MuJoCo renderer typically returns RGB; OpenCV expects BGR
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]  # RGB->BGR
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            print(f"[WARN] show_wrist_window: unexpected shape {img.shape}")
            return False

        # ---- Resize + contiguous ----
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img)

        # ---- Show + pump events ----
        cv2.imshow(title, img)
        cv2.waitKey(1)  # important on macOS

        return True

    except cv2.error as e:
        print(f"[WARN] OpenCV imshow failed ({title}): {e}")
    except Exception as e:
        print(f"[WARN] show_wrist_window unexpected error ({title}): {e}")

    # Optional fallback: save image for debugging
    if fallback_save_path is not None:
        try:
            import cv2
            cv2.imwrite(fallback_save_path, img)
            print(f"[INFO] Saved wrist image to {fallback_save_path}")
        except Exception:
            pass

    return False

def viewer_use_fixed_camera(viewer, cam_id: int):
    # Render the viewer from a named MJCF camera (your wrist_rgb)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = int(cam_id)

def viewer_use_free_camera(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

