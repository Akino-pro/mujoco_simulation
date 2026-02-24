import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "trimesh is required. Install with: pip install trimesh vhacdx"
    ) from exc


@dataclass(frozen=True)
class TargetSpec:
    body_name: str
    mesh_prefix: str
    geom_prefix: str
    density: float
    max_parts: int
    rgba: str
    out_folder: str
    legacy_geom_names: Tuple[str, ...] = ()
    source_mesh: Optional[str] = None
    source_asset_mesh: Optional[str] = None
    scale: Optional[Tuple[float, float, float]] = None
    geom_pos: Optional[str] = None
    geom_quat: Optional[str] = None


TARGETS = [
    # Objects
    TargetSpec(
        body_name="big_bear",
        source_mesh="objects/teddy_bear/teddy3.obj",
        scale=(0.01, 0.01, 0.01),
        density=10.0,
        mesh_prefix="big_bear_col_mesh_part_",
        geom_prefix="big_bear_col_part_",
        max_parts=16,
        legacy_geom_names=("big_bear_col", "big_bear_col_viz"),
        rgba="1 0.2 0.2 0.18",
        out_folder="objects/_convex_parts",
    ),
    TargetSpec(
        body_name="small_bear",
        source_mesh="objects/teddy_bear/teddy3.obj",
        scale=(0.006, 0.006, 0.006),
        density=10.0,
        mesh_prefix="small_bear_col_mesh_part_",
        geom_prefix="small_bear_col_part_",
        max_parts=12,
        legacy_geom_names=("small_bear_col", "small_bear_col_viz"),
        rgba="0.2 0.6 1 0.18",
        out_folder="objects/_convex_parts",
    ),
    TargetSpec(
        body_name="cardboard_box",
        source_mesh="objects/Cardboard box/Cardboard box.obj",
        scale=(0.15, 0.15, 0.15),
        density=5.0,
        mesh_prefix="cardboard_col_mesh_part_",
        geom_prefix="cardboard_col_part_",
        max_parts=12,
        legacy_geom_names=("cardboard_col", "cardboard_col_viz"),
        rgba="0.2 1 0.4 0.18",
        out_folder="objects/_convex_parts",
    ),
    # Robot arm and wrist links
    TargetSpec(
        body_name="gen3_mount",
        source_asset_mesh="base_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="base_col_mesh_part_",
        geom_prefix="base_col_part_",
        max_parts=4,
        legacy_geom_names=("base_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_shoulder_link",
        source_asset_mesh="shoulder_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="shoulder_col_mesh_part_",
        geom_prefix="shoulder_col_part_",
        max_parts=6,
        legacy_geom_names=("shoulder_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_half_arm_1_link",
        source_asset_mesh="half_arm_1_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="upperarm_col_mesh_part_",
        geom_prefix="upperarm_col_part_",
        max_parts=8,
        legacy_geom_names=("upperarm_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_half_arm_2_link",
        source_asset_mesh="half_arm_2_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="upperarm2_col_mesh_part_",
        geom_prefix="upperarm2_col_part_",
        max_parts=8,
        legacy_geom_names=("upperarm2_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_forearm_link",
        source_asset_mesh="forearm_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="forearm_col_mesh_part_",
        geom_prefix="forearm_col_part_",
        max_parts=8,
        legacy_geom_names=("forearm_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_spherical_wrist_1_link",
        source_asset_mesh="spherical_wrist_1_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="wrist1_col_mesh_part_",
        geom_prefix="wrist1_col_part_",
        max_parts=6,
        legacy_geom_names=("wrist1_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_spherical_wrist_2_link",
        source_asset_mesh="spherical_wrist_2_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="wrist2_col_mesh_part_",
        geom_prefix="wrist2_col_part_",
        max_parts=6,
        legacy_geom_names=("wrist2_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_bracelet_link",
        source_asset_mesh="bracelet_no_vision_link",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="bracelet_col_mesh_part_",
        geom_prefix="bracelet_col_part_",
        max_parts=6,
        legacy_geom_names=("bracelet_col",),
        rgba="1 0.9 0.2 0.12",
        out_folder="robot/_convex_parts",
    ),
    TargetSpec(
        body_name="gen3_bracelet_link",
        source_asset_mesh="robotiq_base",
        scale=(1.0, 1.0, 1.0),
        density=1000.0,
        mesh_prefix="robotiq_base_col_mesh_part_",
        geom_prefix="robotiq_base_col_part_",
        max_parts=6,
        legacy_geom_names=("robotiq_base_col",),
        rgba="1 0.8 0.2 0.12",
        out_folder="robot/_convex_parts",
        geom_pos="0 0 -0.061525",
        geom_quat="0 1 0 0",
    ),
]


def _indent_xml(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _as_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = []
        for g in mesh_or_scene.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)
        if not meshes:
            raise RuntimeError("Scene contains no mesh geometry.")
        return trimesh.util.concatenate(meshes)
    raise RuntimeError(f"Unsupported mesh type: {type(mesh_or_scene)!r}")


def _decompose(mesh: trimesh.Trimesh, max_parts: int):
    try:
        parts = trimesh.decomposition.convex_decomposition(
            mesh,
            maxConvexHulls=max_parts,
            resolution=500_000,
            minimumVolumePercentErrorAllowed=0.01,
            maxRecursionDepth=15,
            maxNumVerticesPerCH=64,
            findBestPlane=True,
        )
    except Exception as exc:
        print(
            f"[WARN] Convex decomposition failed ({exc}). Falling back to one convex hull.",
            file=sys.stderr,
        )
        return [mesh.convex_hull]

    if isinstance(parts, trimesh.Trimesh):
        return [parts]
    if isinstance(parts, list):
        out = []
        for p in parts:
            if isinstance(p, trimesh.Trimesh):
                out.append(p)
                continue
            if isinstance(p, dict) and "vertices" in p and "faces" in p:
                out.append(
                    trimesh.Trimesh(
                        vertices=np.asarray(p["vertices"], dtype=float),
                        faces=np.asarray(p["faces"], dtype=np.int64),
                        process=False,
                    )
                )
        return out if out else [mesh.convex_hull]
    return [mesh.convex_hull]


def _find_body(root: ET.Element, name: str):
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    return None


def _find_asset_mesh(asset: ET.Element, mesh_name: str):
    for mesh in asset.findall("mesh"):
        if mesh.get("name") == mesh_name:
            return mesh
    return None


def _remove_prefixed(parent: ET.Element, tag: str, prefixes: List[str]):
    to_delete = []
    for child in parent.findall(tag):
        name = child.get("name", "")
        if any(name.startswith(p) for p in prefixes):
            to_delete.append(child)
    for child in to_delete:
        parent.remove(child)


def _remove_named(parent: ET.Element, tag: str, names: Tuple[str, ...]):
    if not names:
        return
    targets = set(names)
    to_delete = []
    for child in parent.findall(tag):
        if child.get("name", "") in targets:
            to_delete.append(child)
    for child in to_delete:
        parent.remove(child)


def _write_parts(mesh_parts: List[trimesh.Trimesh], out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    for i, part in enumerate(mesh_parts):
        part_path = out_dir / f"{stem}_{i:02d}.stl"
        part.export(part_path)
        part_paths.append(part_path)
    return part_paths


def _parse_scale(scale_text: Optional[str], fallback: Tuple[float, float, float]):
    if not scale_text:
        return np.asarray(fallback, dtype=float)
    vals = [float(v) for v in scale_text.split()]
    if len(vals) != 3:
        raise ValueError(f"Invalid mesh scale '{scale_text}'")
    return np.asarray(vals, dtype=float)


def _resolve_source(base_xml: Path, asset: ET.Element, spec: TargetSpec):
    if spec.source_mesh:
        src = (base_xml.parent / spec.source_mesh).resolve()
        if spec.scale is None:
            raise RuntimeError(f"Explicit source_mesh requires scale for {spec.body_name}")
        scale = np.asarray(spec.scale, dtype=float)
        return src, scale

    if spec.source_asset_mesh:
        mesh_elem = _find_asset_mesh(asset, spec.source_asset_mesh)
        if mesh_elem is None:
            raise RuntimeError(f"Asset mesh '{spec.source_asset_mesh}' not found.")
        file_attr = mesh_elem.get("file")
        if not file_attr:
            raise RuntimeError(f"Asset mesh '{spec.source_asset_mesh}' has no file attribute.")
        src = (base_xml.parent / file_attr).resolve()
        fallback_scale = spec.scale if spec.scale is not None else (1.0, 1.0, 1.0)
        scale = _parse_scale(mesh_elem.get("scale"), fallback_scale)
        return src, scale

    raise RuntimeError(f"No source mesh configured for body '{spec.body_name}'")


def build_convex_scene(base_xml: Path, out_xml: Path, debug_viz: bool):
    tree = ET.parse(base_xml)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("No <asset> found in base MJCF.")

    _remove_prefixed(asset, "mesh", [t.mesh_prefix for t in TARGETS])
    for t in TARGETS:
        body = _find_body(root, t.body_name)
        if body is None:
            raise RuntimeError(f"Body '{t.body_name}' not found.")
        _remove_prefixed(body, "geom", [t.geom_prefix])
        _remove_named(body, "geom", t.legacy_geom_names)

    for t in TARGETS:
        src, scale = _resolve_source(base_xml, asset, t)
        if not src.exists():
            raise FileNotFoundError(f"Mesh not found: {src}")

        raw = trimesh.load(src, force="scene")
        mesh = _as_mesh(raw).copy()
        mesh.apply_scale(scale)
        parts = _decompose(mesh, max_parts=t.max_parts)

        out_dir = (base_xml.parent / t.out_folder / t.body_name).resolve()
        part_files = _write_parts(parts, out_dir, stem=t.body_name)

        body = _find_body(root, t.body_name)
        assert body is not None

        for i, part_file in enumerate(part_files):
            mesh_name = f"{t.mesh_prefix}{i:02d}"
            geom_name = f"{t.geom_prefix}{i:02d}"
            rel_path = part_file.relative_to(base_xml.parent).as_posix()

            asset.append(
                ET.Element(
                    "mesh",
                    attrib={
                        "name": mesh_name,
                        "file": rel_path,
                    },
                )
            )

            geom_attrib = {
                "name": geom_name,
                "type": "mesh",
                "mesh": mesh_name,
                "contype": "1",
                "conaffinity": "1",
                "density": str(t.density),
                "rgba": t.rgba,
            }
            if t.geom_pos:
                geom_attrib["pos"] = t.geom_pos
            if t.geom_quat:
                geom_attrib["quat"] = t.geom_quat

            body.append(ET.Element("geom", attrib=geom_attrib))
            if debug_viz:
                body.append(
                    ET.Element(
                        "geom",
                        attrib={
                            "name": f"{geom_name}_viz",
                            "type": "mesh",
                            "mesh": mesh_name,
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "1",
                            "rgba": "0.2 1 1 0.12",
                        },
                    )
                )

        print(f"[INFO] {t.body_name}: wrote {len(part_files)} convex parts")

    _indent_xml(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote convex MJCF: {out_xml}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate robotsuit_convex.xml with convex collision parts from visual meshes."
    )
    parser.add_argument(
        "--base",
        default="robotsuit_cubes.xml",
        help="Base MJCF path (default: robotsuit_cubes.xml)",
    )
    parser.add_argument(
        "--out",
        default="robotsuit_convex.xml",
        help="Output MJCF path (default: robotsuit_convex.xml)",
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Add extra non-colliding visualization geoms.",
    )
    args = parser.parse_args()

    base_xml = Path(args.base).resolve()
    out_xml = Path(args.out).resolve()
    if not base_xml.exists():
        raise FileNotFoundError(f"Base MJCF not found: {base_xml}")

    build_convex_scene(base_xml=base_xml, out_xml=out_xml, debug_viz=args.debug_viz)


if __name__ == "__main__":
    main()
