#!/usr/bin/env python3
"""Generate an interactive 3D HTML preview with embedded 2D projections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.slab_setup import create_slab_system
from common.workflow_support import (
    active_workflow_uuid,
    ensure_output_directories,
    get_workflow_logger,
    load_config,
    preview_html_path,
    slab_info_path,
)


def _downsample(positions: np.ndarray, max_points: int = 50_000) -> np.ndarray:
    if len(positions) <= max_points:
        return positions
    indices = np.random.choice(len(positions), max_points, replace=False)
    return positions[indices]


def _build_threejs_html(
    crystal: np.ndarray,
    fluid: np.ndarray,
    box: np.ndarray,
    slab_info: dict,
    max_points: int = 50_000,
) -> str:
    """Return a self-contained HTML string with an interactive Three.js scene
    and embedded x-z and y-z canvas projections."""

    crystal_sample = _downsample(crystal, max_points)
    fluid_sample = _downsample(fluid, max_points)

    # Centre of the box; the scene is rendered around the origin.
    cx, cy, cz = box / 2.0
    data = {
        "crystal": crystal_sample.tolist(),
        "fluid": fluid_sample.tolist(),
        "box": box.tolist(),
        "centre": [cx, cy, cz],
        "info": slab_info,
    }
    data_json = json.dumps(data)

    Lx, Ly, Lz = box
    z_slab = float(slab_info["z_slab"])
    z_c = Lz / 2.0
    half_slab = z_slab / 2.0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crystal Slab Preview</title>
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      font-family: sans-serif;
      background: #111;
      color: #eee;
    }}
    #info {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 10;
      background: rgba(0,0,0,0.7);
      padding: 12px;
      border-radius: 6px;
      max-width: 320px;
      font-size: 13px;
      line-height: 1.4;
    }}
    #info h1 {{ margin: 0 0 8px; font-size: 16px; }}
    #info p {{ margin: 4px 0; }}
    #controls {{
      position: absolute;
      bottom: 10px;
      left: 10px;
      z-index: 10;
      background: rgba(0,0,0,0.7);
      padding: 10px;
      border-radius: 6px;
    }}
    button {{ margin: 4px; padding: 6px 12px; cursor: pointer; }}
    #canvas-container {{
      position: absolute;
      top: 0;
      left: 0;
      width: 70vw;
      height: 100vh;
    }}
    #projections {{
      position: absolute;
      top: 0;
      right: 0;
      width: 30vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
      background: #1a1a1a;
    }}
    .projection {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      border-top: 1px solid #333;
    }}
    .projection:first-child {{ border-top: none; }}
    .projection h3 {{
      margin: 8px 0 4px;
      font-size: 14px;
      color: #ccc;
    }}
    .projection canvas {{
      max-width: 95%;
      max-height: 85%;
      border: 1px solid #444;
      background: #0d0d0d;
    }}
  </style>
</head>
<body>
  <div id="info">
    <h1>Crystal Slab Geometry</h1>
    <p><strong>Crystal:</strong> {slab_info.get("crystal", "?")} {slab_info.get("hkl", "?")}</p>
    <p><strong>Box:</strong> Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}</p>
    <p><strong>Slab:</strong> N_cryst={slab_info.get("N_crystal", "?")}, z_slab={z_slab:.3f}</p>
    <p><strong>Fluid:</strong> N_fluid={slab_info.get("n_fluid", "?")}, vf_final={slab_info.get("vf_final", "?")}</p>
    <p><strong>Aspect:</strong> z/Lx={slab_info.get("z_aspect_actual", "?"):.3f}</p>
    <p><strong>Downsampled to</strong> {len(fluid_sample)} fluid / {len(crystal_sample)} crystal points for display.</p>
  </div>
  <div id="controls">
    <button onclick="toggleCrystal()">Toggle crystal</button>
    <button onclick="toggleFluid()">Toggle fluid</button>
    <button onclick="toggleBox()">Toggle box</button>
    <button onclick="resetCamera()">Reset camera</button>
  </div>
  <div id="canvas-container"></div>
  <div id="projections">
    <div class="projection">
      <h3>x-z projection</h3>
      <canvas id="xz-canvas"></canvas>
    </div>
    <div class="projection">
      <h3>y-z projection</h3>
      <canvas id="yz-canvas"></canvas>
    </div>
  </div>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const data = {data_json};
    const box = new THREE.Vector3(...data.box);
    const centre = new THREE.Vector3(...data.centre);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const container = document.getElementById('canvas-container');
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(box.x * 1.5, box.y * 1.5, box.z * 1.2);
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.update();

    // Lights
    const ambient = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 1.5);
    directional.position.set(box.x, box.y, box.z);
    scene.add(directional);

    // Box wireframe, centered at the origin so it spans [-box/2, +box/2].
    const boxGeometry = new THREE.BoxGeometry(box.x, box.y, box.z);
    const boxEdges = new THREE.EdgesGeometry(boxGeometry);
    const boxMaterial = new THREE.LineBasicMaterial({{ color: 0x888888 }});
    const boxLines = new THREE.LineSegments(boxEdges, boxMaterial);
    boxLines.position.set(0, 0, 0);
    scene.add(boxLines);

    function makePointCloud(positions, color, size) {{
      const geometry = new THREE.BufferGeometry();
      const vertices = new Float32Array(positions.length * 3);
      for (let i = 0; i < positions.length; i++) {{
        vertices[i * 3 + 0] = positions[i][0] - centre.x;
        vertices[i * 3 + 1] = positions[i][1] - centre.y;
        vertices[i * 3 + 2] = positions[i][2] - centre.z;
      }}
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      const material = new THREE.PointsMaterial({{ color: color, size: size }});
      return new THREE.Points(geometry, material);
    }}

    const crystalCloud = makePointCloud(data.crystal, 0xff5555, 0.12);
    const fluidCloud = makePointCloud(data.fluid, 0x55aaff, 0.08);
    scene.add(crystalCloud);
    scene.add(fluidCloud);

    window.toggleCrystal = () => {{ crystalCloud.visible = !crystalCloud.visible; }};
    window.toggleFluid = () => {{ fluidCloud.visible = !fluidCloud.visible; }};
    window.toggleBox = () => {{ boxLines.visible = !boxLines.visible; }};
    window.resetCamera = () => {{
      camera.position.set(box.x * 1.5, box.y * 1.5, box.z * 1.2);
      controls.target.set(0, 0, 0);
      controls.update();
    }};

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});

    // 2D projections
    function drawProjection(canvasId, lateral, vertical, lateralLabel, verticalLabel) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const pad = 40;
      const w = canvas.parentElement.clientWidth - 2 * pad;
      const h = canvas.parentElement.clientHeight - 60;
      canvas.width = w + 2 * pad;
      canvas.height = h + 2 * pad;

      ctx.fillStyle = '#0d0d0d';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scaleX = w / lateral;
      const scaleY = h / vertical;

      // Box outline
      ctx.strokeStyle = '#cccccc';
      ctx.lineWidth = 2;
      ctx.strokeRect(pad, pad, w, h);

      // Slab (centred in z)
      const zSlab = {z_slab};
      const zC = vertical / 2.0;
      const slabTop = pad + (1.0 - (zC + zSlab / 2.0) / vertical) * h;
      const slabH = (zSlab / vertical) * h;
      ctx.fillStyle = 'rgba(255, 85, 85, 0.5)';
      ctx.fillRect(pad, slabTop, w, slabH);

      // Fluid region outline (optional)
      ctx.strokeStyle = 'rgba(85, 170, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad + w, pad);
      ctx.lineTo(pad + w, slabTop);
      ctx.moveTo(pad + w, slabTop + slabH);
      ctx.lineTo(pad + w, pad + h);
      ctx.lineTo(pad, pad + h);
      ctx.lineTo(pad, slabTop + slabH);
      ctx.moveTo(pad, slabTop);
      ctx.lineTo(pad, pad);
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#eeeeee';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(lateralLabel, pad + w / 2, pad + h + 25);
      ctx.save();
      ctx.translate(15, pad + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(verticalLabel, 0, 0);
      ctx.restore();

      // Dimensions
      ctx.textAlign = 'left';
      ctx.fillText(lateral.toFixed(2), pad + w - 40, pad + h + 25);
      ctx.fillText(vertical.toFixed(2), pad + 5, pad + 12);
    }}

    drawProjection('xz-canvas', {Lx}, {Lz}, 'X / σ', 'Z / σ');
    drawProjection('yz-canvas', {Ly}, {Lz}, 'Y / σ', 'Z / σ');
  </script>
</body>
</html>
"""


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("preview", current_uuid)

    n_particles = int(conf["System"]["n"])
    sigma = 1
    vf_init = float(conf["System"]["vf_init"])
    vf_final = float(conf["System"]["vf_final"])
    r_skin = float(conf["System"]["r_skin"])
    vf_crystal = float(conf["Boundary"]["vf_crystal"])
    slab_thickness = float(conf["Boundary"]["slab_thickness"])
    z_aspect = float(conf["Boundary"]["z_aspect"])
    min_gap = float(conf["Boundary"]["min_gap"])
    boundary_thickness = float(conf["Boundary"]["boundary_thickness"])
    crystal = conf["Boundary"]["crystal"].strip().lower()
    hkl_raw = conf["Boundary"]["hkl"].split()
    hkl = tuple(int(x) for x in hkl_raw)

    logger.info("Building slab geometry for preview")

    ensure_output_directories()

    state = create_slab_system(
        n_particles,
        sigma,
        vf_init,
        vf_final,
        r_skin,
        vf_crystal,
        slab_thickness,
        z_aspect,
        min_gap,
        boundary_thickness,
        crystal,
        hkl,
    )
    system = state["system"]
    slab = state["slab"]
    indices_to_move = state["indices_to_move"]
    positions = system.copy_positions().T
    fluid = positions[indices_to_move]
    box = np.array(system.get_box())

    slab_info = dict(state["slab_info"])
    slab_info["crystal"] = crystal
    slab_info["hkl"] = list(hkl)
    slab_info["n_fluid"] = n_particles
    slab_info["vf_final"] = vf_final

    # Persist slab_info so other stages can read it without recomputing geometry.
    with open(slab_info_path(current_uuid), "w") as f:
        json.dump(slab_info, f, indent=2)

    html_path = preview_html_path(current_uuid)
    logger.info("Writing interactive 3D preview: %s", html_path.name)
    html_path.write_text(_build_threejs_html(slab, fluid, box, slab_info))

    logger.info("Preview complete")


if __name__ == "__main__":
    main()
