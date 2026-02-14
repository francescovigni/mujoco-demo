/**
 * Three.js 3D Scene for Rover Navigation
 * 
 * Renders the simulation client-side for smooth 60fps performance.
 */

import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Convert MuJoCo quaternion [w, x, y, z] to Three.js Quaternion
function mujocoQuatToThree(q) {
  // MuJoCo: [w, x, y, z], Three.js: (x, y, z, w)
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

export default function Scene3D({
  sceneData,       // { obstacles, target, groundSize }
  roverPosition,   // [x, y, z]
  roverQuaternion, // [w, x, y, z]
  targetPosition,  // [x, y]
  path,            // [[x, y], [x, y], ...]
  isRunning,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const roverRef = useRef(null);
  const targetRef = useRef(null);
  const pathLineRef = useRef(null);
  const frameIdRef = useRef(null);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || !sceneData) return;
    
    // Capture container ref for cleanup
    const container = containerRef.current;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    scene.fog = new THREE.Fog(0x1a1a2e, 30, 80);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      200
    );
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.1;
    controls.minDistance = 2;
    controls.maxDistance = 50;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(10, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 60;
    dirLight.shadow.camera.left = -30;
    dirLight.shadow.camera.right = 30;
    dirLight.shadow.camera.top = 30;
    dirLight.shadow.camera.bottom = -30;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
    fillLight.position.set(-10, 10, -10);
    scene.add(fillLight);

    // Ground plane
    const groundSize = sceneData.groundSize || 40;
    const groundGeom = new THREE.PlaneGeometry(groundSize * 2, groundSize * 2);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x2a3a4a,
      roughness: 0.8,
      metalness: 0.2,
    });
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Grid
    const gridHelper = new THREE.GridHelper(groundSize * 2, 40, 0x444466, 0x333355);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    // Obstacles
    sceneData.obstacles.forEach((obs) => {
      const geom = new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 16);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(obs.color[0], obs.color[1], obs.color[2]),
        roughness: 0.6,
        metalness: 0.3,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(obs.position[0], obs.height / 2, obs.position[1]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);
    });

    // Rover (simplified box model matching MuJoCo dimensions)
    const roverGroup = new THREE.Group();
    
    // Chassis
    const chassisGeom = new THREE.BoxGeometry(0.56, 0.08, 0.36);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x1a3a6a,
      roughness: 0.4,
      metalness: 0.6,
    });
    const chassis = new THREE.Mesh(chassisGeom, chassisMat);
    chassis.position.y = 0.04;
    chassis.castShadow = true;
    roverGroup.add(chassis);

    // Top plate
    const topGeom = new THREE.BoxGeometry(0.40, 0.03, 0.28);
    const top = new THREE.Mesh(topGeom, chassisMat);
    top.position.y = 0.095;
    top.castShadow = true;
    roverGroup.add(top);

    // Sensor mast
    const mastGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.16, 8);
    const hubMat = new THREE.MeshStandardMaterial({ color: 0x909099, metalness: 0.8 });
    const mast = new THREE.Mesh(mastGeom, hubMat);
    mast.position.set(0.08, 0.18, 0);
    roverGroup.add(mast);

    // LIDAR dome
    const lidarGeom = new THREE.SphereGeometry(0.04, 16, 12);
    const accentMat = new THREE.MeshStandardMaterial({ color: 0xdd4422, metalness: 0.5 });
    const lidar = new THREE.Mesh(lidarGeom, accentMat);
    lidar.position.set(0.08, 0.28, 0);
    roverGroup.add(lidar);

    // Wheels
    const wheelGeom = new THREE.CylinderGeometry(0.065, 0.065, 0.056, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.9 });
    const wheelPositions = [
      [0.19, 0, 0.22],   // FL
      [0.19, 0, -0.22],  // FR
      [-0.19, 0, 0.22],  // RL
      [-0.19, 0, -0.22], // RR
    ];
    wheelPositions.forEach(([x, y, z]) => {
      const wheel = new THREE.Mesh(wheelGeom, wheelMat);
      wheel.position.set(x, y, z);
      wheel.rotation.x = Math.PI / 2;
      wheel.castShadow = true;
      roverGroup.add(wheel);
    });

    // Headlights
    const headlightGeom = new THREE.CylinderGeometry(0.018, 0.018, 0.01, 8);
    const headlightMat = new THREE.MeshBasicMaterial({ color: 0xffffcc });
    [-0.10, 0.10].forEach((z) => {
      const hl = new THREE.Mesh(headlightGeom, headlightMat);
      hl.position.set(0.28, 0.04, z);
      hl.rotation.z = Math.PI / 2;
      roverGroup.add(hl);
    });

    roverGroup.position.set(0, 0.12, 0);
    scene.add(roverGroup);
    roverRef.current = roverGroup;

    // Target marker
    const targetGroup = new THREE.Group();
    
    // Base ring
    const ringGeom = new THREE.RingGeometry(0.3, 0.5, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.02;
    targetGroup.add(ring);

    // Vertical beam
    const beamGeom = new THREE.CylinderGeometry(0.05, 0.05, 2, 8);
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.4,
    });
    const beam = new THREE.Mesh(beamGeom, beamMat);
    beam.position.y = 1;
    targetGroup.add(beam);

    // Flag
    const flagGeom = new THREE.BoxGeometry(0.4, 0.25, 0.02);
    const flagMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
    const flag = new THREE.Mesh(flagGeom, flagMat);
    flag.position.set(0.2, 1.9, 0);
    targetGroup.add(flag);

    targetGroup.position.set(sceneData.target[0], 0, sceneData.target[1]);
    scene.add(targetGroup);
    targetRef.current = targetGroup;

    // Path line
    const pathMat = new THREE.LineBasicMaterial({
      color: 0xffaa00,
      linewidth: 2,
      transparent: true,
      opacity: 0.8,
    });
    const pathGeom = new THREE.BufferGeometry();
    const pathLine = new THREE.Line(pathGeom, pathMat);
    scene.add(pathLine);
    pathLineRef.current = pathLine;

    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameIdRef.current);
      controls.dispose();
      renderer.dispose();
      container?.removeChild(renderer.domElement);
    };
  }, [sceneData]);

  // Update rover position and rotation
  useEffect(() => {
    if (!roverRef.current || !roverPosition) return;
    // MuJoCo: x forward, y left → Three.js: x right, z forward
    // Actually MuJoCo and Three.js both use right-hand, but we map y → z for ground plane
    roverRef.current.position.set(
      roverPosition[0],
      roverPosition[2] + 0.12, // z is height in MuJoCo, add chassis offset
      roverPosition[1]
    );
    if (roverQuaternion) {
      const q = mujocoQuatToThree(roverQuaternion);
      // Convert from MuJoCo frame to Three.js frame (swap y/z)
      roverRef.current.quaternion.set(q.x, q.z, q.y, q.w);
    }
  }, [roverPosition, roverQuaternion]);

  // Update target position
  useEffect(() => {
    if (!targetRef.current || !targetPosition) return;
    targetRef.current.position.set(targetPosition[0], 0, targetPosition[1]);
  }, [targetPosition]);

  // Update path
  useEffect(() => {
    if (!pathLineRef.current || !path || path.length === 0) return;
    const points = path.map(([x, y]) => new THREE.Vector3(x, 0.1, y));
    pathLineRef.current.geometry.dispose();
    pathLineRef.current.geometry = new THREE.BufferGeometry().setFromPoints(points);
  }, [path]);

  // Camera follow (when running)
  useEffect(() => {
    if (!isRunning || !controlsRef.current || !roverPosition) return;
    controlsRef.current.target.set(roverPosition[0], 0.3, roverPosition[1]);
  }, [isRunning, roverPosition]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
      }}
    />
  );
}
