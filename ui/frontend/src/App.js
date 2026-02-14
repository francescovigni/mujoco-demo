import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';

// Resolve API / WS URLs relative to the current host so the app works
// both in local dev (CRA proxy) and behind a reverse-proxy in production.
const _base = process.env.REACT_APP_API_URL || '';
const API_URL  = _base || '';
const _wsproto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL   = process.env.REACT_APP_WS_URL
  || `${_wsproto}//${window.location.host}/ws/simulation`;

/* ------------------------------------------------------------------ */
/*  Collapsible panel                                                  */
/* ------------------------------------------------------------------ */
function Panel({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="panel">
      <button className="panel-header" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className={`chevron ${open ? 'open' : ''}`}>›</span>
      </button>
      {open && <div className="panel-body">{children}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */
function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [initialized, setInitialized] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const [targetX, setTargetX] = useState('10.0');
  const [targetY, setTargetY] = useState('5.0');

  const [currentFrame, setCurrentFrame] = useState(null);
  const [simState, setSimState] = useState({
    position: [0, 0], target: [10, 5], distance: 0, speed: 0, reward: 0,
  });

  /* camera */
  const [cameraDistance, setCameraDistance] = useState(5.4);
  const [cameraAzimuth, setCameraAzimuth] = useState(13.0);
  const [cameraElevation, setCameraElevation] = useState(-26.0);
  const cameraRef = useRef({ distance: 5.0, azimuth: 90.0, elevation: -20.0 });
  const dragRef   = useRef({ isDragging: false, lastX: 0, lastY: 0 });

  /* repulsive field */
  const [repulseRange, setRepulseRange]     = useState(1.2);
  const [repulseGain, setRepulseGain]       = useState(2.5);
  const [obstacleMargin, setObstacleMargin] = useState(0.45);

  /* rover params */
  const [speedFactor, setSpeedFactor]       = useState(1.0);
  const [numPathPoints, setNumPathPoints]   = useState(40);

  const wsRef  = useRef(null);
  const imgRef = useRef(null);

  /* ---- fetch models ---- */
  useEffect(() => {
    fetch(`${API_URL}/humanoids`)
      .then(r => r.json())
      .then(d => { setModels(d.humanoids); if (d.humanoids.length) setSelectedModel(d.humanoids[0].id); })
      .catch(() => {});
  }, []);

  /* ---- websocket ---- */
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setConnectionStatus('connecting');
    const ws = new WebSocket(WS_URL);
    ws.onopen  = () => setConnectionStatus('connected');
    ws.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.type === 'state') {
        setSimState({ position: d.position, target: d.target, distance: d.distance, speed: d.height, reward: d.reward });
        if (d.frame) setCurrentFrame(`data:image/jpeg;base64,${d.frame}`);
      } else if (d.type === 'episode_end') {
        setIsRunning(false);
      }
    };
    ws.onerror = () => setConnectionStatus('disconnected');
    ws.onclose = () => { setConnectionStatus('disconnected'); wsRef.current = null; };
    wsRef.current = ws;
  }, []);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  /* ---- helpers ---- */
  const wsSend = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(obj));
  }, []);

  const sendCamera = useCallback((dist, az, el) => {
    wsSend({ type: 'camera_update', distance: dist, azimuth: az, elevation: el });
  }, [wsSend]);

  const sendRepulsive = useCallback((range, gain, margin) => {
    wsSend({ type: 'repulsive_params', repulse_range: range, repulse_gain: gain, obstacle_margin: margin });
  }, [wsSend]);

  const sendRoverParams = useCallback((speed, points) => {
    wsSend({ type: 'rover_params', speed_factor: speed, num_path_points: points });
  }, [wsSend]);

  /* ---- actions ---- */
  const handleInit = async () => {
    if (!selectedModel) return;
    try {
      const r = await fetch(`${API_URL}/initialize`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel }),
      });
      if (r.ok) { setInitialized(true); connectWS(); }
    } catch {}
  };

  const handleStart = () => { wsSend({ type: 'start' }); setIsRunning(true); };
  const handleStop  = () => { wsSend({ type: 'stop'  }); setIsRunning(false); };
  const handleReset = () => { wsSend({ type: 'reset' }); setIsRunning(false); };

  const handleSetTarget = () => {
    const x = parseFloat(targetX), y = parseFloat(targetY);
    if (!isNaN(x) && !isNaN(y)) wsSend({ type: 'set_target', x, y });
  };

  /* ---- camera interaction ---- */
  const onMouseDown = useCallback((e) => {
    dragRef.current = { isDragging: true, lastX: e.clientX, lastY: e.clientY };
    e.preventDefault();
  }, []);

  const onMouseMove = useCallback((e) => {
    const dr = dragRef.current;
    if (!dr.isDragging) return;
    const cam = cameraRef.current;
    cam.azimuth   += (e.clientX - dr.lastX) * 0.5;
    cam.elevation  = Math.max(-89, Math.min(89, cam.elevation + (e.clientY - dr.lastY) * 0.5));
    dr.lastX = e.clientX; dr.lastY = e.clientY;
    setCameraAzimuth(cam.azimuth); setCameraElevation(cam.elevation);
    sendCamera(cam.distance, cam.azimuth, cam.elevation);
  }, [sendCamera]);

  const onMouseUp = useCallback(() => { dragRef.current.isDragging = false; }, []);

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      const cam = cameraRef.current;
      cam.distance = Math.max(1, Math.min(20, cam.distance + e.deltaY * 0.01));
      setCameraDistance(cam.distance);
      sendCamera(cam.distance, cam.azimuth, cam.elevation);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [sendCamera, currentFrame]);

  const resetCamera = () => {
    const c = { distance: 5, azimuth: 90, elevation: -20 };
    cameraRef.current = { ...c };
    setCameraDistance(c.distance); setCameraAzimuth(c.azimuth); setCameraElevation(c.elevation);
    sendCamera(c.distance, c.azimuth, c.elevation);
  };

  /* ---- connection dot ---- */
  const connColor = connectionStatus === 'connected' ? '#34d399'
                   : connectionStatus === 'connecting' ? '#fbbf24' : '#666';

  /* ================================================================ */
  return (
    <div className="app">
      {/* ---------- sidebar ---------- */}
      <aside className="sidebar">
        <div className="sidebar-top">
          {/* brand */}
          <h1 className="brand">Rover&nbsp;Nav</h1>
          <span className="conn-dot" style={{ background: connColor }} title={connectionStatus} />
        </div>

        {/* project info */}
        <Panel title="About" defaultOpen={!initialized}>
          <p className="about-text">
            A 4-wheel differential-drive rover navigates to a target in a
            MuJoCo physics sim. A <em>potential-field</em> controller blends
            attractive pull toward the goal with repulsive push from obstacles.
            The planned path is drawn in real-time on the 3-D scene.
          </p>
          <div className="tech-tags">
            <span className="tag">MuJoCo</span>
            <span className="tag">Python</span>
            <span className="tag">FastAPI</span>
            <span className="tag">WebSocket</span>
            <span className="tag">React</span>
          </div>
        </Panel>

        {/* model select + init */}
        {!initialized && (
          <Panel title="Setup">
            <div className="model-list">
              {models.map(m => (
                <button
                  key={m.id}
                  className={`model-btn ${selectedModel === m.id ? 'active' : ''}`}
                  onClick={() => setSelectedModel(m.id)}
                >
                  {m.name}
                </button>
              ))}
            </div>
            <button className="btn btn-primary" onClick={handleInit} disabled={!selectedModel}>
              Initialize
            </button>
          </Panel>
        )}

        {initialized && (
          <>
            {/* controls */}
            <Panel title="Controls">
              <Slider label="Speed" value={speedFactor} min={0.1} max={3} step={0.1} unit="×"
                onChange={v => { setSpeedFactor(v); sendRoverParams(v, numPathPoints); }} />
              <div className="btn-row" style={{ marginTop: '0.4rem' }}>
                {!isRunning
                  ? <button className="btn btn-go" onClick={handleStart} disabled={connectionStatus !== 'connected'}>Start</button>
                  : <button className="btn btn-stop" onClick={handleStop}>Stop</button>
                }
                <button className="btn btn-outline" onClick={handleReset} disabled={connectionStatus !== 'connected'}>Reset</button>
              </div>
            </Panel>

            {/* target */}
            <Panel title="Target">
              <div className="input-row">
                <label>
                  <span className="input-label">X</span>
                  <input type="number" step="0.5" value={targetX} onChange={e => setTargetX(e.target.value)} />
                </label>
                <label>
                  <span className="input-label">Y</span>
                  <input type="number" step="0.5" value={targetY} onChange={e => setTargetY(e.target.value)} />
                </label>
                <button className="btn btn-sm" onClick={handleSetTarget}>Set</button>
              </div>
            </Panel>

            {/* repulsive field */}
            <Panel title="Path Planner" defaultOpen={true}>
              <Slider label="Path Points" value={numPathPoints} min={10} max={100} step={5} unit=""
                onChange={v => { setNumPathPoints(v); sendRoverParams(speedFactor, v); }} />
              <Slider label="Range" value={repulseRange} min={0.2} max={3} step={0.1} unit="m"
                onChange={v => { setRepulseRange(v); sendRepulsive(v, repulseGain, obstacleMargin); }} />
              <Slider label="Strength" value={repulseGain} min={0} max={8} step={0.2} unit=""
                onChange={v => { setRepulseGain(v); sendRepulsive(repulseRange, v, obstacleMargin); }} />
              <Slider label="Margin" value={obstacleMargin} min={0.1} max={1.5} step={0.05} unit="m"
                onChange={v => { setObstacleMargin(v); sendRepulsive(repulseRange, repulseGain, v); }} />
              <button className="btn btn-link" onClick={() => {
                setRepulseRange(1.2); setRepulseGain(2.5); setObstacleMargin(0.45);
                setNumPathPoints(40); setSpeedFactor(1.0);
                sendRepulsive(1.2, 2.5, 0.45);
                sendRoverParams(1.0, 40);
              }}>Reset defaults</button>
            </Panel>

            {/* camera */}
            <Panel title="Camera" defaultOpen={false}>
              <div className="cam-stats">
                <span>Dist {cameraDistance.toFixed(1)}m</span>
                <span>Az {cameraAzimuth.toFixed(0)}°</span>
                <span>El {cameraElevation.toFixed(0)}°</span>
              </div>
              <button className="btn btn-link" onClick={resetCamera}>Reset camera</button>
              <p className="help-hint">Drag viewport to orbit · scroll to zoom</p>
            </Panel>
          </>
        )}

        <div className="sidebar-footer">
          Built with MuJoCo + React
        </div>
      </aside>

      {/* ---------- viewport ---------- */}
      <main
        className="viewport"
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {currentFrame ? (
          <>
            <img
              ref={imgRef}
              src={currentFrame}
              alt="Simulation"
              className="sim-frame"
              onMouseDown={onMouseDown}
              draggable={false}
            />
            {/* telemetry strip */}
            <div className="telemetry">
              <span>Pos ({simState.position[0].toFixed(1)}, {simState.position[1].toFixed(1)})</span>
              <span className="sep" />
              <span>Dist {simState.distance.toFixed(2)}m</span>
              <span className="sep" />
              <span>Speed {simState.speed.toFixed(2)} m/s</span>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-icon" />
            <p>Select a model and click <strong>Initialize</strong> to begin.</p>
          </div>
        )}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Slider                                                             */
/* ------------------------------------------------------------------ */
function Slider({ label, value, min, max, step, unit, onChange }) {
  return (
    <div className="slider">
      <div className="slider-head">
        <span className="slider-label">{label}</span>
        <span className="slider-value">{step >= 1 ? Math.round(value) : value.toFixed(step < 0.1 ? 2 : 1)}{unit && <small> {unit}</small>}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

export default App;
