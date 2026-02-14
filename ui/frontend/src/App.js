import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import Scene3D from './Scene3D';

// Resolve API / WS URLs relative to the current host so the app works
// both in local dev (CRA proxy) and behind a reverse-proxy in production.
const _pathPrefix = process.env.PUBLIC_URL || '';
const API_URL  = process.env.REACT_APP_API_URL || _pathPrefix;
const _wsproto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL   = process.env.REACT_APP_WS_URL
  || `${_wsproto}//${window.location.host}${_pathPrefix}/ws/simulation`;

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

  // Scene data from /initialize (obstacles, target, etc.)
  const [sceneData, setSceneData] = useState(null);
  
  // Real-time rover state from WebSocket
  const [roverPosition, setRoverPosition] = useState([0, 0, 0.12]);
  const [roverQuaternion, setRoverQuaternion] = useState([1, 0, 0, 0]);
  const [targetPosition, setTargetPosition] = useState([10, 5]);
  const [path, setPath] = useState([]);
  
  // Telemetry
  const [simState, setSimState] = useState({
    distance: 0, speed: 0,
  });

  /* repulsive field */
  const [repulseRange, setRepulseRange]     = useState(1.2);
  const [repulseGain, setRepulseGain]       = useState(2.5);
  const [obstacleMargin, setObstacleMargin] = useState(0.45);

  /* rover params */
  const [speedFactor, setSpeedFactor]       = useState(1.0);
  const [numPathPoints, setNumPathPoints]   = useState(40);

  const wsRef  = useRef(null);

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
        // Update rover pose
        if (d.position) setRoverPosition(d.position);
        if (d.quaternion) setRoverQuaternion(d.quaternion);
        if (d.target) setTargetPosition(d.target);
        if (d.path) setPath(d.path);
        setSimState({
          distance: d.distance || 0,
          speed: d.speed || 0,
        });
      } else if (d.type === 'target_update') {
        if (d.target) setTargetPosition(d.target);
      } else if (d.type === 'path_update') {
        if (d.path) setPath(d.path);
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
      if (r.ok) {
        const data = await r.json();
        if (data.scene) {
          setSceneData(data.scene);
          setRoverPosition(data.scene.roverPosition || [0, 0, 0.12]);
          setRoverQuaternion(data.scene.roverQuaternion || [1, 0, 0, 0]);
          setTargetPosition(data.scene.target || [10, 5]);
        }
        setInitialized(true);
        connectWS();
      }
    } catch (e) {
      console.error('Init error:', e);
    }
  };

  const handleStart = () => { wsSend({ type: 'start' }); setIsRunning(true); };
  const handleStop  = () => { wsSend({ type: 'stop'  }); setIsRunning(false); };
  const handleReset = () => { wsSend({ type: 'reset' }); setIsRunning(false); };

  const handleSetTarget = () => {
    const x = parseFloat(targetX), y = parseFloat(targetY);
    if (!isNaN(x) && !isNaN(y)) {
      wsSend({ type: 'set_target', x, y });
      setTargetPosition([x, y]);
    }
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
            <span className="tag">Three.js</span>
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

            {/* path planner */}
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

            {/* camera help */}
            <Panel title="Camera" defaultOpen={false}>
              <p className="help-hint">
                Left-drag to orbit · Right-drag to pan · Scroll to zoom
              </p>
            </Panel>
          </>
        )}

        <div className="sidebar-footer">
          Built with MuJoCo + Three.js
        </div>
      </aside>

      {/* ---------- viewport ---------- */}
      <main className="viewport">
        {sceneData ? (
          <>
            <Scene3D
              sceneData={sceneData}
              roverPosition={roverPosition}
              roverQuaternion={roverQuaternion}
              targetPosition={targetPosition}
              path={path}
              isRunning={isRunning}
            />
            {/* telemetry strip */}
            <div className="telemetry">
              <span>Pos ({roverPosition[0].toFixed(1)}, {roverPosition[1].toFixed(1)})</span>
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
