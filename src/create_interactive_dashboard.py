import numpy as np
import json
import os
import csv

def generate_dashboard():
    src_dir = os.path.dirname(__file__)
    data_dir = os.path.join(src_dir, "..", "data")
    docs_dir = os.path.join(src_dir, "..", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Constants & Conversion factors
    # ----------------------------------------------------
    GM_SUN_KM3_S2 = 1.32712440041279419e11
    DAY_IN_S = 86400.0
    AU_IN_KM = 149597870.7
    GM_CONVERSION = (DAY_IN_S ** 2) / (AU_IN_KM ** 3)
    GM_SUN = GM_SUN_KM3_S2 * GM_CONVERSION # ~0.000295912
    M_EARTH_SOLAR = 3.003e-6
    GM_EARTH = M_EARTH_SOLAR * GM_SUN

    # Prepare data for both presets
    presets_data = {}
    
    for preset in ["solar_system", "hot_jupiter"]:
        npz_path = os.path.join(data_dir, f"simulation_{preset}.npz")
        csv_path = os.path.join(data_dir, f"collisions_{preset}.csv")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"❌ Preset data not found at {npz_path}. Run simulator.py first.")
            
        print(f"🌐 Loading simulation results for '{preset}'...")
        data = np.load(npz_path)
        t = data["t"]
        gms = data["gms"]
        r = data["r"]
        names = list(data["names"])
        
        # Downsample for browser performance (from 6000 states to 3000)
        target_steps = 3000
        downsample_factor = max(1, len(t) // target_steps)
        ds_indices = list(range(0, len(t), downsample_factor))
        if ds_indices[-1] != len(t) - 1:
            ds_indices.append(len(t) - 1)
            
        t_ds = t[ds_indices]
        gms_ds = gms[ds_indices]
        r_ds = r[ds_indices]
        
        # Convert time to years
        time_years = np.round(t_ds / 365.25, 2).tolist()
        
        # Pre-round coordinates and masses to compress the JSON file size
        orbit_x = {}
        orbit_y = {}
        mass_history = {}
        
        # We only save active trajectories of particles to save space.
        # If a particle is never active (mass = 0 at start), we skip it.
        # If it becomes inactive, we keep its coordinate as 0.0 or let JS know.
        for i, name in enumerate(names):
            # Check if this body is ever active
            if np.max(gms_ds[:, i]) > 0:
                orbit_x[i] = np.round(r_ds[:, i, 0], 4).tolist()
                orbit_y[i] = np.round(r_ds[:, i, 1], 4).tolist()
                # Mass in Earth Masses
                mass_history[i] = np.round(gms_ds[:, i] / GM_EARTH, 3).tolist()
        
        # Read collisions CSV
        collisions = []
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    collisions.append({
                        "time_years": round(float(row["time_years"]), 2),
                        "survivor_idx": int(row["survivor_idx"]),
                        "merged_idx": int(row["merged_idx"]),
                        "survivor_mass_before": round(float(row["survivor_mass_before_earth"]), 3),
                        "merged_mass_before": round(float(row["merged_mass_before_earth"]), 3),
                        "new_mass": round(float(row["new_mass_earth"]), 3),
                        "dist_au": round(float(row["dist_au"]), 3)
                    })
        
        # Identify top 5 bodies (excluding star index 0) for plotting
        final_masses = gms_ds[-1, 1:] / GM_EARTH
        top_5_local_indices = np.argsort(final_masses)[-5:][::-1]
        top_5_global_indices = (top_5_local_indices + 1).tolist() # adjust for star offset
        
        presets_data[preset] = {
            "time_years": time_years,
            "orbit_x": orbit_x,
            "orbit_y": orbit_y,
            "mass_history": mass_history,
            "collisions": collisions,
            "top_bodies": top_5_global_indices,
            "num_bodies": len(names)
        }

    # Embed data in HTML template
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Protoplanetary Disk Accretion & Migration Simulator</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f17;
            --panel-bg: rgba(17, 22, 34, 0.85);
            --border-color: rgba(48, 54, 61, 0.4);
            --text-color: #c9d1d9;
            --accent-color: #58a6ff;
            --accent-glow: rgba(88, 166, 255, 0.35);
            --danger-color: #f85149;
            --success-color: #56e39f;
            --star-color: #ffca3a;
            --planet-color: #58a6ff;
            --planetesimal-color: #8b949e;
        }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            overflow: hidden;
            display: flex;
            height: 100vh;
        }
        
        #sidebar {
            width: 380px;
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border-color);
            padding: 24px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 16px;
            z-index: 10;
            overflow-y: auto;
            backdrop-filter: blur(16px);
            box-shadow: 8px 0 32px rgba(0,0,0,0.6);
        }
        
        #main-content {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background-color: #05070c;
        }
        
        #canvas-container {
            flex-grow: 1;
            position: relative;
            cursor: grab;
            overflow: hidden;
        }
        
        #canvas-container:active {
            cursor: grabbing;
        }
        
        canvas#orbitCanvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        
        #chart-container {
            height: 250px;
            background-color: rgba(10, 14, 23, 0.95);
            border-top: 1px solid var(--border-color);
            padding: 16px 24px;
            box-sizing: border-box;
            position: relative;
        }
        
        h1 {
            font-size: 22px;
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #58a6ff, #bd93f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            font-size: 11px;
            color: #8b949e;
            margin-top: -2px;
            margin-bottom: 8px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.5px;
        }
        
        h2 {
            font-size: 12px;
            margin: 0 0 6px 0;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        
        .control-group {
            background: rgba(22, 27, 34, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .btn-row {
            display: flex;
            gap: 8px;
        }
        
        button {
            flex-grow: 1;
            background: rgba(33, 38, 45, 0.8);
            border: 1px solid var(--border-color);
            color: #c9d1d9;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-family: 'Outfit', sans-serif;
            transition: all 0.2s ease;
        }
        
        button:hover {
            background: rgba(56, 62, 71, 0.8);
            border-color: #8b949e;
        }
        
        button.active {
            background: var(--accent-color);
            color: #0b0f17;
            border-color: var(--accent-color);
            font-weight: 600;
            box-shadow: 0 0 12px var(--accent-glow);
        }
        
        .slider-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: #8b949e;
        }
        
        .slider-label span:last-child {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-color);
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #21262d;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-color);
            cursor: pointer;
            box-shadow: 0 0 6px var(--accent-glow);
            transition: transform 0.1s ease;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        
        .stat-card {
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        
        .stat-title {
            font-size: 11px;
            color: #8b949e;
            text-transform: uppercase;
        }
        
        .stat-val {
            font-size: 16px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: #f0f6fc;
        }
        
        #collision-panel {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            background: rgba(22, 27, 34, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px;
            max-height: 250px;
        }
        
        #collision-log {
            flex-grow: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            padding-right: 4px;
        }
        
        .log-entry {
            padding: 6px 8px;
            border-radius: 4px;
            background: rgba(33, 38, 45, 0.5);
            border-left: 3px solid var(--accent-color);
            animation: fadeIn 0.3s ease;
        }
        
        .log-entry.swallowed {
            border-left-color: var(--danger-color);
        }
        
        .log-entry.grow {
            border-left-color: var(--success-color);
        }
        
        #hud-overlay {
            position: absolute;
            top: 24px;
            right: 24px;
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            padding: 12px 18px;
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            pointer-events: none;
        }
        
        .hud-row {
            display: flex;
            justify-content: space-between;
            gap: 24px;
        }
        
        .hud-value {
            color: var(--accent-color);
            font-weight: 600;
        }
        
        #canvas-tip {
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            color: #8b949e;
            background: rgba(13, 17, 23, 0.7);
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            pointer-events: none;
        }

        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #8b949e;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <div>
            <h1>Solar System Accretion</h1>
            <div class="subtitle">N-BODY ACCRETION & MIGRATION</div>
        </div>
        
        <!-- Preset Selection -->
        <div class="control-group">
            <h2>Select Physical Scenario</h2>
            <div class="btn-row">
                <button id="btn-preset-solar" class="active" onclick="switchPreset('solar_system')">Solar System</button>
                <button id="btn-preset-jupiter" onclick="switchPreset('hot_jupiter')">Hot Jupiter</button>
            </div>
            <p style="font-size: 11.5px; color: #8b949e; margin: 2px 0 0 0; line-height: 1.4;">
                <span id="preset-desc">Solar System: Tenuous gas disk. Planetesimals merge into stable orbits without substantial inward migration.</span>
            </p>
        </div>
        
        <!-- Playback Controls -->
        <div class="control-group">
            <h2>Playback & Speed</h2>
            <div class="btn-row">
                <button id="btn-play-pause" class="active" onclick="togglePlay()">Pause</button>
                <button id="btn-reset" onclick="resetSim()">Restart</button>
            </div>
            
            <div class="slider-row">
                <div class="slider-label">
                    <span>Playback Speed</span>
                    <span id="speed-val">1.5x</span>
                </div>
                <input type="range" id="speed-slider" min="0.1" max="25.0" step="0.1" value="1.5" oninput="updateSpeed(this.value)">
            </div>
            
            <div class="slider-row">
                <div class="slider-label">
                    <span>Timeline (Years)</span>
                    <span id="time-val">0.0</span>
                </div>
                <input type="range" id="time-slider" min="0" max="100" value="0" oninput="seekTo(this.value)">
            </div>
        </div>
        
        <!-- Zoom Controls -->
        <div class="control-group">
            <h2>Zoom & View Controls</h2>
            <div class="btn-row">
                <button onclick="zoomIn()">+ Zoom In</button>
                <button onclick="zoomOut()">- Zoom Out</button>
                <button onclick="resetZoom()">Reset View</button>
            </div>
        </div>
        
        <!-- Stats Panel -->
        <div class="control-group">
            <h2>Current State</h2>
            <div class="stat-grid">
                <div class="stat-card">
                    <span class="stat-title">Remaining Bodies</span>
                    <span class="stat-val" id="stat-count">-</span>
                </div>
                <div class="stat-card">
                    <span class="stat-title">Max Planet Mass</span>
                    <span class="stat-val" id="stat-max-mass">-</span>
                </div>
            </div>
        </div>
        
        <!-- Accretion Log -->
        <div id="collision-panel">
            <h2>Accretion Event Log</h2>
            <div id="collision-log">
                <!-- Javascript populated -->
            </div>
        </div>
    </div>
    
    <div id="main-content">
        <div id="canvas-container">
            <canvas id="orbitCanvas"></canvas>
            
            <div id="hud-overlay">
                <div class="hud-row">
                    <span>Simulation Age:</span>
                    <span class="hud-value" id="hud-time">0.00 yr</span>
                </div>
                <div class="hud-row">
                    <span>Total Planetesimals:</span>
                    <span class="hud-value" id="hud-count">-</span>
                </div>
                <div class="hud-row">
                    <span>Zoom Scale:</span>
                    <span class="hud-value" id="hud-zoom">1.0x</span>
                </div>
            </div>
            
            <div id="canvas-tip">Scroll to Zoom • Drag to Pan • Hover particles to view details</div>
        </div>
        
        <div id="chart-container">
            <canvas id="massChart"></canvas>
        </div>
    </div>

    <script>
        // Embed the downsampled simulation data directly
        const datasets = %DATA_PAYLOAD%;
        
        // Interactive state
        let currentPreset = "solar_system";
        let currentIdx = 0;
        let isPlaying = false;
        let playSpeed = 1.5;
        let lastTimestamp = 0;
        
        // Visual angle tracking to resolve stroboscopic aliasing at high speeds
        let visualAngles = {};
        let lastTimeYears = null;
        let frameStates = {};
        
        // Canvas & Coordinate variables
        const canvas = document.getElementById("orbitCanvas");
        const ctx = canvas.getContext("2d");
        let zoom = 1.0;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let startDragX = 0;
        let startDragY = 0;
        let hoveredBodyId = null;
        
        // ChartJS instance
        let massChart = null;

        // Colors mapping
        const colors = {
            star: "#ffca3a",
            embryo: "#bd93f9",
            planetesimal: "#58a6ff",
            dead: "rgba(0,0,0,0)",
            trails: [
                "#ff5233", "#ffb03a", "#56e39f", "#9bf6ff", "#bd93f9"
            ]
        };

        function resizeCanvas() {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
            drawFrame();
        }

        window.addEventListener("resize", resizeCanvas);
        
        // Dragging & Panning
        canvas.addEventListener("mousedown", (e) => {
            isDragging = true;
            startDragX = e.clientX - offsetX;
            startDragY = e.clientY - offsetY;
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        canvas.addEventListener("mousemove", (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (isDragging) {
                offsetX = e.clientX - startDragX;
                offsetY = e.clientY - startDragY;
                drawFrame();
            } else {
                // Check hover
                checkHover(mouseX, mouseY);
            }
        });

        // Zooming with mouse scroll
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Zoom center relative to mouse
            const dataX = (mouseX - canvas.width / 2 - offsetX) / zoom;
            const dataY = (mouseY - canvas.height / 2 - offsetY) / zoom;
            
            const zoomFactor = 1.15;
            if (e.deltaY < 0) {
                zoom *= zoomFactor;
            } else {
                zoom /= zoomFactor;
            }
            
            zoom = Math.max(0.1, Math.min(zoom, 100.0));
            
            offsetX = mouseX - canvas.width / 2 - dataX * zoom;
            offsetY = mouseY - canvas.height / 2 - dataY * zoom;
            
            document.getElementById("hud-zoom").innerText = zoom.toFixed(1) + "x";
            drawFrame();
        });

        // Zoom Functions
        function zoomIn() {
            zoom *= 1.3;
            zoom = Math.min(zoom, 100.0);
            document.getElementById("hud-zoom").innerText = zoom.toFixed(1) + "x";
            drawFrame();
        }
        function zoomOut() {
            zoom /= 1.3;
            zoom = Math.max(0.1, zoom);
            document.getElementById("hud-zoom").innerText = zoom.toFixed(1) + "x";
            drawFrame();
        }
        function resetZoom() {
            zoom = 1.0;
            offsetX = 0;
            offsetY = 0;
            document.getElementById("hud-zoom").innerText = "1.0x";
            drawFrame();
        }

        // Toggle Preset
        function switchPreset(preset) {
            currentPreset = preset;
            currentIdx = 0;
            hoveredBodyId = null;
            visualAngles = {};
            lastTimeYears = null;
            frameStates = {};
            
            // Update button styles
            document.getElementById("btn-preset-solar").className = (preset === "solar_system") ? "active" : "";
            document.getElementById("btn-preset-jupiter").className = (preset === "hot_jupiter") ? "active" : "";
            
            // Update description text
            const desc = document.getElementById("preset-desc");
            if (preset === "solar_system") {
                desc.innerText = "Solar System: Tenuous gas disk. Planetesimals merge into stable orbits without substantial inward migration.";
                zoom = 1.0;
                offsetX = 0;
                offsetY = 0;
            } else {
                desc.innerText = "Hot Jupiter: Dense gas disk. The massive headwind dampens orbits, forcing growing protoplanets to spiral rapidly into the star.";
                zoom = 1.0;
                offsetX = 0;
                offsetY = 0;
            }
            document.getElementById("hud-zoom").innerText = zoom.toFixed(1) + "x";

            // Initialize timeline slider limit
            const ds = datasets[currentPreset];
            const timeSlider = document.getElementById("time-slider");
            timeSlider.max = ds.time_years.length - 1;
            timeSlider.value = 0;

            // Rebuild Chart
            initializeChart();
            
            // Rebuild Accretion Log
            buildAccretionLog();
            
            drawFrame();
        }

        // Toggle Play
        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById("btn-play-pause").innerText = isPlaying ? "Pause" : "Play";
            document.getElementById("btn-play-pause").className = isPlaying ? "active" : "";
            if (isPlaying) {
                lastTimestamp = performance.now();
                requestAnimationFrame(animationLoop);
            }
        }

        // Reset
        function resetSim() {
            currentIdx = 0;
            document.getElementById("time-slider").value = 0;
            visualAngles = {};
            lastTimeYears = null;
            frameStates = {};
            drawFrame();
            if (isPlaying) {
                isPlaying = false;
                togglePlay();
            }
        }

        // Seek
        function seekTo(val) {
            currentIdx = parseInt(val);
            visualAngles = {};
            lastTimeYears = null;
            frameStates = {};
            drawFrame();
        }

        // Speed
        function updateSpeed(val) {
            playSpeed = parseFloat(val);
            document.getElementById("speed-val").innerText = playSpeed.toFixed(1) + "x";
        }

        // Animation Loop
        function animationLoop(timestamp) {
            if (!isPlaying) return;
            
            const ds = datasets[currentPreset];
            const maxIdx = ds.time_years.length - 1;
            
            if (currentIdx >= maxIdx) {
                isPlaying = false;
                document.getElementById("btn-play-pause").innerText = "Play";
                document.getElementById("btn-play-pause").className = "";
                return;
            }

            const elapsed = timestamp - lastTimestamp;
            // Advance frames based on speed & elapsed time
            // 1.5x speed means 1.5 years of simulation per real second
            const yearsPerStep = ds.time_years[1] - ds.time_years[0];
            const stepsToAdvance = (playSpeed * (elapsed / 1000.0)) / yearsPerStep;
            
            currentIdx = Math.min(maxIdx, currentIdx + stepsToAdvance);
            
            document.getElementById("time-slider").value = Math.floor(currentIdx);
            
            drawFrame();
            
            lastTimestamp = timestamp;
            requestAnimationFrame(animationLoop);
        }

        // Polar interpolation helper to guarantee perfectly smooth circular/elliptical movement
        function getInterpolatedState(ds, bodyId, idx) {
            const step = Math.floor(idx);
            const frac = idx - step;
            const maxIdx = ds.time_years.length - 1;
            
            const x_arr = ds.orbit_x[bodyId];
            const y_arr = ds.orbit_y[bodyId];
            const m_arr = ds.mass_history[bodyId];
            
            if (!x_arr || step >= x_arr.length) return null;
            
            const m1 = m_arr[step];
            if (m1 <= 0) return null;
            
            // Central Star remains stationary at the origin
            if (parseInt(bodyId) === 0) {
                return {
                    x: 0,
                    y: 0,
                    r: 0,
                    theta: 0,
                    mass: m1
                };
            }
            
            if (step === maxIdx) {
                const x = x_arr[step];
                const y = y_arr[step];
                return {
                    x: x,
                    y: y,
                    r: Math.sqrt(x*x + y*y),
                    theta: Math.atan2(y, x),
                    mass: m1
                };
            }
            
            const m2 = m_arr[step + 1];
            if (m2 <= 0) {
                const x = x_arr[step];
                const y = y_arr[step];
                return {
                    x: x,
                    y: y,
                    r: Math.sqrt(x*x + y*y),
                    theta: Math.atan2(y, x),
                    mass: m1
                };
            }
            
            const x1 = x_arr[step];
            const y1 = y_arr[step];
            const x2 = x_arr[step + 1];
            const y2 = y_arr[step + 1];
            
            const r1 = Math.sqrt(x1*x1 + y1*y1);
            const r2 = Math.sqrt(x2*x2 + y2*y2);
            
            const theta1 = Math.atan2(y1, x1);
            const theta2 = Math.atan2(y2, x2);
            
            // Handle angle wrap-around across the -PI to PI boundary
            let dTheta = theta2 - theta1;
            while (dTheta < -Math.PI) dTheta += 2 * Math.PI;
            while (dTheta > Math.PI) dTheta -= 2 * Math.PI;
            
            const r_interp = r1 + frac * (r2 - r1);
            const theta_interp = theta1 + frac * dTheta;
            const mass_interp = m1 + frac * (m2 - m1);
            
            return {
                x: r_interp * Math.cos(theta_interp),
                y: r_interp * Math.sin(theta_interp),
                r: r_interp,
                theta: theta_interp,
                mass: mass_interp
            };
        }

        // Check particle hover using cached frame states (fast O(N))
        function checkHover(mouseX, mouseY) {
            let found = null;
            
            const centerX = canvas.width / 2 + offsetX;
            const centerY = canvas.height / 2 + offsetY;
            const scale = Math.min(canvas.width, canvas.height) / 25.0 * zoom;

            for (const bodyId in frameStates) {
                const state = frameStates[bodyId];
                if (!state) continue;
                
                const screenX = centerX + state.x * scale;
                const screenY = centerY + state.y * scale;
                
                const dx = mouseX - screenX;
                const dy = mouseY - screenY;
                const dist = Math.sqrt(dx*dx + dy*dy);
                
                const drawRadius = getDrawRadius(state.mass, bodyId);
                
                if (dist <= Math.max(5, drawRadius)) {
                    found = bodyId;
                    break;
                }
            }
            
            if (hoveredBodyId !== found) {
                hoveredBodyId = found;
                drawFrame();
            }
        }

        // Get particle radius in screen pixels
        function getDrawRadius(mass, bodyId) {
            if (parseInt(bodyId) === 0) return 8; // Star size
            
            // Base size based on mass (m^(1/3))
            const baseSize = Math.max(2.0, Math.pow(mass, 1/3) * 3.5);
            return baseSize;
        }

        // Draw current frame
        // Draw current frame
        function drawFrame() {
            const ds = datasets[currentPreset];
            const step = Math.floor(currentIdx);
            const frac = currentIdx - step;
            const nextStep = Math.min(ds.time_years.length - 1, step + 1);
            const t_yr = ds.time_years[step] + frac * (ds.time_years[nextStep] - ds.time_years[step]);
            
            // Calculate dt_years
            let dt_years = 0;
            let resetAngles = false;
            if (lastTimeYears === null || Math.abs(t_yr - lastTimeYears) > 0.25 || t_yr < lastTimeYears) {
                resetAngles = true;
            } else {
                dt_years = t_yr - lastTimeYears;
            }
            lastTimeYears = t_yr;

            // Update HUD values
            document.getElementById("hud-time").innerText = t_yr.toFixed(2) + " yr";
            document.getElementById("time-val").innerText = t_yr.toFixed(0);
            
            // Calculate active bodies & max mass
            let activeCount = 0;
            let maxMass = 0;
            for (const bodyId in ds.mass_history) {
                if (parseInt(bodyId) === 0) continue; // Skip star
                const mass = ds.mass_history[bodyId][step];
                if (mass > 0) {
                    activeCount++;
                    if (mass > maxMass) maxMass = mass;
                }
            }
            
            document.getElementById("hud-count").innerText = activeCount;
            document.getElementById("stat-count").innerText = activeCount;
            document.getElementById("stat-max-mass").innerText = maxMass.toFixed(2) + " M⊕";

            // Clear screen
            ctx.fillStyle = "#05070c";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw radial grid lines
            drawGrid();

            // Screen barycenter
            const centerX = canvas.width / 2 + offsetX;
            const centerY = canvas.height / 2 + offsetY;
            const scale = Math.min(canvas.width, canvas.height) / 25.0 * zoom;

            // Compute and cache all visual states for this frame, applying physical Keplerian angular velocity to solve stroboscopic aliasing
            frameStates = {};
            for (const bodyId in ds.orbit_x) {
                const state = getInterpolatedState(ds, bodyId, currentIdx);
                if (!state) continue;
                
                if (resetAngles || visualAngles[bodyId] === undefined) {
                    visualAngles[bodyId] = state.theta;
                } else if (state.r > 0.04) {
                    // Angular velocity omega = 2*pi / T = 2*pi * a^(-1.5) radians/year
                    const omega = 2 * Math.PI * Math.pow(state.r, -1.5);
                    visualAngles[bodyId] += omega * dt_years;
                }
                
                state.theta = visualAngles[bodyId];
                state.x = state.r * Math.cos(state.theta);
                state.y = state.r * Math.sin(state.theta);
                
                frameStates[bodyId] = state;
            }

            // Draw orbits/trails of top 5 bodies using cached frameStates
            drawTrails(ds, centerX, centerY, scale, frameStates);

            // Draw bodies using cached frameStates
            for (const bodyId in ds.orbit_x) {
                const state = frameStates[bodyId];
                if (!state) continue; // Dead/absorbed body
                
                const screenX = centerX + state.x * scale;
                const screenY = centerY + state.y * scale;
                
                const drawRadius = getDrawRadius(state.mass, bodyId);
                
                ctx.beginPath();
                ctx.arc(screenX, screenY, drawRadius, 0, 2*Math.PI);
                
                // Styles
                if (parseInt(bodyId) === 0) {
                    // Central Star
                    ctx.fillStyle = colors.star;
                    ctx.shadowColor = colors.star;
                    ctx.shadowBlur = 15;
                } else {
                    // Planetesimal/Protoplanet
                    const isTop = ds.top_bodies.includes(parseInt(bodyId));
                    if (isTop) {
                        const rank = ds.top_bodies.indexOf(parseInt(bodyId));
                        ctx.fillStyle = colors.trails[rank];
                        ctx.shadowColor = colors.trails[rank];
                        ctx.shadowBlur = 10;
                    } else {
                        ctx.fillStyle = colors.planetesimal;
                        ctx.shadowBlur = 0;
                    }
                }
                
                ctx.fill();
                ctx.shadowBlur = 0; // reset
                
                // Draw selection ring for hover
                if (hoveredBodyId === bodyId) {
                    ctx.beginPath();
                    ctx.arc(screenX, screenY, drawRadius + 4, 0, 2*Math.PI);
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    
                    // Render hover text info
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 13px Outfit";
                    const isStar = (parseInt(bodyId) === 0);
                    const nameStr = isStar ? "Central Sun" : "Body #" + bodyId;
                    const massStr = isStar ? "1.0 Solar Mass" : state.mass.toFixed(2) + " Earth Masses";
                    const distStr = isStar ? "Origin" : state.r.toFixed(2) + " AU";
                    
                    ctx.fillText(nameStr, screenX + drawRadius + 6, screenY - 6);
                    ctx.font = "12px JetBrains Mono";
                    ctx.fillStyle = "#8b949e";
                    ctx.fillText(`M: ${massStr}`, screenX + drawRadius + 6, screenY + 8);
                    ctx.fillText(`R: ${distStr}`, screenX + drawRadius + 6, screenY + 20);
                }
            }
            
            // Update Chart Indicator
            updateChartMarker(t_yr);
            
            // Highlight current log entries
            updateAccretionLogHighlight(t_yr);
        }

        // Draw radial concentric grids (distances)
        function drawGrid() {
            const centerX = canvas.width / 2 + offsetX;
            const centerY = canvas.height / 2 + offsetY;
            const scale = Math.min(canvas.width, canvas.height) / 25.0 * zoom;
            
            const radii = [1.0, 2.0, 4.0, 8.0, 12.0];
            ctx.lineWidth = 0.8;
            ctx.strokeStyle = "rgba(48, 54, 61, 0.15)";
            ctx.fillStyle = "rgba(139, 148, 158, 0.25)";
            ctx.font = "10px JetBrains Mono";

            radii.forEach(r => {
                const screenR = r * scale;
                ctx.beginPath();
                ctx.arc(centerX, centerY, screenR, 0, 2*Math.PI);
                ctx.stroke();
                
                // Labels for distances
                ctx.fillText(r + " AU", centerX + screenR + 4, centerY - 4);
            });
            
            // Center barycenter axis lines
            ctx.beginPath();
            ctx.moveTo(centerX - 10, centerY);
            ctx.lineTo(centerX + 10, centerY);
            ctx.moveTo(centerX, centerY - 10);
            ctx.lineTo(centerX, centerY + 10);
            ctx.strokeStyle = "rgba(88, 166, 255, 0.25)";
            ctx.stroke();
        }

        // Draw smooth orbit paths & circular arc trails for the top protoplanets
        function drawTrails(ds, centerX, centerY, scale, frameStates) {
            ds.top_bodies.forEach((bodyId, rank) => {
                const state = frameStates[bodyId];
                if (!state) return;
                
                const color = colors.trails[rank];
                
                // 1. Draw smooth Keplerian orbit path as a full concentric ring (faint)
                ctx.beginPath();
                ctx.arc(centerX, centerY, state.r * scale, 0, 2*Math.PI);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.0;
                ctx.globalAlpha = 0.12; // Faint background path
                ctx.stroke();
                ctx.globalAlpha = 1.0;
                
                // 2. Draw smooth circular arc trail behind the planet (eliminates polygonal shapes)
                const trailAngle = Math.PI / 4.5; // ~40 degree tail segment
                ctx.beginPath();
                ctx.arc(centerX, centerY, state.r * scale, state.theta - trailAngle, state.theta, false);
                
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.2;
                ctx.globalAlpha = 0.45; // Glowing trail
                ctx.stroke();
                ctx.globalAlpha = 1.0; // reset
            });
        }

        // Initialize ChartJS Mass chart
        function initializeChart() {
            const ds = datasets[currentPreset];
            
            if (massChart) {
                massChart.destroy();
            }
            
            const datasetsList = ds.top_bodies.map((bodyId, rank) => {
                return {
                    label: `Planet #${bodyId}`,
                    data: ds.mass_history[bodyId],
                    borderColor: colors.trails[rank],
                    backgroundColor: "transparent",
                    borderWidth: 2,
                    pointRadius: 0,
                    hoverRadius: 4,
                    tension: 0.1
                };
            });
            
            const ctxChart = document.getElementById("massChart").getContext("2d");
            massChart = new Chart(ctxChart, {
                type: 'line',
                data: {
                    labels: ds.time_years,
                    datasets: datasetsList
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'right',
                            labels: {
                                color: '#8b949e',
                                font: {
                                    family: 'Outfit',
                                    size: 11
                                }
                            }
                        },
                        title: {
                            display: true,
                            text: 'Top Protoplanet Mass Evolution (Earth Masses)',
                            color: '#c9d1d9',
                            font: {
                                family: 'Outfit',
                                size: 13,
                                weight: '600'
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Time (years)',
                                color: '#8b949e'
                            },
                            ticks: {
                                color: '#8b949e'
                            },
                            grid: {
                                color: 'rgba(48, 54, 61, 0.15)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Mass (M_earth)',
                                color: '#8b949e'
                            },
                            ticks: {
                                color: '#8b949e'
                            },
                            grid: {
                                color: 'rgba(48, 54, 61, 0.15)'
                            }
                        }
                    }
                },
                plugins: [{
                    // Custom plugin to draw vertical red cursor line
                    id: 'cursorLine',
                    afterDraw: (chart) => {
                        if (chart.tooltip?._active?.length) {}
                        
                        const xAxis = chart.scales.x;
                        const yAxis = chart.scales.y;
                        const xVal = ds.time_years[Math.floor(currentIdx)];
                        const xPixel = xAxis.getPixelForValue(xVal);
                        
                        const ctxLine = chart.ctx;
                        ctxLine.save();
                        ctxLine.beginPath();
                        ctxLine.moveTo(xPixel, yAxis.top);
                        ctxLine.lineTo(xPixel, yAxis.bottom);
                        ctxLine.strokeStyle = '#f85149';
                        ctxLine.lineWidth = 1.5;
                        ctxLine.setLineDash([4, 4]);
                        ctxLine.stroke();
                        ctxLine.restore();
                    }
                }]
            });
        }

        function updateChartMarker(t_yr) {
            if (massChart) {
                massChart.draw(); // forces redraw to update custom cursor line
            }
        }

        // Build Accretion Log List in Sidebar
        function buildAccretionLog() {
            const ds = datasets[currentPreset];
            const logContainer = document.getElementById("collision-log");
            logContainer.innerHTML = "";
            
            if (ds.collisions.length === 0) {
                logContainer.innerHTML = `<div style="color: #8b949e; text-align: center; padding-top: 20px;">No collisions logged.</div>`;
                return;
            }
            
            ds.collisions.forEach((c, idx) => {
                const entry = document.createElement("div");
                entry.className = "log-entry";
                entry.id = `log-entry-${idx}`;
                
                const isStar = (c.survivor_idx === 0);
                const survivorName = isStar ? "Central Star" : `Planet #${c.survivor_idx}`;
                
                if (isStar) {
                    entry.className = "log-entry swallowed";
                    entry.innerHTML = `<strong>yr ${c.time_years.toFixed(1)}</strong>: Star swallowed Planetesimal #${c.merged_idx} (mass: ${c.merged_mass_before.toFixed(2)} M⊕)`;
                } else {
                    entry.className = "log-entry grow";
                    entry.innerHTML = `<strong>yr ${c.time_years.toFixed(1)}</strong>: Planetesimal #${c.merged_idx} merged into Embryo #${c.survivor_idx}. New mass: <strong>${c.new_mass.toFixed(2)} M⊕</strong>`;
                }
                
                logContainer.appendChild(entry);
            });
        }

        // Scroll log to highlight currently occurring collisions
        function updateAccretionLogHighlight(t_yr) {
            const ds = datasets[currentPreset];
            ds.collisions.forEach((c, idx) => {
                const el = document.getElementById(`log-entry-${idx}`);
                if (!el) return;
                
                if (c.time_years <= t_yr) {
                    el.style.opacity = "1.0";
                    el.style.filter = "none";
                } else {
                    el.style.opacity = "0.3";
                    el.style.filter = "grayscale(80%)";
                }
            });
        }

        // Initialize on page load
        window.onload = () => {
            resizeCanvas();
            switchPreset("solar_system");
            togglePlay(); // Start playing automatically
        };
    </script>
</body>
</html>
""".replace("%DATA_PAYLOAD%", json.dumps(presets_data))

    # Write HTML dashboard
    html_path = os.path.join(docs_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html_content)
        
    print(f"🎉 Generated interactive dashboard HTML at: {html_path}")
    print(f"📂 Size: {os.path.getsize(html_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_dashboard()
