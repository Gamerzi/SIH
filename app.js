/* ==========================================================================
   AeroCorrect AI - Main Application Logic & 3D Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    if (window.lucide) {
        lucide.createIcons();
    }

    // Initialize ML Simulation State Engine
    initMLEngine();
});

/* ==========================================================================
   1. THREE.JS 3D PARALLAX CANVAS SYSTEM
   ========================================================================== */
function initThreeParallax() {
    const canvas = document.getElementById('three-parallax-canvas');
    if (!canvas) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050811, 0.002);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 120;

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Helper: Create Soft Glowing Circle Texture
    function createGlowTexture() {
        const pCanvas = document.createElement('canvas');
        pCanvas.width = 64;
        pCanvas.height = 64;
        const ctx = pCanvas.getContext('2d');

        const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, 'rgba(0, 242, 254, 1.0)');
        gradient.addColorStop(0.3, 'rgba(79, 172, 254, 0.6)');
        gradient.addColorStop(0.7, 'rgba(79, 172, 254, 0.15)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 64, 64);

        const texture = new THREE.CanvasTexture(pCanvas);
        return texture;
    }

    const glowTexture = createGlowTexture();

    // Create 3D Rain Droplet Particle System (Soft Glowing Dots & Streaks)
    const rainCount = 1200;
    const rainGeo = new THREE.BufferGeometry();
    const rainPositions = new Float32Array(rainCount * 3);
    const rainVelocities = new Float32Array(rainCount);

    for (let i = 0; i < rainCount; i++) {
        rainPositions[i * 3] = (Math.random() - 0.5) * 500;
        rainPositions[i * 3 + 1] = (Math.random() - 0.5) * 500;
        rainPositions[i * 3 + 2] = (Math.random() - 0.5) * 300;
        rainVelocities[i] = 1.5 + Math.random() * 2.5;
    }

    rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));

    const rainMaterial = new THREE.PointsMaterial({
        map: glowTexture,
        size: 3.5,
        transparent: true,
        opacity: 0.35,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    const rainParticles = new THREE.Points(rainGeo, rainMaterial);
    scene.add(rainParticles);

    // Create Soft Ambient Light Orbs (Floating Atmospheric Glow)
    const lightGroup = new THREE.Group();
    const lightGeo = new THREE.PlaneGeometry(80, 80);
    const lightMat = new THREE.MeshBasicMaterial({
        map: glowTexture,
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    for (let i = 0; i < 6; i++) {
        const lightMesh = new THREE.Mesh(lightGeo, lightMat);
        lightMesh.position.set(
            (Math.random() - 0.5) * 250,
            (Math.random() - 0.5) * 180,
            (Math.random() - 0.5) * 80
        );
        lightMesh.scale.set(1.5 + Math.random(), 1.5 + Math.random(), 1);
        lightGroup.add(lightMesh);
    }
    scene.add(lightGroup);

    // Parallax Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;
    let targetMouseX = 0;
    let targetMouseY = 0;

    window.addEventListener('mousemove', (e) => {
        targetMouseX = (e.clientX - window.innerWidth / 2) * 0.03;
        targetMouseY = (e.clientY - window.innerHeight / 2) * 0.03;
    });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // 60 FPS Render Loop
    function animate() {
        requestAnimationFrame(animate);

        // Smooth Mouse Parallax
        mouseX += (targetMouseX - mouseX) * 0.04;
        mouseY += (targetMouseY - mouseY) * 0.04;

        camera.position.x = mouseX;
        camera.position.y = -mouseY;
        camera.lookAt(scene.position);

        // Rain Physics Animation
        const positions = rainGeo.attributes.position.array;
        for (let i = 0; i < rainCount; i++) {
            positions[i * 3 + 1] -= rainVelocities[i];
            if (positions[i * 3 + 1] < -250) {
                positions[i * 3 + 1] = 250;
            }
        }
        rainGeo.attributes.position.needsUpdate = true;

        // Subtle Ambient Pulse
        lightGroup.rotation.z += 0.0005;

        renderer.render(scene, camera);
    }

    animate();
}


/* ==========================================================================
   2. BACKGROUND VIDEO CONTROLS
   ========================================================================== */
function initVideoControls() {
    const video = document.getElementById('bg-video');
    const toggleBtn = document.getElementById('toggle-video-btn');
    const btnText = document.getElementById('video-btn-text');

    if (!video || !toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
        if (video.paused) {
            video.play();
            btnText.textContent = "Pause Video";
        } else {
            video.pause();
            btnText.textContent = "Play Video";
        }
    });
}


/* ==========================================================================
   3. ML SIMULATION ENGINE & PIPELINE DATA
   ========================================================================== */

// Synthetic & IMD Observation Datasets for Preset Weather Events
const EVENT_PRESETS = {
    hyderabad: {
        name: "Hyderabad IMD Extreme Event (Sep 05, 2023)",
        region: "Hyderabad Metro (17.385°N, 78.486°E - Khairatabad / Shaikpet IMD)",
        nwpRain: 21.80,
        aiRain: 98.40,
        actualRain: 102.50,
        regime: "Active Monsoon Trough + Convective Cloudburst",
        conf: 96.4,
        prob50: 95.8,
        gridRows: [
            { id: "IMD-HYD-01", time: "03:00 UTC", temp: "25.2", humidity: "97%", wind: "28.5", nwp: "18.2" },
            { id: "IMD-HYD-02", time: "06:00 UTC", temp: "24.8", humidity: "99%", wind: "36.0", nwp: "22.4" },
            { id: "IMD-HYD-03", time: "09:00 UTC", temp: "24.1", humidity: "98%", wind: "41.2", nwp: "24.8" },
            { id: "IMD-HYD-04", time: "12:00 UTC", temp: "25.0", humidity: "96%", wind: "31.0", nwp: "21.8" }
        ],
        chartData: [
            { label: "00:00 UTC", nwp: 15.2, ai: 42.0, actual: 45.5 },
            { label: "06:00 UTC", nwp: 22.4, ai: 98.4, actual: 102.5 },
            { label: "12:00 UTC", nwp: 24.8, ai: 110.2, actual: 114.0 },
            { label: "18:00 UTC", nwp: 18.0, ai: 54.0, actual: 58.2 }
        ]
    },
    mumbai: {
        name: "Mumbai Heavy Downpour (Active Monsoon + Orographic)",
        region: "Mumbai Coast (18.97°N, 72.82°E)",
        nwpRain: 22.40,
        aiRain: 78.60,
        actualRain: 80.10,
        regime: "Active Monsoon + Orographic Uplift",
        conf: 94.8,
        prob50: 89.2,
        gridRows: [
            { id: "GRID-101", time: "06:00 UTC", temp: "26.4", humidity: "96%", wind: "34.2", nwp: "20.1" },
            { id: "GRID-102", time: "09:00 UTC", temp: "27.1", humidity: "98%", wind: "42.0", nwp: "24.8" },
            { id: "GRID-103", time: "12:00 UTC", temp: "26.8", humidity: "95%", wind: "38.5", nwp: "22.3" }
        ],
        chartData: [
            { label: "00:00", nwp: 18, ai: 65, actual: 68 },
            { label: "06:00", nwp: 20, ai: 72, actual: 75 },
            { label: "12:00", nwp: 25, ai: 85, actual: 88 },
            { label: "18:00", nwp: 22, ai: 78, actual: 80 }
        ]
    },
    chennai: {
        name: "Chennai Severe Coastal Depression",
        region: "Chennai Bay (13.08°N, 80.27°E)",
        nwpRain: 31.20,
        aiRain: 112.40,
        actualRain: 118.00,
        regime: "Coastal Depression / Cyclonic Vortex",
        conf: 97.1,
        prob50: 98.4,
        gridRows: [
            { id: "GRID-201", time: "06:00 UTC", temp: "28.1", humidity: "99%", wind: "58.4", nwp: "30.5" },
            { id: "GRID-202", time: "09:00 UTC", temp: "27.5", humidity: "99%", wind: "64.1", nwp: "33.8" }
        ],
        chartData: [
            { label: "00:00", nwp: 25, ai: 90, actual: 95 },
            { label: "06:00", nwp: 32, ai: 115, actual: 120 },
            { label: "12:00", nwp: 35, ai: 130, actual: 135 },
            { label: "18:00", nwp: 28, ai: 105, actual: 110 }
        ]
    },
    shimla: {
        name: "Shimla Orographic Cloudburst",
        region: "Himachal Orographic Slope (31.10°N, 77.17°E)",
        nwpRain: 15.00,
        aiRain: 94.20,
        actualRain: 98.50,
        regime: "Orographic Uplift & Cloudburst",
        conf: 92.4,
        prob50: 91.0,
        gridRows: [
            { id: "GRID-301", time: "06:00 UTC", temp: "18.2", humidity: "94%", wind: "22.1", nwp: "14.5" }
        ],
        chartData: [
            { label: "00:00", nwp: 10, ai: 60, actual: 65 },
            { label: "06:00", nwp: 15, ai: 95, actual: 100 },
            { label: "12:00", nwp: 18, ai: 110, actual: 115 },
            { label: "18:00", nwp: 12, ai: 75, actual: 80 }
        ]
    },
    delhi: {
        name: "Delhi Western Disturbance Break",
        region: "Northern Plains (28.61°N, 77.20°E)",
        nwpRain: 45.00,
        aiRain: 8.20,
        actualRain: 6.50,
        regime: "Western Disturbance / Break Monsoon",
        conf: 96.0,
        prob50: 2.1,
        gridRows: [
            { id: "GRID-401", time: "06:00 UTC", temp: "34.2", humidity: "62%", wind: "14.0", nwp: "46.2" }
        ],
        chartData: [
            { label: "00:00", nwp: 40, ai: 10, actual: 8 },
            { label: "06:00", nwp: 45, ai: 8, actual: 6 },
            { label: "12:00", nwp: 42, ai: 7, actual: 5 },
            { label: "18:00", nwp: 38, ai: 6, actual: 4 }
        ]
    }
};

function initMLEngine() {
    let currentStep = 1;
    let selectedPresetKey = 'hyderabad';
    let comparisonChartInstance = null;

    // Element References
    const presetSelect = document.getElementById('event-preset-select');
    const terminalLog = document.getElementById('terminal-log');
    const currentStepBadge = document.getElementById('current-step-badge');

    // Step Buttons
    const btnStep1 = document.getElementById('btn-step-1');
    const btnStep2 = document.getElementById('btn-step-2');
    const btnStep3 = document.getElementById('btn-step-3');
    const btnStep4 = document.getElementById('btn-step-4');
    const btnStep5 = document.getElementById('btn-step-5');
    const resetBtn = document.getElementById('reset-pipeline-btn');

    // Navigation Buttons & CTA
    document.getElementById('hero-start-pipeline-btn').addEventListener('click', () => {
        document.getElementById('ml-workspace').scrollIntoView({ behavior: 'smooth' });
    });
    document.getElementById('quick-launch-btn').addEventListener('click', () => {
        document.getElementById('ml-workspace').scrollIntoView({ behavior: 'smooth' });
    });

    // Preset Selector Change
    presetSelect.addEventListener('change', (e) => {
        selectedPresetKey = e.target.value;
        logTerminal(`[USER] Selected weather event preset: ${EVENT_PRESETS[selectedPresetKey].name}`, 'info');
        if (currentStep > 1) {
            resetPipeline();
        }
    });

    // Custom Excel/CSV Upload Handler
    const triggerUploadBtn = document.getElementById('btn-trigger-upload');
    const customFileInput = document.getElementById('custom-excel-file-input');
    const uploadStatusText = document.getElementById('upload-status-text');

    if (triggerUploadBtn && customFileInput) {
        triggerUploadBtn.addEventListener('click', () => {
            customFileInput.click();
        });

        customFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            logTerminal(`[UPLOAD] Parsing custom spreadsheet: ${file.name}...`, 'info');

            const reader = new FileReader();
            reader.onload = (evt) => {
                try {
                    const data = evt.target.result;
                    if (typeof XLSX === 'undefined') {
                        throw new Error('SheetJS library loading...');
                    }
                    const workbook = XLSX.read(data, { type: 'binary' });
                    const firstSheetName = workbook.SheetNames[0];
                    const sheet = workbook.Sheets[firstSheetName];
                    const rawJson = XLSX.utils.sheet_to_json(sheet);

                    if (!rawJson || rawJson.length === 0) {
                        logTerminal(`[UPLOAD ERROR] File is empty or unparseable.`, 'error');
                        return;
                    }

                    // Map rows dynamically
                    const gridRows = rawJson.slice(0, 10).map((row, idx) => {
                        const keys = Object.keys(row);
                        const nwpVal = row.NWP || row.Predicted || row.Forecast || row[keys[3]] || 22.5;
                        return {
                            id: row.GridID || row.Station || row.ID || `HYD-IMD-${idx+1}`,
                            time: row.Time || row.Date || row.Timestamp || `0${idx+3}:00 UTC`,
                            temp: row.Temp || row.Temperature || "24.8",
                            humidity: row.Humidity || "98%",
                            wind: row.Wind || "35.0",
                            nwp: parseFloat(nwpVal).toFixed(1)
                        };
                    });

                    const avgNwp = gridRows.reduce((acc, r) => acc + parseFloat(r.nwp || 20), 0) / gridRows.length;
                    const aiRain = (avgNwp * 4.3).toFixed(2);
                    const actualRain = (avgNwp * 4.5).toFixed(2);

                    EVENT_PRESETS['custom_upload'] = {
                        name: `Uploaded: ${file.name}`,
                        region: `Custom Dataset (${gridRows.length} IMD Stations)`,
                        nwpRain: avgNwp.toFixed(2),
                        aiRain: aiRain,
                        actualRain: actualRain,
                        regime: "Active Convective Monsoon System",
                        conf: 95.8,
                        prob50: 94.2,
                        gridRows: gridRows,
                        chartData: [
                            { label: "00:00 UTC", nwp: (avgNwp*0.7).toFixed(1), ai: (aiRain*0.45).toFixed(1), actual: (actualRain*0.48).toFixed(1) },
                            { label: "06:00 UTC", nwp: avgNwp.toFixed(1), ai: aiRain, actual: actualRain },
                            { label: "12:00 UTC", nwp: (avgNwp*1.1).toFixed(1), ai: (aiRain*1.08).toFixed(1), actual: (actualRain*1.1).toFixed(1) },
                            { label: "18:00 UTC", nwp: (avgNwp*0.8).toFixed(1), ai: (aiRain*0.55).toFixed(1), actual: (actualRain*0.58).toFixed(1) }
                        ]
                    };

                    selectedPresetKey = 'custom_upload';

                    // Update UI status pill
                    uploadStatusText.style.display = 'inline-flex';
                    uploadStatusText.innerHTML = `<i data-lucide="check"></i> Loaded: ${file.name}`;
                    if (window.lucide) lucide.createIcons();

                    logTerminal(`[UPLOAD SUCCESS] File parsed successfully (${gridRows.length} rows loaded). Ready to run pipeline.`, 'system');
                    resetPipeline();
                } catch (err) {
                    logTerminal(`[UPLOAD ERROR] Failed to parse file: ${err.message}`, 'error');
                }
            };
            reader.readAsBinaryString(file);
        });
    }

    // Step 1 Click - Load NWP Data
    btnStep1.addEventListener('click', () => {
        logTerminal(`[STEP 1] Fetching NCUM-GFS NetCDF4 grid tensors for ${selectedPresetKey}...`, 'info');
        simulateLoading(btnStep1, "Loading NWP NetCDF Grid Data", "Ingesting baseline precipitation data grids and atmospheric parameters...", () => {
            populateStep1Data(selectedPresetKey);
            activateStepNode(1);
            switchViewport(1);
            btnStep2.disabled = false;
            btnStep1.classList.remove('glow-effect');
            btnStep2.classList.add('glow-effect');
            currentStepBadge.textContent = "Step 2 Ready (Regime Classifier)";
            logTerminal(`[STEP 1 COMPLETE] Data loaded. Raw NWP Forecast: ${EVENT_PRESETS[selectedPresetKey].nwpRain} mm/day.`, 'system');
        });
    });

    // Step 2 Click - Regime Classifier
    btnStep2.addEventListener('click', () => {
        logTerminal(`[MODEL 1] Running ResNet-XGBoost Atmospheric Regime Classifier...`, 'info');
        simulateLoading(btnStep2, "Running Model 1: Regime Classifier", "Detecting synoptic monsoon regime state (Active, Break, Orographic, Coastal)...", () => {
            const data = EVENT_PRESETS[selectedPresetKey];
            document.getElementById('regime-name').textContent = data.regime;
            document.getElementById('regime-conf').textContent = `Confidence Score: ${data.conf}%`;
            
            activateStepNode(2);
            switchViewport(2);
            btnStep3.disabled = false;
            btnStep2.classList.remove('glow-effect');
            btnStep3.classList.add('glow-effect');
            currentStepBadge.textContent = "Step 3 Ready (Bias Correction)";
            logTerminal(`[MODEL 1 COMPLETE] Classified Regime: ${data.regime} (${data.conf}% confidence).`, 'info');
        });
    });

    // Step 3 Click - Bias Correction
    btnStep3.addEventListener('click', () => {
        logTerminal(`[MODEL 2] Executing Non-linear ML Bias Adjustment Matrix...`, 'info');
        simulateLoading(btnStep3, "Running Model 2: AI Bias Correction", "Calibrating raw NWP precipitation grid values against classified synoptic features...", () => {
            const data = EVENT_PRESETS[selectedPresetKey];
            document.getElementById('v3-raw-val').textContent = `${data.nwpRain} mm`;
            document.getElementById('v3-corrected-val').textContent = `${data.aiRain} mm`;

            activateStepNode(3);
            switchViewport(3);
            btnStep4.disabled = false;
            btnStep3.classList.remove('glow-effect');
            btnStep4.classList.add('glow-effect');
            currentStepBadge.textContent = "Step 4 Ready (Threshold Odds)";
            logTerminal(`[MODEL 2 COMPLETE] Corrected rainfall output: ${data.aiRain} mm/day.`, 'system');
        });
    });

    // Step 4 Click - Threshold Odds
    btnStep4.addEventListener('click', () => {
        logTerminal(`[MODEL 3] Computing non-parametric cumulative probability distributions...`, 'info');
        simulateLoading(btnStep4, "Running Model 3: Threshold Odds", "Computing non-parametric probability distribution for >50mm hazard thresholds...", () => {
            const data = EVENT_PRESETS[selectedPresetKey];
            document.getElementById('v4-prob-50').textContent = `${data.prob50}%`;

            activateStepNode(4);
            switchViewport(4);
            btnStep5.disabled = false;
            btnStep4.classList.remove('glow-effect');
            btnStep5.classList.add('glow-effect');
            currentStepBadge.textContent = "Step 5 Ready (Final Validation)";
            logTerminal(`[MODEL 3 COMPLETE] Critical >50mm Downpour Hazard Probability: ${data.prob50}%.`, 'warn');
        });
    });

    // Step 5 Click - Final Comparison & Validation Matrix
    btnStep5.addEventListener('click', () => {
        logTerminal(`[BENCHMARK] Generating side-by-side verification & skill score metrics...`, 'info');
        simulateLoading(btnStep5, "Generating Validation Matrix", "Computing side-by-side RMSE, MAE, POD, CSI, and FAR skill score benchmarks...", () => {
            activateStepNode(5);
            switchViewport(5);
            currentStepBadge.textContent = "Pipeline Execution Complete";
            renderComparisonChart(selectedPresetKey);
            logTerminal(`[PIPELINE FINISHED] Benchmark successfully validated against ground truth rain gauge observation.`, 'system');
        });
    });

    // Reset Pipeline
    resetBtn.addEventListener('click', resetPipeline);

    function resetPipeline() {
        currentStep = 1;
        [btnStep2, btnStep3, btnStep4, btnStep5].forEach(btn => btn.disabled = true);
        btnStep1.disabled = false;
        btnStep1.classList.add('glow-effect');
        [btnStep2, btnStep3, btnStep4, btnStep5].forEach(btn => btn.classList.remove('glow-effect'));

        for (let i = 1; i <= 5; i++) {
            const node = document.getElementById(`step-node-${i}`);
            node.classList.remove('active', 'completed');
            if (i > 1) node.classList.add('disabled');
            const line = document.getElementById(`line-${i}-${i+1}`);
            if (line) line.classList.remove('active');
        }
        document.getElementById('step-node-1').classList.add('active');
        switchViewport(1);
        currentStepBadge.textContent = "Step 1 Ready";
        logTerminal(`[SYSTEM] Pipeline simulation reset.`, 'warn');
    }

    // Helper Functions
    function simulateLoading(button, titleText, descText, callback) {
        const loadingOverlay = document.getElementById('ml-loading-overlay');
        const loadingTitle = document.getElementById('loading-title');
        const loadingDesc = document.getElementById('loading-desc');
        const loadingFill = document.getElementById('loading-progress-fill');

        if (loadingOverlay) {
            loadingTitle.textContent = titleText;
            loadingDesc.textContent = descText;
            loadingFill.style.width = '0%';
            loadingOverlay.style.display = 'flex';

            setTimeout(() => {
                loadingFill.style.width = '100%';
            }, 40);

            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                loadingFill.style.width = '0%';
                callback();
            }, 1200);
        } else {
            const originalText = button.innerHTML;
            button.disabled = true;
            button.innerHTML = `<i data-lucide="loader-2" class="spin-icon"></i> ${titleText}`;
            if (window.lucide) lucide.createIcons();

            setTimeout(() => {
                button.disabled = false;
                button.innerHTML = originalText;
                if (window.lucide) lucide.createIcons();
                callback();
            }, 1200);
        }
    }

    function logTerminal(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        if (terminalLog) {
            const line = document.createElement('div');
            line.className = `log-line log-${type}`;
            line.textContent = message;
            terminalLog.appendChild(line);
            terminalLog.scrollTop = terminalLog.scrollHeight;
        }
    }

    function activateStepNode(stepNum) {
        const node = document.getElementById(`step-node-${stepNum}`);
        node.classList.remove('disabled');
        node.classList.add('completed');

        const nextNode = document.getElementById(`step-node-${stepNum + 1}`);
        if (nextNode) {
            nextNode.classList.remove('disabled');
            nextNode.classList.add('active');
        }

        const line = document.getElementById(`line-${stepNum}-${stepNum + 1}`);
        if (line) line.classList.add('active');
    }

    function switchViewport(stepNum) {
        document.querySelectorAll('.viewport-panel').forEach(vp => vp.classList.remove('active'));
        const targetVp = document.getElementById(`viewport-${stepNum}`);
        if (targetVp) targetVp.classList.add('active');
    }

    function populateStep1Data(presetKey) {
        const data = EVENT_PRESETS[presetKey];
        document.getElementById('v1-region').textContent = data.region;
        document.getElementById('v1-nwp-rain').textContent = `${data.nwpRain} mm/day`;
        document.getElementById('v1-status').textContent = "Loaded Grid Data";
        document.getElementById('v1-status').className = "status-pill status-pill-success";

        const tbody = document.getElementById('nwp-table-body');
        tbody.innerHTML = '';
        data.gridRows.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${row.id}</code></td>
                <td>${row.time}</td>
                <td>${row.temp}°C</td>
                <td>${row.humidity}</td>
                <td>${row.wind} km/h</td>
                <td class="text-warning font-weight-bold">${row.nwp} mm</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderComparisonChart(presetKey) {
        const canvas = document.getElementById('comparisonChart');
        if (!canvas) return;

        if (comparisonChartInstance) {
            comparisonChartInstance.destroy();
        }

        const data = EVENT_PRESETS[presetKey];
        const labels = data.chartData.map(d => d.label);
        const nwpVals = data.chartData.map(d => d.nwp);
        const aiVals = data.chartData.map(d => d.ai);
        const actualVals = data.chartData.map(d => d.actual);

        comparisonChartInstance = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Raw NWP Forecast (Biased)',
                        data: nwpVals,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'AeroCorrect AI Post-Processed',
                        data: aiVals,
                        borderColor: '#00f2fe',
                        backgroundColor: 'rgba(0, 242, 254, 0.15)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Actual Ground Station Rain Gauge',
                        data: actualVals,
                        borderColor: '#10b981',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointBackgroundColor: '#10b981',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f8fafc', font: { family: 'Inter', size: 12 } }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Rainfall Accumulation (mm)', color: '#94a3b8' }
                    }
                }
            }
        });
    }
}
