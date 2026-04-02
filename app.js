// ====================================================================
// FedSTGNN – Map-based Interactive Visualization (v3)
// ====================================================================
(function () {
    'use strict';

    const C = {
        srv: ['#3B82F6', '#8B5CF6', '#EC4899', '#06B6D4'],
        srvBg: ['rgba(59,130,246,0.18)', 'rgba(139,92,246,0.18)', 'rgba(236,72,153,0.18)', 'rgba(6,182,212,0.18)'],
        srvName: ['서버 A', '서버 B', '서버 C', '서버 D'],
        green: '#10B981', central: '#4F46E5',
        trafficFree: '#10B981', trafficMod: '#F59E0B', trafficJam: '#EF4444',
    };

    // ===== Sensors =====
    const SENSORS = [
        { id: 0, lat: 34.020, lng: -118.490, cluster: 0 },
        { id: 1, lat: 34.045, lng: -118.475, cluster: 0 },
        { id: 2, lat: 34.015, lng: -118.450, cluster: 0 },
        { id: 3, lat: 34.050, lng: -118.435, cluster: 0 },
        { id: 4, lat: 34.030, lng: -118.410, cluster: 0 },
        { id: 5, lat: 34.065, lng: -118.420, cluster: 0 },
        { id: 6, lat: 34.140, lng: -118.380, cluster: 1 },
        { id: 7, lat: 34.110, lng: -118.395, cluster: 1 },
        { id: 8, lat: 34.095, lng: -118.350, cluster: 1 },
        { id: 9, lat: 34.130, lng: -118.320, cluster: 1 },
        { id: 10, lat: 34.100, lng: -118.300, cluster: 1 },
        { id: 11, lat: 34.080, lng: -118.280, cluster: 1 },
        { id: 12, lat: 34.050, lng: -118.260, cluster: 2 },
        { id: 13, lat: 34.070, lng: -118.230, cluster: 2 },
        { id: 14, lat: 34.030, lng: -118.245, cluster: 2 },
        { id: 15, lat: 34.035, lng: -118.210, cluster: 2 },
        { id: 16, lat: 34.010, lng: -118.225, cluster: 2 },
        { id: 17, lat: 33.990, lng: -118.240, cluster: 2 },
        { id: 18, lat: 33.960, lng: -118.390, cluster: 3 },
        { id: 19, lat: 33.975, lng: -118.350, cluster: 3 },
        { id: 20, lat: 33.930, lng: -118.365, cluster: 3 },
        { id: 21, lat: 33.920, lng: -118.330, cluster: 3 },
        { id: 22, lat: 33.955, lng: -118.300, cluster: 3 },
        { id: 23, lat: 33.910, lng: -118.290, cluster: 3 },
    ];

    const EDGES = [
        [0,1], [0,2], [1,3], [2,4], [3,5], [4,5], [2,3],
        [6,7], [7,8], [6,9], [8,9], [8,10], [10,11], [9,10],
        [12,13], [12,14], [13,15], [14,16], [15,16], [16,17], [13,14],
        [18,19], [18,20], [19,22], [20,21], [21,22], [21,23], [19,20],
        [5,7], [3,8], [11,12], [10,13], [17,22], [14,19], [18,2], [19,4]
    ];

    const CENTRAL_POS = { lat: 34.045, lng: -118.355 };
    // GNN mode: Server A (cluster 0) is the focal server.
    // Server A KNOWS its own sensors (cluster 0), but does NOT know other clusters' sensors.
    const FOCAL_CLUSTER = 0; // Server A
    const UNKNOWN_CLUSTERS = [1, 2, 3]; // B, C, D are unknown to A

    // ===== Adjacency =====
    const ADJ = Array.from({ length: 24 }, () => []);
    EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });

    function clusterCenter(ci) {
        const ns = SENSORS.filter(s => s.cluster === ci);
        return { lat: ns.reduce((a, n) => a + n.lat, 0) / ns.length, lng: ns.reduce((a, n) => a + n.lng, 0) / ns.length };
    }

    // ===== Traffic =====
    function getSimHour(t) { return (t * 0.0003) % 24; }
    function trafficSpeed(hour, seed) {
        let base = 56; // default free
        const isCommute = (seed % 4 === 1);
        const isDowntown = (seed % 5 === 2);

        if (isCommute) {
            if (hour >= 7 && hour <= 9.5) base = 25; // Jam
            else if (hour >= 17 && hour <= 19.5) base = 18; // Heavy jam
            else if (hour > 9 && hour < 17) base = 42; // Moderate
        } else if (isDowntown) {
            if (hour >= 11 && hour <= 20) base = 35; // Congested all day
        } else {
            if (hour > 8 && hour < 21) base = 50; 
        }
        return base + Math.sin(seed * 7.13 + hour * 1.5) * 5;
    }
    function trafficColor(sp) { return sp > 45 ? C.trafficFree : sp > 28 ? C.trafficMod : C.trafficJam; }
    function trafficCondKR(sp) { return sp > 45 ? { t: '원활', c: 'free' } : sp > 28 ? { t: '서행', c: 'moderate' } : { t: '정체', c: 'congested' }; }

    const TRAFFIC_ACTUAL = [];
    for (let i = 0; i < 48; i++) TRAFFIC_ACTUAL.push(trafficSpeed(i * 0.5, 42));
    function makePredicted(m) {
        const n = { zero: 16, neighbor: 8, propagation: 3 }[m] || 3;
        return TRAFFIC_ACTUAL.map((v, i) => Math.max(5, v + (Math.sin(i * 3.7 + m.length) * 0.5 - 0.25) * n * 2));
    }

    // ===== Loss curves =====
    function makeLoss(m) {
        const c = { zero: { d: 14, f: 0.34, n: 0.035 }, neighbor: { d: 22, f: 0.19, n: 0.02 }, propagation: { d: 32, f: 0.07, n: 0.012 } }[m];
        const r = [];
        for (let i = 0; i < 100; i++) r.push(0.88 * Math.exp(-i / c.d) + c.f + Math.sin(i * 1.3 + m.length) * 0.5 * c.n);
        return r;
    }
    const LOSS = { zero: makeLoss('zero'), neighbor: makeLoss('neighbor'), propagation: makeLoss('propagation') };

    // ===== Edge particles =====
    const edgeParticles = [];
    EDGES.forEach((_, i) => { for (let j = 0; j < 3; j++) edgeParticles.push({ edge: i, pos: Math.random(), dir: j % 2 === 0 ? 1 : -1 }); });

    // ===== State =====
    const state = { mode: 'overview', flPlaying: false, flRound: 1, flStep: 0, flProgress: 0, flSpeed: 3, imputeMethod: 'zero', waves: [], lossFrame: 0, online: [true, true, true, true], gnnAnimating: false, gnnAnimStart: 0 };

    // ===== Map & Canvas =====
    let map, canvas, ctx;
    function initMap() {
        map = L.map('map', { center: [34.025, -118.35], zoom: 11, zoomControl: false, attributionControl: true });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { attribution: '&copy; OSM &copy; CARTO', maxZoom: 18 }).addTo(map);
        L.control.zoom({ position: 'bottomright' }).addTo(map);
        map.on('click', e => { if (state.mode === 'gnn') { const ci = findClosest(e.latlng); if (ci >= 0) state.waves.push({ source: ci, t: 0 }); } });
    }
    function findClosest(ll) { let b = -1, bd = Infinity; SENSORS.forEach((s, i) => { const d = map.distance(ll, L.latLng(s.lat, s.lng)); if (d < bd) { b = i; bd = d; } }); return bd < 2500 ? b : -1; }
    function initCanvas() { canvas = document.getElementById('overlay'); ctx = canvas.getContext('2d'); resize(); window.addEventListener('resize', resize); }
    function resize() { const d = window.devicePixelRatio || 1; canvas.width = innerWidth * d; canvas.height = innerHeight * d; canvas.style.width = innerWidth + 'px'; canvas.style.height = innerHeight + 'px'; ctx.setTransform(d, 0, 0, d, 0, 0); }
    function px(s) { const p = map.latLngToContainerPoint([s.lat, s.lng]); return { x: p.x, y: p.y }; }
    function pxLL(lat, lng) { const p = map.latLngToContainerPoint([lat, lng]); return { x: p.x, y: p.y }; }
    function lerp(a, b, t) { return a + (b - a) * t; }
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function hexA(c, a) { return c + Math.round(clamp(a, 0, 1) * 255).toString(16).padStart(2, '0'); }

    function nearestOnlineCluster(sensor) {
        let best = -1, bd = Infinity;
        for (let ci = 0; ci < 4; ci++) { if (!state.online[ci] || ci === sensor.cluster) continue; const cc = clusterCenter(ci); const d = Math.hypot(sensor.lat - cc.lat, sensor.lng - cc.lng); if (d < bd) { best = ci; bd = d; } }
        return best;
    }

    // ===== MAIN DRAW LOOP =====
    function draw(time) {
        ctx.clearRect(0, 0, innerWidth, innerHeight);
        const hour = getSimHour(time);

        drawEdges(time, hour);
        drawTrafficParticles(time, hour); // overview/resilience only: colored circles = cars
        drawNodes(time, hour);
        drawServers(time);

        if (state.mode === 'fl') drawFL(time);       // ◆ diamonds = model parameter packets
        if (state.mode === 'gnn') drawGNN(time);     // pulse rings = GNN message passing
        if (state.mode === 'resilience') drawResilience(time); // dashed lines = rerouting

        if (state.flPlaying && state.mode === 'fl') {
            state.flProgress += 0.004 * state.flSpeed;
            if (state.flProgress >= 1) { state.flProgress = 0; state.flStep = (state.flStep + 1) % 4; if (state.flStep === 0) state.flRound = Math.min(100, state.flRound + 1); updateFLPanel(); }
        }
        // GNN diffusion phase advances
        if (state.mode === 'gnn' && state.lossFrame < 99) state.lossFrame += 0.2;
        updateTimeDisplay(hour);
        requestAnimationFrame(draw);
    }

    // ===== Draw edges =====
    function drawEdges(time, hour) {
        if (state.mode === 'gnn' && state.gnnAnimating) return; // Handled dynamically in drawGNNAnimation
        EDGES.forEach(([a, b], ei) => {
            const pa = px(SENSORS[a]), pb = px(SENSORS[b]);
            let color = 'rgba(100,116,139,0.2)', lw = 1.5;
            if (state.mode === 'overview' || state.mode === 'resilience') {
                // 실선 트래픽 표시 (장애 모드에서도 동일하게 실선 트래픽 표시 유지)
                color = trafficColor(trafficSpeed(hour, ei));
                lw = 2.5; 
            }
            ctx.strokeStyle = color; ctx.lineWidth = lw;
            ctx.globalAlpha = (state.mode === 'overview' || state.mode === 'resilience') ? 0.7 : 0.4;
            ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
            ctx.globalAlpha = 1;
        });
    }

    // ===== Traffic particles =====
    function drawTrafficParticles(time, hour) {
        if (state.mode !== 'overview' && state.mode !== 'resilience') return;
        edgeParticles.forEach(p => {
            p.pos += 0.002 * (trafficSpeed(hour, p.edge) / 60) * p.dir;
            if (p.pos > 1) p.pos = 0; if (p.pos < 0) p.pos = 1;
            const [a, b] = EDGES[p.edge];
            if (state.mode === 'resilience' && !state.online[SENSORS[a].cluster] && !state.online[SENSORS[b].cluster]) return;
            const pa = px(SENSORS[a]), pb = px(SENSORS[b]);
            ctx.globalAlpha = 0.65; ctx.fillStyle = trafficColor(trafficSpeed(hour, p.edge));
            ctx.beginPath(); ctx.arc(lerp(pa.x, pb.x, p.pos), lerp(pa.y, pb.y, p.pos), 2.5, 0, Math.PI * 2); ctx.fill();
            ctx.globalAlpha = 1;
        });
    }

    // ===== Draw sensor nodes =====
    function drawNodes(time, hour) {
        if (state.mode === 'gnn' && state.gnnAnimating) return; // Handled dynamically in drawGNNAnimation

        SENSORS.forEach((s, i) => {
            const p = px(s), color = C.srv[s.cluster];

            if (state.mode === 'gnn') {
                const isFocal = s.cluster === FOCAL_CLUSTER; // Server A's own sensors

                if (isFocal) {
                    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 16);
                    g.addColorStop(0, hexA(C.srv[FOCAL_CLUSTER], 0.3)); g.addColorStop(1, hexA(C.srv[FOCAL_CLUSTER], 0));
                    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 16, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = C.srv[FOCAL_CLUSTER]; ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
                } else {
                    if (state.imputeMethod === 'zero') {
                        ctx.strokeStyle = '#D1D5DB'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
                        ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
                        ctx.fillStyle = '#C4C4C4'; ctx.font = 'bold 8px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                        ctx.fillText('0', p.x, p.y); ctx.textBaseline = 'alphabetic';
                    } else if (state.imputeMethod === 'neighbor') {
                        const nbrAlpha = ADJ[s.id].some(nb => SENSORS[nb].cluster === FOCAL_CLUSTER) ? 0.7 : 0.3;
                        ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
                        ctx.fillStyle = hexA(color, nbrAlpha);
                        ctx.beginPath(); ctx.arc(p.x, p.y, 6, -Math.PI / 2, Math.PI / 2); ctx.fill();
                        ctx.strokeStyle = hexA(color, nbrAlpha); ctx.lineWidth = 1.5;
                        ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.stroke();
                    } else {
                        const pulse = Math.sin(time * 0.003 + i * 0.7) * 0.3 + 0.7;
                        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
                        g.addColorStop(0, hexA(color, 0.25 * pulse)); g.addColorStop(1, hexA(color, 0));
                        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2); ctx.fill();
                        ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
                        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill();
                    }
                }
                return;
            }

            // All other modes: ALWAYS colored
            // Identify actual working cluster for sensor coloring
            let activeCluster = s.cluster;
            if (state.mode === 'resilience' && !state.online[s.cluster]) {
                const nearCi = nearestOnlineCluster(s);
                if (nearCi >= 0) activeCluster = nearCi;
            }

            const nodeColor = (state.mode === 'overview') ? trafficColor(trafficSpeed(hour, i)) : C.srv[activeCluster];
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
            g.addColorStop(0, hexA(nodeColor, 0.2)); g.addColorStop(1, hexA(nodeColor, 0));
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = nodeColor; ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill();
        });
    }

    // ===== Draw servers — ALWAYS visible on map =====
    function drawRoundRect(c, x, y, w, h, r) {
        c.beginPath(); c.moveTo(x+r, y); c.lineTo(x+w-r, y);
        c.quadraticCurveTo(x+w, y, x+w, y+r); c.lineTo(x+w, y+h-r);
        c.quadraticCurveTo(x+w, y+h, x+w-r, y+h); c.lineTo(x+r, y+h);
        c.quadraticCurveTo(x, y+h, x, y+h-r); c.lineTo(x, y+r);
        c.quadraticCurveTo(x, y, x+r, y); c.closePath();
    }

    function drawServers(time) {
        ctx.save();
        // 4 edge servers at cluster centers
        for (let ci = 0; ci < 4; ci++) {
            const cc = clusterCenter(ci), p = pxLL(cc.lat, cc.lng);
            const isOff = state.mode === 'resilience' && !state.online[ci];
            const col = C.srv[ci];
            const bgCol = isOff ? '#F3F4F6' : '#FFFFFF';
            const strokeCol = isOff ? '#D1D5DB' : col;
            
            const w = 28, h = 34, r = 6;
            const sx = p.x - w/2, sy = p.y - h/2 - 4;
            
            ctx.shadowColor = isOff ? 'transparent' : hexA(col, 0.3); 
            ctx.shadowBlur = 12; ctx.shadowOffsetY = 4;
            drawRoundRect(ctx, sx, sy, w, h, r);
            ctx.fillStyle = bgCol; ctx.fill();
            
            ctx.shadowColor = 'transparent';
            ctx.strokeStyle = strokeCol; ctx.lineWidth = isOff ? 1.5 : 2; ctx.stroke();

            // Server rack blades
            ctx.fillStyle = isOff ? '#E5E7EB' : hexA(col, 0.15);
            drawRoundRect(ctx, sx + 5, sy + 5, w - 10, 4, 1.5); ctx.fill();
            drawRoundRect(ctx, sx + 5, sy + 13, w - 10, 4, 1.5); ctx.fill();
            drawRoundRect(ctx, sx + 5, sy + 21, w - 10, 4, 1.5); ctx.fill();

            // Status LED
            if (!isOff) {
                const ledCol = (Math.sin(time * 0.005 + ci) > 0) ? '#10B981' : '#34D399';
                ctx.fillStyle = ledCol;
                ctx.beginPath(); ctx.arc(sx + w - 7, sy + 7, 1.5, 0, Math.PI*2); ctx.fill();
            } else {
                ctx.fillStyle = '#EF4444';
                ctx.beginPath(); ctx.arc(sx + w - 7, sy + 7, 1.5, 0, Math.PI*2); ctx.fill();
            }
            
            // Server Label Below
            ctx.fillStyle = isOff ? '#9CA3AF' : col; 
            ctx.font = 'bold 11px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(C.srvName[ci], p.x, sy + h + 6);
            
            // Offline X Overlay
            if (isOff) {
                ctx.strokeStyle = '#EF4444'; ctx.lineWidth = 2.5;
                ctx.beginPath(); ctx.moveTo(p.x - 8, p.y - 8); ctx.lineTo(p.x + 8, p.y + 8);
                ctx.moveTo(p.x + 8, p.y - 8); ctx.lineTo(p.x - 8, p.y + 8); ctx.stroke();
            }
        }

        // Central Server - Represents Cloud / Hub
        const cp = pxLL(CENTRAL_POS.lat, CENTRAL_POS.lng);
        const cw = 44, ch = 32, cr = 8;
        const cx = cp.x - cw/2, cy = cp.y - ch/2 - 4;
        
        ctx.shadowColor = hexA(C.central, 0.4); ctx.shadowBlur = 15; ctx.shadowOffsetY = 5;
        drawRoundRect(ctx, cx, cy, cw, ch, cr);
        ctx.fillStyle = '#FFFFFF'; ctx.fill();
        ctx.shadowColor = 'transparent';
        
        ctx.strokeStyle = C.central; ctx.lineWidth = 2.5; ctx.stroke();
        
        // Data center columns
        for (let i = 0; i < 3; i++) {
            const px = cx + 7 + i*11;
            drawRoundRect(ctx, px, cy + 6, 8, ch - 12, 2);
            ctx.fillStyle = hexA(C.central, 0.1); ctx.fill();
            // Blinking lights
            ctx.fillStyle = (Math.sin(time*0.008 + i) > 0.5) ? C.central : hexA(C.central, 0.3);
            ctx.beginPath(); ctx.arc(px + 4, cy + 10, 1.5, 0, Math.PI*2); ctx.fill();
            ctx.beginPath(); ctx.arc(px + 4, cy + 15, 1.2, 0, Math.PI*2); ctx.fill();
        }
        
        ctx.fillStyle = C.central; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText('중앙 서버', cp.x, cy + ch + 8);
        
        ctx.restore();
    }

    // ===== FL overlay =====
    // Step 1: local training (glow), Step 2: params → central (◆ diamonds)
    // Step 3: FedAvg aggregation, Step 4: global model → servers (◆ diamonds green)
    function drawFL(time) {
        const step = state.flStep, prog = state.flProgress;
        const cp = pxLL(CENTRAL_POS.lat, CENTRAL_POS.lng);

        // Solid channel lines
        for (let ci = 0; ci < 4; ci++) {
            const sp = pxLL(clusterCenter(ci).lat, clusterCenter(ci).lng);
            ctx.strokeStyle = hexA(C.srv[ci], 0.25); ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(sp.x, sp.y); ctx.lineTo(cp.x, cp.y); ctx.stroke();
        }

        for (let ci = 0; ci < 4; ci++) {
            const sp = pxLL(clusterCenter(ci).lat, clusterCenter(ci).lng);
            if (step === 0) {
                // Local training: pulsing glow around edge server
                const pulse = Math.sin(time * 0.006 + ci) * 0.5 + 0.5;
                const g = ctx.createRadialGradient(sp.x, sp.y, 0, sp.x, sp.y, 35);
                g.addColorStop(0, hexA(C.srv[ci], 0.20 * pulse)); g.addColorStop(1, hexA(C.srv[ci], 0));
                ctx.fillStyle = g; ctx.beginPath(); ctx.arc(sp.x, sp.y, 35, 0, Math.PI * 2); ctx.fill();
            } else if (step === 1) {
                // ◆ Model parameter packet → central server (ONCE)
                const p = prog * 1.5 - ci * 0.1;
                if (p >= 0 && p <= 1.1) drawDiamond(sp, cp, clamp(p, 0, 1), C.srv[ci]);
            } else if (step === 3) {
                // ◆ Global model → each edge server (ONCE)
                const p = prog * 1.5 - ci * 0.1;
                if (p >= 0 && p <= 1.1) drawDiamond(cp, sp, clamp(p, 0, 1), C.central);
            }
        }
        if (step === 2) {
            // FedAvg aggregation: diamonds orbit the central server and spiral in
            C.srv.forEach((col, ci) => {
                const p = clamp(prog * 1.5, 0, 1);
                const a = (ci / 4) * Math.PI * 2 + time * 0.004;
                const r = 24 * (1 - p);
                if (p < 1) {
                    drawDiamond({ x: cp.x + Math.cos(a - 0.1) * r, y: cp.y + Math.sin(a - 0.1) * r },
                                { x: cp.x + Math.cos(a) * r, y: cp.y + Math.sin(a) * r }, 1, col);
                } else if (p === 1 && ci === 0) {
                    const pulse = Math.sin(time * 0.01) * 0.5 + 0.5;
                    const g = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, 40);
                    g.addColorStop(0, hexA(C.central, 0.4 * pulse)); g.addColorStop(1, hexA(C.central, 0));
                    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cp.x, cp.y, 40, 0, Math.PI * 2); ctx.fill();
                }
            });
        }
    }

    // ◆ Diamond = model parameter packet (NOT a vehicle/traffic circle)
    function drawDiamond(from, to, prog, color) {
        const x = lerp(from.x, to.x, prog), y = lerp(from.y, to.y, prog);
        const s = 5; // half-size
        // Trail
        for (let t = 1; t <= 3; t++) {
            const tp = clamp(prog - t * 0.06, 0, 1);
            const tx = lerp(from.x, to.x, tp), ty = lerp(from.y, to.y, tp);
            ctx.fillStyle = hexA(color, 0.18 - t * 0.04);
            ctx.beginPath(); ctx.moveTo(tx, ty - (s-t)); ctx.lineTo(tx + (s-t), ty); ctx.lineTo(tx, ty + (s-t)); ctx.lineTo(tx - (s-t), ty); ctx.closePath(); ctx.fill();
        }
        // Diamond
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill();
        // Glow
        const g = ctx.createRadialGradient(x, y, 0, x, y, 10);
        g.addColorStop(0, hexA(color, 0.25)); g.addColorStop(1, hexA(color, 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill();
    }

    function drawGNN(time) {
        if (state.gnnAnimating) {
            drawGNNAnimation(time);
            return;
        }
        const method = state.imputeMethod;
        if (method === 'neighbor') drawNeighborImputation(time);
        else if (method === 'propagation') drawFeaturePropagation(time);
    }

    function drawNeighborImputation(time) {
        const anim = (Math.sin(time * 0.003) + 1) / 2;
        EDGES.forEach(([a, b]) => {
            const sa = SENSORS[a], sb = SENSORS[b];
            const aIsFocal = sa.cluster === FOCAL_CLUSTER;
            const bIsFocal = sb.cluster === FOCAL_CLUSTER;
            if (aIsFocal === bIsFocal) return; 
            const focalNode = aIsFocal ? sa : sb;
            const unknownNode = aIsFocal ? sb : sa;
            const fp = px(focalNode), up = px(unknownNode);
            ctx.strokeStyle = hexA(C.srv[FOCAL_CLUSTER], 0.5 + anim * 0.3);
            ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.moveTo(fp.x, fp.y); ctx.lineTo(up.x, up.y); ctx.stroke();
            const p = (time * 0.0014 + focalNode.id * 0.2) % 1;
            const px2 = lerp(fp.x, up.x, p), py2 = lerp(fp.y, up.y, p);
            ctx.fillStyle = hexA(C.srv[FOCAL_CLUSTER], Math.sin(p * Math.PI) * 0.8);
            ctx.beginPath(); ctx.arc(px2, py2, 3.5, 0, Math.PI * 2); ctx.fill();
        });
    }

    function drawFeaturePropagation(time) {
        const bfsDist = new Array(24).fill(-1);
        const bfsQ = SENSORS.filter(s => s.cluster === FOCAL_CLUSTER).map(s => s.id);
        bfsQ.forEach(id => (bfsDist[id] = 0));
        for (let h = 0; h < bfsQ.length; h++) {
            const u = bfsQ[h];
            for (const v of ADJ[u]) { if (bfsDist[v] < 0) { bfsDist[v] = bfsDist[u] + 1; bfsQ.push(v); } }
        }
        const maxDist = Math.max(...bfsDist.filter(d => d >= 0));
        const wavePeriod = 3000; 
        const wavePhase = (time % wavePeriod) / wavePeriod; 
        for (let d = 1; d <= maxDist; d++) {
            const ringPhase = d / maxDist;
            const t = (wavePhase - ringPhase + 1) % 1; 
            const alpha = t < 0.3 ? t / 0.3 * 0.4 : (1 - (t - 0.3) / 0.7) * 0.15;
            if (alpha < 0.01) continue;
            EDGES.forEach(([a, b]) => {
                const da = bfsDist[a], db = bfsDist[b];
                if (Math.min(da, db) !== d - 1 || Math.max(da, db) !== d) return;
                const pa2 = px(SENSORS[da === d - 1 ? a : b]);
                const pb2 = px(SENSORS[da === d - 1 ? b : a]);
                ctx.strokeStyle = hexA(C.srv[FOCAL_CLUSTER], alpha * 2);
                ctx.lineWidth = 2;
                ctx.beginPath(); ctx.moveTo(pa2.x, pa2.y); ctx.lineTo(pb2.x, pb2.y); ctx.stroke();
                const edgeP = ((time * 0.001 + a * 0.2) % 1);
                ctx.fillStyle = hexA(C.srv[FOCAL_CLUSTER], alpha * 3);
                ctx.beginPath(); ctx.arc(lerp(pa2.x, pb2.x, edgeP), lerp(pa2.y, pb2.y, edgeP), 3, 0, Math.PI * 2); ctx.fill();
            });
            SENSORS.forEach(s => {
                if (bfsDist[s.id] !== d) return;
                const sp = px(s);
                const gAlpha = alpha * 1.5;
                const g = ctx.createRadialGradient(sp.x, sp.y, 0, sp.x, sp.y, 18);
                g.addColorStop(0, hexA(C.srv[FOCAL_CLUSTER], gAlpha)); g.addColorStop(1, hexA(C.srv[FOCAL_CLUSTER], 0));
                ctx.fillStyle = g; ctx.beginPath(); ctx.arc(sp.x, sp.y, 18, 0, Math.PI * 2); ctx.fill();
            });
        }
    }

    // ===== GNN overlay (Map-to-HUD Animation) =====
    function drawGNNAnimation(time) {
        const m = state.imputeMethod;
        const cycle = 12000; // Slower animation cycle
        const elapsed = time - state.gnnAnimStart;
        if (elapsed > cycle) {
            state.gnnAnimating = false;
            renderMathPanel();
            return;
        }
        const t = (elapsed % cycle) / cycle;

        // HUD dimensions and location
        const panel = document.getElementById('math-panel');
        let hudX = 176, hudY = 320; 
        if (panel) {
            const rect = panel.getBoundingClientRect();
            hudX = rect.left + rect.width / 2;
            hudY = rect.top + 210; 
        }
        
        ctx.save();
        ctx.setTransform(1,0,0,1,0,0); // reset scale to avoid messing up device ratio mapping since px() returns window coords
        const dpr = devicePixelRatio || 1;
        ctx.scale(dpr, dpr);

        const matA_X = hudX - 85; // Adjusted for better spacing
        const matX_X = hudX - 20;
        const nnBox_X = hudX + 50;
        const h_X = nnBox_X + 65; // Position of H vector
        const matY_start = hudY - 70; // 24 nodes * 6px = 144px height entirely
        
        const p_flyOut = clamp((t - 0.05) / 0.15, 0, 1);
        const p_eqForm = clamp((t - 0.25) / 0.15, 0, 1);
        const p_insert = clamp((t - 0.45) / 0.15, 0, 1);
        const p_compute = clamp((t - 0.60) / 0.15, 0, 1);
        const p_shoot = clamp((t - 0.75) / 0.1, 0, 1);
        const p_flyBack = clamp((t - 0.85) / 0.15, 0, 1);

        // Smoother easing functions
        const easeOut = x => 1 - Math.pow(1 - x, 4);
        const easeInOut = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
        const easeIn = x => Math.pow(x, 4);

        const e_flyOut = easeOut(p_flyOut);
        const e_insert = easeInOut(p_insert);
        const e_flyBack = easeIn(p_flyBack);

        // Define expected logical colors
        const nColors = SENSORS.map(s => {
            if (s.cluster === FOCAL_CLUSTER) return C.srv[FOCAL_CLUSTER];
            if (m === 'zero') return '#9CA3AF';
            if (m === 'neighbor') return ADJ[s.id].some(nb => SENSORS[nb].cluster === FOCAL_CLUSTER) ? '#60A5FA' : '#D1D5DB';
            return C.srv[FOCAL_CLUSTER]; // propagation fills all
        });

        // 1. Calculate positions for each node
        const positions = SENSORS.map((s, i) => {
            const origin = px(s), targetX = matX_X, targetY = matY_start + i * 6;
            
            let cX = lerp(origin.x, targetX, e_flyOut);
            cX = lerp(cX, nnBox_X, e_insert);
            
            const e_shoot_curve = 1 - Math.pow(1 - p_shoot, 4);
            cX = lerp(cX, h_X, e_shoot_curve);
            cX = lerp(cX, origin.x, e_flyBack);

            let cY = lerp(origin.y, targetY, e_flyOut);
            cY = lerp(cY, origin.y, e_flyBack);
            
            return { x: cX, y: cY };
        });

        // 2. Draw Edges (Fade out progressively as they fly to Matrix)
        const edgeAlpha = Math.max(0, 1 - p_flyOut * 1.5);
        if (edgeAlpha > 0) {
            ctx.lineWidth = 1.5;
            EDGES.forEach(([a, b]) => {
                ctx.strokeStyle = `rgba(156,163,175,${edgeAlpha * 0.5})`;
                ctx.beginPath(); ctx.moveTo(positions[a].x, positions[a].y); ctx.lineTo(positions[b].x, positions[b].y); ctx.stroke();
            });
        }

        // 3. Draw Matrices (Forming equation)
        const matOp = p_eqForm * (1 - p_insert);
        if (matOp > 0) {
            const aOff = lerp(0, nnBox_X - matA_X - 5, e_insert);
            const aX = matA_X + aOff;
            ctx.globalAlpha = matOp;
            // Bracket A
            ctx.strokeStyle = '#6B7280'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(aX - 25, matY_start - 3); ctx.lineTo(aX - 28, matY_start - 3); ctx.lineTo(aX - 28, matY_start + 145); ctx.lineTo(aX - 25, matY_start + 145); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(aX + 25, matY_start - 3); ctx.lineTo(aX + 28, matY_start - 3); ctx.lineTo(aX + 28, matY_start + 145); ctx.lineTo(aX + 25, matY_start + 145); ctx.stroke();
            ctx.fillStyle = '#6B7280'; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center'; ctx.fillText('Ã', aX, matY_start - 12);
            ctx.font = 'bold 15px Inter'; ctx.fillText('×', aX + 37, hudY);
            
            // Matrix A dots
            for (let r = 0; r < 24; r++) { for (let c = 0; c < 24; c++) {
                const isEdge = ADJ[r].includes(c) || r === c;
                ctx.fillStyle = isEdge ? '#374151' : '#E5E7EB';
                ctx.fillRect((aX - 23 + c * 2)|0, (matY_start + r * 6)|0, 1.5, 1.5);
            }}
            ctx.globalAlpha = 1;
        }

        // Bracket for Matrix X
        const xOp = p_flyOut * (1 - p_insert);
        if (xOp > 0) {
            const xOff = lerp(0, nnBox_X - matX_X - 5, e_insert);
            const xx = matX_X + xOff;
            ctx.globalAlpha = xOp;
            ctx.strokeStyle = '#6B7280'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(xx - 8, matY_start - 3); ctx.lineTo(xx - 11, matY_start - 3); ctx.lineTo(xx - 11, matY_start + 145); ctx.lineTo(xx - 8, matY_start + 145); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(xx + 8, matY_start - 3); ctx.lineTo(xx + 11, matY_start - 3); ctx.lineTo(xx + 11, matY_start + 145); ctx.lineTo(xx + 8, matY_start + 145); ctx.stroke();
            ctx.fillStyle = '#6B7280'; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center'; ctx.fillText('X', xx, matY_start - 12);
            ctx.globalAlpha = 1;
        }

        // 4. Draw GNN Processing Box
        ctx.fillStyle = '#1E293B';
        drawRoundRect(ctx, nnBox_X - 35, hudY - 70, 70, 140, 8); ctx.fill();
        ctx.fillStyle = 'white'; ctx.font = 'bold 14px Inter'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
        ctx.fillText('GNN', nnBox_X, hudY - 14);
        ctx.fillText('Layer', nnBox_X, hudY + 14);

        if (p_compute > 0 && p_compute < 1) {
            const pulse = Math.sin(p_compute * Math.PI * 6) * 0.5 + 0.5;
            ctx.strokeStyle = hexA('#3B82F6', pulse); ctx.lineWidth = 4;
            drawRoundRect(ctx, nnBox_X - 37, hudY - 72, 74, 144, 10); ctx.stroke();
        }

        // 5. Draw Output Bracket H 
        if (p_shoot > 0 && p_flyBack < 1) {
            const hX = h_X;
            ctx.globalAlpha = p_shoot * (1 - p_flyBack);
            ctx.strokeStyle = '#10B981'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(hX - 8, matY_start - 3); ctx.lineTo(hX - 11, matY_start - 3); ctx.lineTo(hX - 11, matY_start + 145); ctx.lineTo(hX - 8, matY_start + 145); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(hX + 8, matY_start - 3); ctx.lineTo(hX + 11, matY_start - 3); ctx.lineTo(hX + 11, matY_start + 145); ctx.lineTo(hX + 8, matY_start + 145); ctx.stroke();
            ctx.fillStyle = '#10B981'; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center'; ctx.fillText('H', hX, matY_start - 12);
            ctx.globalAlpha = 1;
        }

        // 6. Draw Nodes 
        positions.forEach((p, i) => {
            let nodeAlpha = 1;
            if (p_insert > 0 && p_shoot === 0) nodeAlpha = 1 - e_insert;
            let col = nColors[i];
            
            if (p_shoot > 0) {
                nodeAlpha = Math.min(1, p_shoot * 2); // Fade in to avoid harsh overlap with box text
                col = p_flyBack > 0.95 ? C.srv[SENSORS[i].cluster] : '#10B981';
            }

            const scale = p_flyOut === 0 ? 1 : lerp(1, 0.35, e_flyOut);
            const finalScale = p_shoot > 0 ? lerp(0.35, 1.0, e_flyBack) : scale;

            ctx.globalAlpha = nodeAlpha;
            if (finalScale > 0.8) {
                const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
                g.addColorStop(0, hexA(col, 0.3)); g.addColorStop(1, hexA(col, 0));
                ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2); ctx.fill();
            }

            ctx.fillStyle = col;
            if (p_shoot === 0 && p_flyOut === 0 && m === 'zero' && SENSORS[i].cluster !== FOCAL_CLUSTER) {
                ctx.strokeStyle = '#D1D5DB'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
                ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
                ctx.fillStyle = '#C4C4C4'; ctx.font = 'bold 8px Inter'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
                ctx.fillText('0', p.x, p.y); ctx.textBaseline='alphabetic';
            } else if (p_shoot === 0 && p_flyOut === 0 && m === 'neighbor' && SENSORS[i].cluster !== FOCAL_CLUSTER) {
                const isNbr = ADJ[i].some(nb => SENSORS[nb].cluster === FOCAL_CLUSTER);
                ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = hexA(col, isNbr ? 0.7 : 0.3); ctx.beginPath(); ctx.arc(p.x, p.y, 6, -Math.PI/2, Math.PI/2); ctx.fill();
                ctx.strokeStyle = hexA(col, isNbr ? 0.7 : 0.3); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.stroke();
            } else {
                ctx.beginPath(); ctx.arc(p.x, p.y, 7 * finalScale, 0, Math.PI * 2); ctx.fill();
                if (finalScale > 0.3) { ctx.fillStyle='white'; ctx.beginPath(); ctx.arc(p.x, p.y, 4 * finalScale, 0, Math.PI*2); ctx.fill(); }
            }
            ctx.globalAlpha = 1;
        });
        
        // 7. Splash Effect & Predicted Traffic Lines
        let sAlpha = 0;
        if (p_flyBack > 0.95) {
            sAlpha = 1 - (p_flyBack - 0.95) / 0.05;
            SENSORS.forEach(s => {
                const p = px(s);
                ctx.strokeStyle = hexA('#10B981', sAlpha); ctx.lineWidth = 2.5;
                ctx.beginPath(); ctx.arc(p.x, p.y, 14 + (1-sAlpha)*16, 0, Math.PI*2); ctx.stroke();
            });
        }

        // Output lines: show predicted traffic mapped back onto physical edges
        let outTrafficAlpha = 0;
        if (p_flyBack > 0.0) outTrafficAlpha = p_flyBack;
        else if (t < 0.05) outTrafficAlpha = 1;
        else if (p_flyOut > 0 && p_eqForm === 0) outTrafficAlpha = 1 - p_flyOut;

        if (outTrafficAlpha > 0) {
            const simHour = getSimHour(time);
            ctx.lineWidth = 2.5;
            EDGES.forEach(([a, b], ei) => {
                const pa = px(SENSORS[a]), pb = px(SENSORS[b]);
                const edgeC = trafficColor(trafficSpeed(simHour, ei));
                ctx.strokeStyle = hexA(edgeC, outTrafficAlpha * 0.85);
                ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
                
                // Animated traffic glow particles on the prediction edges
                if (outTrafficAlpha > 0.2) {
                    const pg = ((time * 0.0006) + ei * 0.1) % 1;
                    ctx.fillStyle = hexA('white', outTrafficAlpha * 0.8);
                    ctx.beginPath(); ctx.arc(lerp(pa.x, pb.x, pg), lerp(pa.y, pb.y, pg), 2.0, 0, Math.PI * 2); ctx.fill();
                }
            });
        }

        ctx.restore();
    }

    // ===== Resilience overlay — rerouting arrows from sensors to nearest online server =====
    function drawResilience(time) {
        SENSORS.forEach(s => {
            if (state.online[s.cluster]) return;
            const nearCi = nearestOnlineCluster(s);
            if (nearCi < 0) return;
            const sp = px(s), cc = clusterCenter(nearCi), tp = pxLL(cc.lat, cc.lng);
            const col = C.srv[nearCi];

            // Animated arrow particle pointing to the new server
            const p = ((time * 0.0008 + s.id * 0.15) % 1);
            const pX = lerp(sp.x, tp.x, p);
            const pY = lerp(sp.y, tp.y, p);
            const angle = Math.atan2(tp.y - sp.y, tp.x - sp.x);

            ctx.fillStyle = hexA(col, (1 - Math.abs(p - 0.5) * 2) * 0.9);
            ctx.beginPath();
            ctx.moveTo(pX + Math.cos(angle) * 8, pY + Math.sin(angle) * 8);
            ctx.lineTo(pX + Math.cos(angle - 2.5) * 6, pY + Math.sin(angle - 2.5) * 6);
            ctx.lineTo(pX + Math.cos(angle + 2.5) * 6, pY + Math.sin(angle + 2.5) * 6);
            ctx.closePath();
            ctx.fill();
        });

        // Destination halos
        for (let ci = 0; ci < 4; ci++) {
            if (!state.online[ci]) continue;
            if (!SENSORS.some(s => !state.online[s.cluster] && nearestOnlineCluster(s) === ci)) continue;
            const p = pxLL(clusterCenter(ci).lat, clusterCenter(ci).lng);
            const pulse = Math.sin(time * 0.003) * 0.15 + 0.85;
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 40);
            g.addColorStop(0, hexA(C.srv[ci], 0.15 * pulse)); g.addColorStop(1, hexA(C.srv[ci], 0));
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 40, 0, Math.PI * 2); ctx.fill();
        }
    }

    // ===== Charts =====
    function drawTrafficChart() {
        const cvs = document.getElementById('traffic-canvas'); if (!cvs) return;
        const cw = cvs.parentElement.clientWidth - 24, ch = 90, dpr = devicePixelRatio || 1;
        cvs.width = cw * dpr; cvs.height = ch * dpr; cvs.style.width = cw + 'px'; cvs.style.height = ch + 'px';
        const c = cvs.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, cw, ch);
        const pred = makePredicted(state.mode === 'gnn' ? state.imputeMethod : 'propagation');
        c.strokeStyle = '#F3F4F6'; c.lineWidth = 1;
        [0.25, 0.5, 0.75].forEach(r => { c.beginPath(); c.moveTo(0, ch * r); c.lineTo(cw, ch * r); c.stroke(); });
        // Actual area
        c.beginPath();
        TRAFFIC_ACTUAL.forEach((v, i) => { const x = (i / 47) * cw, y = ch - (v / 70) * ch; i === 0 ? c.moveTo(x, y) : c.lineTo(x, y); });
        c.lineTo(cw, ch); c.lineTo(0, ch); c.closePath();
        const fg = c.createLinearGradient(0, 0, 0, ch); fg.addColorStop(0, 'rgba(59,130,246,0.15)'); fg.addColorStop(1, 'rgba(59,130,246,0)');
        c.fillStyle = fg; c.fill();
        c.strokeStyle = '#3B82F6'; c.lineWidth = 2; c.beginPath();
        TRAFFIC_ACTUAL.forEach((v, i) => { const x = (i / 47) * cw, y = ch - (v / 70) * ch; i === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }); c.stroke();
        // Predicted
        c.strokeStyle = '#EF4444'; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath();
        pred.forEach((v, i) => { const x = (i / 47) * cw, y = ch - (v / 70) * ch; i === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }); c.stroke(); c.setLineDash([]);
        c.fillStyle = '#9CA3AF'; c.font = '9px Inter'; c.textAlign = 'center';
        [0, 6, 12, 18, 24].forEach(h => c.fillText(h + 'h', (h / 24) * cw, ch - 2));
        c.textAlign = 'left'; c.fillText('mph', 2, 10);
    }

    function drawLossChart() {
        const cvs = document.getElementById('loss-canvas'); if (!cvs) return;
        const cw = cvs.parentElement.clientWidth - 24, ch = 100, dpr = devicePixelRatio || 1;
        cvs.width = cw * dpr; cvs.height = ch * dpr; cvs.style.width = cw + 'px'; cvs.style.height = ch + 'px';
        const c = cvs.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, cw, ch);
        const frame = Math.min(Math.floor(state.lossFrame), 99);
        const methods = ['zero', 'neighbor', 'propagation'], colors = ['#9CA3AF', '#F59E0B', '#10B981'];
        c.strokeStyle = '#F3F4F6'; c.lineWidth = 1;
        [0.25, 0.5, 0.75].forEach(r => { c.beginPath(); c.moveTo(0, ch * r); c.lineTo(cw, ch * r); c.stroke(); });
        methods.forEach((m, mi) => {
            const data = LOSS[m], isA = m === state.imputeMethod;
            c.strokeStyle = colors[mi]; c.lineWidth = isA ? 2.5 : 1.2; c.globalAlpha = isA ? 1 : 0.35;
            c.beginPath();
            for (let i = 0; i <= frame; i++) { const x = (i / 99) * cw, y = ch - data[i] * ch * 0.7 - ch * 0.05; i === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }
            c.stroke();
            if (isA && frame > 0) { const ex = (frame / 99) * cw, ey = ch - data[frame] * ch * 0.7 - ch * 0.05; c.globalAlpha = 1; c.fillStyle = colors[mi]; c.beginPath(); c.arc(ex, ey, 3.5, 0, Math.PI * 2); c.fill(); }
            c.globalAlpha = 1;
        });
        c.fillStyle = '#9CA3AF'; c.font = '9px Inter'; c.textAlign = 'left'; c.fillText('Loss', 2, 10);
        c.textAlign = 'right'; c.fillText('Epoch ' + frame, cw - 2, ch - 4);
    }

    function updateTimeDisplay(hour) {
        const el = document.getElementById('sim-time'); if (!el) return;
        el.textContent = String(Math.floor(hour)).padStart(2, '0') + ':' + String(Math.floor((hour % 1) * 60)).padStart(2, '0');
        const condEl = document.getElementById('traffic-cond');
        if (condEl) { const cd = trafficCondKR(trafficSpeed(hour, 20)); condEl.textContent = cd.t; condEl.className = 'traffic-condition ' + cd.c; }
    }

    setInterval(() => {
        if (state.mode === 'overview' || state.mode === 'gnn') drawTrafficChart();
        if (state.mode === 'gnn') drawLossChart();
    }, 200);

    // ===== Panel HTML =====
    function getOverviewPanel() {
        return `<h2>📍 LA 교통 센서 네트워크</h2>
        <p class="desc">METR-LA 데이터셋 기반 <strong>고속도로 교통 센서</strong> 24개의 실시간 교통 상태를 보여줍니다.</p>
        <div class="section-divider"></div>
        <div class="stat-row"><span class="stat-label">센서 수</span><span class="stat-value">24개</span></div>
        <div class="stat-row"><span class="stat-label">연결 수</span><span class="stat-value">${EDGES.length}개</span></div>
        <div class="stat-row"><span class="stat-label">엣지 서버</span><span class="stat-value">4대</span></div>
        <div class="stat-row"><span class="stat-label">시뮬레이션 시각</span><span class="time-badge"><span class="clock">🕐</span> <span id="sim-time">07:00</span></span></div>
        <div class="stat-row"><span class="stat-label">교통 상태</span><span class="traffic-condition free" id="traffic-cond">원활</span></div>
        <div class="section-divider"></div>
        <h3>교통량 (실제 vs 예측)</h3>
        <div class="chart-wrap"><canvas id="traffic-canvas" height="90"></canvas>
            <div class="curve-legend"><div class="curve-legend-item"><div class="curve-legend-line" style="background:#3B82F6"></div>실제</div><div class="curve-legend-item"><div class="curve-legend-line" style="background:#EF4444"></div>예측</div></div>
        </div>
        <div class="hint">💡 상단 모드 버튼으로 각 단계를 탐색해 보세요</div>`;
    }

    const FL_DESCS = [
        '각 엣지 서버가 자신의 센서 데이터로 로컬 STGNN 모델을 학습합니다. 원본 데이터는 외부로 전송되지 않습니다.',
        '학습된 모델 파라미터만 중앙 서버로 전송합니다. 전체 데이터의 약 7% 크기입니다.',
        '중앙 서버에서 모든 로컬 파라미터를 FedAvg로 통합합니다. θ = (1/n) Σθᵢ',
        '통합된 글로벌 모델을 모든 엣지 서버에 배포합니다.',
    ];

    function getFLPanel() {
        return `<h2>🔄 연합학습 과정</h2>
        <p class="desc">4개 엣지 서버가 로컬 학습 후 모델 파라미터만 교환하여 글로벌 모델을 만듭니다.</p>
        <div class="section-divider"></div>
        <div class="stat-row"><span class="stat-label">현재 라운드</span><span class="round-badge" id="fl-round-badge">1 / 100</span></div>
        <div class="step-track" id="step-track">
            <div class="step-dot-wrap active" data-s="0"><div class="step-dot active">1</div><span class="step-name">로컬학습</span></div><div class="step-line"></div>
            <div class="step-dot-wrap" data-s="1"><div class="step-dot">2</div><span class="step-name">전송</span></div><div class="step-line"></div>
            <div class="step-dot-wrap" data-s="2"><div class="step-dot">3</div><span class="step-name">통합</span></div><div class="step-line"></div>
            <div class="step-dot-wrap" data-s="3"><div class="step-dot">4</div><span class="step-name">배포</span></div>
        </div>
        <div id="fl-step-desc" class="desc" style="min-height:40px">${FL_DESCS[0]}</div>
        <div class="section-divider"></div>
        <div class="controls-row">
            <button class="ctrl-btn" id="fl-play" title="재생">▶</button>
            <button class="ctrl-btn" id="fl-step-btn" title="다음 단계">⏭</button>
            <button class="ctrl-btn" id="fl-reset" title="초기화">↺</button>
            <div class="speed-slider"><span>속도</span><input type="range" id="fl-speed" min="1" max="5" value="3"></div>
        </div>`;
    }

    const IMPUTE_DESCS = {
        zero: '<strong>서버 A</strong>는 B/C/D의 센서 값을 모릅니다. 결측값을 <strong>0으로 채워</strong> GNN을 학습합니다. 잘못된 정보가 학습을 방해합니다.',
        neighbor: '<strong>서버 A</strong>와 직접 연결된 B/C/D 센서(<strong>1-hop</strong>)의 값을 평균내어 결측치를 추정합니다. 가까운 이웃만 참조하여 정확도가 제한적입니다.',
        propagation: '<strong>X<sup>(k+1)</sup> = (1-α)·Ã·X<sup>(k)</sup> + α·X₀</strong> — A의 알려진 값을 그래프 전체로 반복 확산하여, 멀리 떨어진 센서도 점진적으로 채워갑니다.',
    };

    function getGNNPanel() {
        const m = state.imputeMethod;
        const mLabel = m === 'zero' ? 'Zero-fill' : m === 'neighbor' ? '이웃 평균' : '특징 전파';
        return `<h2>🧠 데이터 보간 & GNN 학습</h2>
        <p class="desc"><strong>서버 A</strong>는 자신의 센서(🔵) 데이터만 알고, 다른 서버 센서는 미지수입니다. 아래 방식으로 결측치를 채워 GNN을 학습합니다.</p>
        <div class="section-divider"></div>
        <div class="server-a-legend">
            <span style="color:${C.srv[0]}">●</span> 서버 A 센서 (알려진 데이터)
            &nbsp;|&nbsp;
            <span style="color:#9CA3AF">○</span> 다른 서버 센서 (결측 → 보간)
        </div>
        <h3 style="margin-top:10px">보간 방식 선택</h3>
        <div class="impute-group">
            <button class="impute-btn ${m==='zero'?'active':''}" data-imp="zero">Zero-fill</button>
            <button class="impute-btn ${m==='neighbor'?'active':''}" data-imp="neighbor">이웃 평균</button>
            <button class="impute-btn ${m==='propagation'?'active':''}" data-imp="propagation">특징 전파</button>
        </div>
        <p class="desc" id="impute-desc">${IMPUTE_DESCS[m]}</p>
        <div class="section-divider"></div>
        <h3>학습 손실 비교 (Loss)</h3>
        <div class="chart-wrap"><canvas id="loss-canvas" height="100"></canvas></div>
        <div class="curve-legend">
            <div class="curve-legend-item ${m==='zero'?'active':''}"><div class="curve-legend-line" style="background:#9CA3AF"></div>Zero-fill</div>
            <div class="curve-legend-item ${m==='neighbor'?'active':''}"><div class="curve-legend-line" style="background:#F59E0B"></div>이웃 평균</div>
            <div class="curve-legend-item ${m==='propagation'?'active':''}"><div class="curve-legend-line" style="background:#10B981"></div>특징 전파</div>
        </div>
        <div class="section-divider"></div>
        <h3>방식별 성능 비교</h3>
        <table class="cmp-table">
            <tr><th>방식</th><th>최종 Loss</th><th>MAAPE</th><th>수렴</th></tr>
            <tr class="${m==='zero'?'active-row':''}"><td class="method-name">Zero-fill</td><td>0.34</td><td>12.8%</td><td><span class="speed-bar"><span class="speed-block on"></span><span class="speed-block off"></span><span class="speed-block off"></span></span></td></tr>
            <tr class="${m==='neighbor'?'active-row':''}"><td class="method-name">이웃 평균</td><td>0.19</td><td>8.3%</td><td><span class="speed-bar"><span class="speed-block on"></span><span class="speed-block on"></span><span class="speed-block off"></span></span></td></tr>
            <tr class="${m==='propagation'?'active-row':''}"><td class="method-name">특징 전파</td><td>0.07</td><td>5.2%</td><td><span class="speed-bar"><span class="speed-block on"></span><span class="speed-block on"></span><span class="speed-block on"></span></span></td></tr>
        </table>
        <div class="section-divider"></div>
        <h3>교통량 (실제 vs 예측)</h3>
        <div class="chart-wrap"><canvas id="traffic-canvas" height="90"></canvas>
            <div class="curve-legend"><div class="curve-legend-item"><div class="curve-legend-line" style="background:#3B82F6"></div>실제</div><div class="curve-legend-item"><div class="curve-legend-line" style="background:#EF4444"></div>예측 (${mLabel})</div></div>
        </div>
        <div class="hint">🖱️ 지도 위의 센서를 클릭하면 신호 전파를 볼 수 있습니다</div>`;
    }

    function getResiliencePanel() {
        const onCnt = state.online.filter(Boolean).length;
        const acc = [0, 88, 92, 94, 95][onCnt];
        const latency = [0, 480, 240, 150, 85][onCnt];
        let rr = '';
        if (onCnt < 4 && onCnt > 0) {
            rr = '<div class="reroute-info">';
            for (let ci = 0; ci < 4; ci++) { if (state.online[ci]) continue; const n = nearestOnlineCluster({ lat: clusterCenter(ci).lat, lng: clusterCenter(ci).lng, cluster: ci }); if (n >= 0) rr += `<div class="reroute-line">${C.srvName[ci]} 센서 <span class="reroute-arrow">→</span> <strong style="color:${C.srv[n]}">${C.srvName[n]}</strong>로 재배정</div>`; }
            rr += '</div>';
        }
        return `<h2>⚡ 서버 장애 및 우회(Rerouting)</h2>
        <p class="desc">서버 장애 시 인접 서버로 데이터를 우회 전송합니다. 센서 정보가 보존되어 <strong>정확도는 유지</strong>되지만, 서버 간 통신이 몰려 <strong>네트워크 통신 지연(Latency)</strong>이 증가합니다.</p>
        <div class="section-divider"></div>
        <h3>엣지 서버 제어판</h3>
        <div class="server-grid">${[0,1,2,3].map(i => `<button class="srv-btn ${state.online[i]?'online':'offline'}" data-server="${i}"><span class="srv-icon">🖥️</span><span class="srv-name">${C.srvName[i]}</span><span class="srv-status">${state.online[i]?'온라인':'오프라인'}</span></button>`).join('')}</div>
        ${rr}
        <div class="section-divider"></div>
        <h3>시스템 영향도</h3>
        <div style="display:flex; justify-content:center; align-items:center; gap:20px; padding:10px 0;">
            <div style="text-align:center;">
                <svg class="accuracy-ring" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="#E5E7EB" stroke-width="8"/>
                    <circle cx="60" cy="60" r="50" fill="none" stroke="${acc>=90?'#10B981':acc>=80?'#F59E0B':'#EF4444'}" stroke-width="8" stroke-dasharray="314" stroke-dashoffset="${314*(1-acc/100)}" stroke-linecap="round" transform="rotate(-90 60 60)" style="transition:all .6s ease"/>
                    <text x="60" y="66" text-anchor="middle" fill="#1F2937" font-size="22" font-weight="800">${acc}%</text>
                </svg>
                <div style="font-size:12px; font-weight:600; color:var(--text2); margin-top:8px;">예측 정확도</div>
            </div>
            <div style="text-align:center; padding:15px; background:#FEF2F2; border-radius:12px; min-width:110px;">
                <div style="font-size:26px; font-weight:800; color:#EF4444;">${latency}<span style="font-size:14px; font-weight:600;">ms</span></div>
                <div style="font-size:12px; font-weight:600; color:#991B1B; margin-top:4px;">우회 지연시간</div>
            </div>
        </div>
        <div class="hint">⚠️ 장애 시 우회하는 센서의 색상이 <b>해당 서버의 색</b>으로 변경되며 화살표가 표시됩니다.</div>`;
    }

    function renderMathPanel() {
        const mp = document.getElementById('math-panel');
        if (!mp) return;
        if (state.mode !== 'gnn') {
            mp.classList.remove('visible');
            return;
        }
        mp.classList.add('visible');

        const m = state.imputeMethod;
        const mName = m === 'zero' ? 'Zero-fill' : m === 'neighbor' ? '이웃 평균' : '특징 전파';

        if (state.gnnAnimating) {
            mp.innerHTML = `<h2>🧮 GNN 다이나믹 연산 모델</h2>
            <p class="desc" style="margin-bottom: 270px;">지도상의 관측 데이터(센서 네트워크 변수)가 행렬 구조로 재배열되어 시공간 특징을 학습하는 과정을 역동적으로 추적합니다. (현재 보간 상태: <strong>${mName}</strong>)</p>
            <div style="font-size:11.5px; line-height:1.5; color:var(--text2); background:#F3F4F6; padding:10px; border-radius:8px;">
                모든 센서 노드가 허공으로 떠오르며 <strong>인접행렬(Ã)</strong>과 <strong>특징벡터(X)</strong>로 치환되고, 딥러닝 뉴럴 네트워크(GNN Layer)를 거쳐 특징 임베딩(H)으로 업데이트되어 다시 지도상의 노드로 회귀합니다.
            </div>`;
            return;
        }

        let nodeVals = [];
        if (m === 'zero') {
            nodeVals = [{c:'v-real',t:'95'}, {c:'v-zero',t:'0'}, {c:'v-zero',t:'0'}, {c:'v-zero',t:'0'}];
        } else if (m === 'neighbor') {
            nodeVals = [{c:'v-real',t:'95'}, {c:'v-avg',t:'47'}, {c:'v-zero',t:'0'}, {c:'v-avg',t:'47'}];
        } else {
            nodeVals = [{c:'v-real',t:'95'}, {c:'v-prop',t:'78'}, {c:'v-prop',t:'65'}, {c:'v-prop',t:'70'}];
        }
        const cellsHTML = nodeVals.map(v => `<div class="mat-cell ${v.c}">${v.t}</div>`).join('');

        mp.innerHTML = `<h2>🧮 GNN 다이나믹 연산 모델</h2>
        <p class="desc">선택된 <strong>${mName}</strong> 결과를 통과한 특징 행렬 <span style="font-family:monospace; font-weight:bold;">X</span>가 GNN 학습에 입력됩니다.</p>
        <div class="math-eq">H<sup>(l+1)</sup> = σ( Ã · X<sup>(l)</sup> · W )</div>
        <div class="matrix-container">
            <div style="text-align:center;">
                <div style="font-size:11px; color:#6B7280; font-weight:bold; margin-bottom:2px;">Ã (인접행렬)</div>
                <div class="matrix" style="grid-template-columns: repeat(4, 14px);">
                    <div class="mat-cell gray">1</div><div class="mat-cell gray">1</div><div class="mat-cell gray">0</div><div class="mat-cell gray">1</div>
                    <div class="mat-cell gray">1</div><div class="mat-cell gray">1</div><div class="mat-cell gray">1</div><div class="mat-cell gray">0</div>
                    <div class="mat-cell gray">0</div><div class="mat-cell gray">1</div><div class="mat-cell gray">1</div><div class="mat-cell gray">1</div>
                    <div class="mat-cell gray">1</div><div class="mat-cell gray">0</div><div class="mat-cell gray">1</div><div class="mat-cell gray">1</div>
                </div>
            </div>
            <div style="font-size:16px; font-weight:bold; color:var(--text2);">×</div>
            <div style="text-align:center;">
                <div style="font-size:11px; color:#6B7280; font-weight:bold; margin-bottom:2px;">X (특징)</div>
                <div class="matrix" style="grid-template-columns: 14px;">
                    ${cellsHTML}
                </div>
            </div>
            <div style="font-size:16px; font-weight:bold; color:var(--text2);">⇒</div>
            <div class="nn-block">
                <div class="pulse"></div>
                <span>GNN</span>
                <span>Layer</span>
            </div>
        </div>
        <button id="btn-run-gnn" style="margin-top:15px; width:100%; padding:10px; background:#10B981; color:white; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">▶ 연산 애니메이션 실행</button>
        <div style="margin-top:10px; font-size:11.5px; line-height:1.5; color:var(--text2); background:#F3F4F6; padding:10px; border-radius:8px;">
            행렬 곱 연산(Ã × X)을 통해 주변 센서 간의 상태 정보가 혼합되며 특징이 추출됩니다.
        </div>`;

        setTimeout(() => {
            const btn = document.getElementById('btn-run-gnn');
            if (btn) btn.addEventListener('click', () => {
                state.gnnAnimating = true;
                state.gnnAnimStart = performance.now();
                renderMathPanel();
            });
        }, 50);
    }

    // ===== Mode switching =====
    function setMode(mode) {
        state.mode = mode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
        renderPanel(); updateLegend();
        if (mode === 'gnn') { state.waves = []; state.lossFrame = 0; }
        if (mode === 'resilience') { state.online = [true, true, true, true]; }
        if (mode === 'fl') { state.flPlaying = false; state.flStep = 0; state.flProgress = 0; state.flRound = 1; }
    }
    function renderPanel() {
        const p = document.getElementById('info-panel');
        if (state.mode === 'overview') p.innerHTML = getOverviewPanel();
        else if (state.mode === 'fl') p.innerHTML = getFLPanel();
        else if (state.mode === 'gnn') p.innerHTML = getGNNPanel();
        else p.innerHTML = getResiliencePanel();
        renderMathPanel();
        bindModeEvents();
    }
    function updateLegend() {
        const lp = document.getElementById('legend-panel');
        const dots = C.srv.map((c, i) => `<div class="legend-item"><div class="legend-dot" style="background:${c}"></div>${C.srvName[i]}</div>`).join('');
        let ex = '';
        if (state.mode === 'overview') ex =
            '<div class="legend-item"><div class="legend-dot" style="background:#10B981"></div>원활 (차량흐름●)</div>' +
            '<div class="legend-item"><div class="legend-dot" style="background:#F59E0B"></div>서행</div>' +
            '<div class="legend-item"><div class="legend-dot" style="background:#EF4444"></div>정체</div>';
        if (state.mode === 'fl') ex =
            '<div class="legend-item"><div class="legend-dot" style="background:#0EA5E9"></div>중앙 서버</div>' +
            '<div class="legend-item"><div style="display:inline-block;width:10px;height:10px;background:#3B82F6;clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%);margin-right:4px"></div>모델 파라미터◆</div>';
        if (state.mode === 'gnn') ex =
            '<div class="legend-item"><div class="legend-dot" style="background:' + C.srv[FOCAL_CLUSTER] + '"></div>서버 A 센서 (알려진값)</div>' +
            '<div class="legend-item"><div class="legend-dot" style="background:#D1D5DB;border:1px solid #9CA3AF"></div>타 서버 (결측→보간)</div>';
        if (state.mode === 'resilience') ex =
            '<div class="legend-item"><div class="legend-dot" style="background:#D1D5DB;border:1px solid #9CA3AF"></div>오프라인 서버</div>' +
            '<div class="legend-item"><div class="legend-dot" style="background:#10B981"></div>재배정 경로</div>';
        lp.innerHTML = dots + '<div class="legend-item"><div class="legend-line"></div>센서 연결</div>' + ex;
    }
    function bindModeEvents() {
        const play = document.getElementById('fl-play');
        if (play) play.addEventListener('click', () => { state.flPlaying = !state.flPlaying; play.textContent = state.flPlaying ? '⏸' : '▶'; play.classList.toggle('active', state.flPlaying); });
        const sb = document.getElementById('fl-step-btn');
        if (sb) sb.addEventListener('click', () => { state.flStep = (state.flStep + 1) % 4; state.flProgress = 0; if (state.flStep === 0) state.flRound = Math.min(100, state.flRound + 1); updateFLPanel(); });
        const rb = document.getElementById('fl-reset');
        if (rb) rb.addEventListener('click', () => { state.flRound = 1; state.flStep = 0; state.flProgress = 0; state.flPlaying = false; const pb = document.getElementById('fl-play'); if (pb) { pb.textContent = '▶'; pb.classList.remove('active'); } updateFLPanel(); });
        const si = document.getElementById('fl-speed');
        if (si) si.addEventListener('input', () => { state.flSpeed = parseInt(si.value); });
        document.querySelectorAll('.impute-btn').forEach(btn => { btn.addEventListener('click', () => { state.imputeMethod = btn.dataset.imp; state.lossFrame = 0; renderPanel(); }); });
        document.querySelectorAll('.srv-btn').forEach(btn => { btn.addEventListener('click', () => { state.online[parseInt(btn.dataset.server)] = !state.online[parseInt(btn.dataset.server)]; renderPanel(); }); });
    }
    function updateFLPanel() {
        const b = document.getElementById('fl-round-badge'); if (b) b.textContent = state.flRound + ' / 100';
        document.querySelectorAll('.step-dot-wrap').forEach(w => { const s = parseInt(w.dataset.s); w.classList.remove('active', 'done'); w.querySelector('.step-dot').classList.remove('active', 'done'); if (s === state.flStep) { w.classList.add('active'); w.querySelector('.step-dot').classList.add('active'); } else if (s < state.flStep) { w.classList.add('done'); w.querySelector('.step-dot').classList.add('done'); } });
        const d = document.getElementById('fl-step-desc'); if (d) d.textContent = FL_DESCS[state.flStep];
    }

    // ===== Init =====
    function init() { initMap(); initCanvas(); document.querySelectorAll('.mode-btn').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode))); setMode('overview'); requestAnimationFrame(draw); }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
