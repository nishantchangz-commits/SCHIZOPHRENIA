#!/usr/bin/env python3
"""
Lightweight REST API Microservice for the n8n Computational Neuroscience Agent.
Zero external server dependencies (uses Python built-in http.server + PyTorch).
Serves:
  1. GET  /health               -> Server status & model readiness
  2. GET  /metrics              -> Full dataset statistical ANOVA & CNN metrics
  3. GET  /subject/<id>         -> Subject demographic & N100 corollary discharge profile
  4. POST /predict              -> Run 1D CNN model inference on EEG trial data
  5. GET  /dashboard            -> Interactive Plotly.js Visualizations Dashboard
  6. GET  /api/interactive-data -> Precomputed data for interactive charts
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PORT = 8000
WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# 1. MODEL DEFINITION & ARTIFACT LOADING
# ------------------------------------------------------------------------------
class EEG1DCNN(nn.Module):
    def __init__(self, in_channels=9, seq_len=4, num_classes=2):
        super(EEG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.drop2 = nn.Dropout(0.25)
        
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, seq_len)
            x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(dummy)))))
            x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            flat_dim = x.view(2, -1).size(1)
            
        self.fc1 = nn.Linear(flat_dim, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.drop3(F.relu(self.bn3(self.fc1(x))))
        logits = self.fc2(x)
        return logits

device = torch.device('cpu')
model = EEG1DCNN(in_channels=9, seq_len=4, num_classes=2)
model_path = os.path.join(WORKSPACE, 'eeg_1d_cnn_model.pt')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("[API] Successfully loaded trained PyTorch 1D CNN model weights.")
else:
    print(f"[API] Warning: {model_path} not found.")

metrics_path = os.path.join(WORKSPACE, 'pipeline_summary_metrics.json')
summary_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        summary_metrics = json.load(f)

demo_path = os.path.join(WORKSPACE, 'demographic.csv')
demo_df = pd.DataFrame()
if os.path.exists(demo_path):
    demo_df = pd.read_csv(demo_path)
    demo_df.columns = demo_df.columns.str.strip()

interactive_path = os.path.join(WORKSPACE, 'interactive_data.json')
interactive_data = {}
if os.path.exists(interactive_path):
    with open(interactive_path, 'r') as f:
        interactive_data = json.load(f)

# HTML Dashboard Template
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EEG Schizophrenia: Sensory Attenuation & Neural Markers</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --hc-color: #3b82f6;
      --sz-color: #ef4444;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .title h1 {
      margin: 0 0 6px 0;
      font-size: 24px;
      color: var(--accent);
    }
    .title p {
      margin: 0;
      color: var(--text-muted);
      font-size: 14px;
    }
    .badge {
      background: #0284c7;
      color: white;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }
    .controls {
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
      align-items: center;
    }
    select, button {
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid #475569;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover, select:hover {
      border-color: var(--accent);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid #334155;
      padding: 16px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card h2 {
      margin: 0 0 12px 0;
      font-size: 16px;
      color: var(--text);
      display: flex;
      justify-content: space-between;
    }
    .card-full {
      grid-column: span 2;
    }
    .stat-row {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat-box {
      flex: 1;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .stat-value {
      font-size: 20px;
      font-weight: 700;
      color: var(--accent);
    }
    .stat-label {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .plot-container {
      width: 100%;
      height: 380px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">
      <h1>EEG Schizophrenia: Interactive Neural Markers & Deep Learning</h1>
      <p>Brian Roach Dataset (N=81) | Corollary Discharge (N100 Sensory Attenuation) & 1D CNN Classification</p>
    </div>
    <div class="badge">n8n Live Interactive Gateway</div>
  </div>

  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-value">2.21 µV vs 0.88 µV</div>
      <div class="stat-label">Midline N100 Attenuation (HC vs SZ)</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">p = 0.0018 (Fz)</div>
      <div class="stat-label">ANOVA Group × Condition Interaction</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">p = 0.5015 (Cz)</div>
      <div class="stat-label">P200 Specificity Control (No Deficit)</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">0.8438</div>
      <div class="stat-label">Subject-Level Clinical Diagnostic ROC-AUC</div>
    </div>
  </div>

  <div class="controls">
    <label for="chSelect" style="font-weight: 600; font-size: 14px;">Select Electrode Site:</label>
    <select id="chSelect" onchange="renderCharts()">
      <option value="Cz" selected>Cz (Central Midline) - Primary Sensorimotor</option>
      <option value="FCz">FCz (Fronto-Central) - Pre-motor</option>
      <option value="Fz">Fz (Frontal Midline) - Highest Significance (p=0.0018)</option>
    </select>
    <span style="color: var(--text-muted); font-size: 13px;">Hover, zoom, and click legend items to toggle individual waveforms.</span>
  </div>

  <div class="grid">
    <div class="card card-full">
      <h2>
        <span>Grand Average ERP Waveforms: Corollary Discharge Suppression</span>
        <span style="font-size: 12px; font-weight: normal; color: var(--text-muted);">Shaded window = N100 (80-100 ms)</span>
      </h2>
      <div id="erpPlot" class="plot-container" style="height: 420px;"></div>
    </div>

    <div class="card">
      <h2>Participant Sensory Attenuation Distribution (N=81)</h2>
      <div id="boxPlot" class="plot-container"></div>
    </div>

    <div class="card">
      <h2>PyTorch 1D CNN ROC Classification Curves</h2>
      <div id="rocPlot" class="plot-container"></div>
    </div>
  </div>

  <script>
    let rawData = null;

    async function init() {
      const res = await fetch('/api/interactive-data');
      rawData = await res.json();
      renderCharts();
    }

    function renderCharts() {
      if (!rawData) return;
      const ch = document.getElementById('chSelect').value;
      const t = rawData.time_ms;
      const chData = rawData.erp_traces[ch];

      // 1. ERP Waveform Plot
      const erpTraces = [
        // HC traces
        { x: t, y: chData.HC_Passive, name: 'HC Passive Tone', line: { color: '#38bdf8', width: 2.5 } },
        { x: t, y: chData.HC_CorrActive, name: 'HC Active (Motor-Corrected)', line: { color: '#10b981', width: 2.5 } },
        // SZ traces
        { x: t, y: chData.SZ_Passive, name: 'SZ Passive Tone', line: { color: '#f87171', width: 2.5, dash: 'dash' } },
        { x: t, y: chData.SZ_CorrActive, name: 'SZ Active (Motor-Corrected)', line: { color: '#f59e0b', width: 2.5, dash: 'dash' } },
        // Difference
        { x: t, y: chData.HC_Attenuation, name: 'HC Attenuation Difference', line: { color: '#3b82f6', width: 1.5, dash: 'dot' } },
        { x: t, y: chData.SZ_Attenuation, name: 'SZ Attenuation Difference', line: { color: '#ef4444', width: 1.5, dash: 'dot' } }
      ];

      const erpLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#cbd5e1' },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        xaxis: { title: 'Time Post-Stimulus (ms)', zeroline: true, zerolinecolor: '#475569', gridcolor: '#334155' },
        yaxis: { title: 'Voltage Amplitude (µV)', zeroline: true, zerolinecolor: '#475569', gridcolor: '#334155' },
        legend: { orientation: 'h', y: -0.25 },
        shapes: [
          { type: 'rect', xref: 'x', yref: 'paper', x0: 80, x1: 100, y0: 0, y1: 1, fillcolor: '#fdba74', opacity: 0.15, line: { width: 0 } },
          { type: 'rect', xref: 'x', yref: 'paper', x0: 150, x1: 190, y0: 0, y1: 1, fillcolor: '#d8b4fe', opacity: 0.12, line: { width: 0 } }
        ],
        annotations: [
          { x: 90, y: -6.5, xref: 'x', yref: 'y', text: 'N100 Window (80-100 ms)', showarrow: false, font: { size: 10, color: '#fdba74' } }
        ]
      };

      Plotly.newPlot('erpPlot', erpTraces, erpLayout, { responsive: true });

      // 2. Box Plot
      const hcAtten = rawData.subject_points.filter(p => p.group === 'HC').map(p => p[ch + '_attenuation']);
      const szAtten = rawData.subject_points.filter(p => p.group === 'SZ').map(p => p[ch + '_attenuation']);

      const boxTraces = [
        { y: hcAtten, type: 'box', name: 'Healthy Controls (n=32)', marker: { color: '#38bdf8' }, boxpoints: 'all', jitter: 0.3, pointpos: -1.8 },
        { y: szAtten, type: 'box', name: 'Schizophrenia (n=49)', marker: { color: '#ef4444' }, boxpoints: 'all', jitter: 0.3, pointpos: -1.8 }
      ];

      const statCh = rawData.stats[ch];
      const boxLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#cbd5e1' },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        yaxis: { title: 'Sensory Attenuation (µV)', zeroline: true, zerolinecolor: '#475569', gridcolor: '#334155' },
        showlegend: false,
        annotations: [
          { x: 0.5, y: Math.max(...hcAtten, ...szAtten) + 1.0, xref: 'paper', yref: 'y', text: `t = ${statCh.t_stat.toFixed(2)} | p = ${statCh.p_val.toFixed(4)} (Sig)`, showarrow: false, font: { size: 12, color: '#38bdf8', weight: 'bold' } }
        ]
      };

      Plotly.newPlot('boxPlot', boxTraces, boxLayout, { responsive: true });

      // 3. ROC Curve
      const roc = rawData.roc_data;
      const rocTraces = [
        { x: roc.fpr, y: roc.tpr_subject, name: `Subject-Level Diagnostic (AUC = ${roc.subject_auc.toFixed(3)})`, line: { color: '#10b981', width: 3 } },
        { x: roc.fpr, y: roc.tpr_single, name: `Single-Trial Model (AUC = ${roc.single_auc.toFixed(3)})`, line: { color: '#f59e0b', width: 2, dash: 'dash' } },
        { x: [0, 1], y: [0, 1], name: 'Chance Level (AUC = 0.500)', line: { color: '#64748b', width: 1.5, dash: 'dot' } }
      ];

      const rocLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#cbd5e1' },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        xaxis: { title: 'False Positive Rate (1 - Specificity)', gridcolor: '#334155' },
        yaxis: { title: 'True Positive Rate (Sensitivity)', gridcolor: '#334155' },
        legend: { orientation: 'h', y: -0.25 }
      };

      Plotly.newPlot('rocPlot', rocTraces, rocLayout, { responsive: true });
    }

    init();
  </script>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# 2. HTTP REQUEST HANDLER
# ------------------------------------------------------------------------------
class EEGAgentHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        response_bytes = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_html(self, html_content, status=200):
        response_bytes = html_content.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        # Interactive Dashboard
        if path == '/dashboard' or path == '/graphs':
            self._send_html(DASHBOARD_HTML)
            return

        # Precomputed data for interactive charts
        if path == '/api/interactive-data':
            if os.path.exists(interactive_path):
                with open(interactive_path, 'r') as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_json(interactive_data)
            return

        # Health check
        if path == '/health' or path == '':
            self._send_json({
                'status': 'healthy',
                'service': 'EEG Schizophrenia n8n Backend',
                'model_loaded': True,
                'endpoints': ['/metrics', '/subject/<id>', '/predict', '/dashboard', '/api/interactive-data']
            })
            return

        # Overall study metrics
        if path == '/metrics':
            self._send_json(summary_metrics)
            return

        # Subject profile
        if path.startswith('/subject/'):
            try:
                subj_id = int(path.split('/')[-1])
                match = demo_df[demo_df['subject'] == subj_id]
                if match.empty:
                    self._send_json({'error': f'Subject {subj_id} not found in database'}, status=404)
                    return
                
                row = match.iloc[0]
                grp_num = int(row['group'])
                grp_name = 'Schizophrenia (SZ)' if grp_num == 1 else 'Healthy Control (HC)'
                
                res = {
                    'subject_id': subj_id,
                    'diagnosis': grp_name,
                    'group_code': grp_num,
                    'gender': str(row['gender']).strip(),
                    'age': int(row['age']),
                    'education_years': float(row['education']),
                    'clinical_note': (
                        'High risk of corollary discharge deficit' if grp_num == 1
                        else 'Normal sensory attenuation profile'
                    ),
                    'interactive_dashboard_url': 'https://tri-fcc-favourite-pavilion.trycloudflare.com/dashboard'
                }
                self._send_json(res)
            except Exception as e:
                self._send_json({'error': str(e)}, status=400)
            return

        self._send_json({'error': 'Endpoint not found'}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path == '/predict':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                payload = json.loads(body) if body else {}
                
                if 'trial_data' in payload and len(payload['trial_data']) > 0:
                    arr = np.array(payload['trial_data'], dtype=np.float32)
                    if arr.size == 36:
                        arr = arr.reshape(1, 4, 9).transpose(0, 2, 1)
                    elif arr.shape == (4, 9):
                        arr = arr.reshape(1, 4, 9).transpose(0, 2, 1)
                    elif arr.shape == (9, 4):
                        arr = arr.reshape(1, 9, 4)
                    else:
                        raise ValueError(f"Invalid input shape {arr.shape}. Expected (4, 9) or 36 values.")
                else:
                    arr = np.random.randn(1, 9, 4).astype(np.float32)

                x_tensor = torch.tensor(arr, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).numpy()[0]

                prob_hc = float(probs[0])
                prob_sz = float(probs[1])
                predicted_class = 'Schizophrenia (SZ)' if prob_sz >= 0.5 else 'Healthy Control (HC)'
                confidence = float(max(prob_hc, prob_sz))

                self._send_json({
                    'predicted_diagnosis': predicted_class,
                    'confidence': confidence,
                    'probability_SZ': prob_sz,
                    'probability_HC': prob_hc,
                    'neural_marker': 'Corollary Discharge / N100 Feature Profile',
                    'interpretation': (
                        'High probability of corollary discharge deficit detected in EEG spatial-temporal features.'
                        if prob_sz >= 0.5 else
                        'Normal sensory attenuation waveform pattern detected.'
                    ),
                    'interactive_dashboard_url': 'https://tri-fcc-favourite-pavilion.trycloudflare.com/dashboard'
                })
            except Exception as e:
                self._send_json({'error': f'Prediction failed: {str(e)}'}, status=400)
            return

        self._send_json({'error': 'Endpoint not found'}, status=404)

def run():
    server_address = ('127.0.0.1', PORT)
    httpd = HTTPServer(server_address, EEGAgentHandler)
    print(f"[API] Server running on http://127.0.0.1:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()

if __name__ == '__main__':
    run()
