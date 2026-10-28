#!/usr/bin/env python3
"""
Launcher that starts both:
1. The Python EEG API microservice (serve_eeg_agent_api.py)
2. The Cloudflare Tunnel (cloudflared.exe)
Extracts the public HTTPS URL, updates the n8n workflow files, and keeps running.
"""

import sys
import os
import re
import time
import subprocess

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
CLOUDFLARED_EXE = os.path.join(WORKSPACE, 'cloudflared.exe')

print("[Launcher] Starting EEG microservice...")
api_proc = subprocess.Popen([PYTHON_EXE, 'serve_eeg_agent_api.py'], cwd=WORKSPACE)
time.sleep(2)

print("[Launcher] Starting Cloudflare Tunnel to port 8000...")
tunnel_proc = subprocess.Popen(
    [CLOUDFLARED_EXE, 'tunnel', '--url', 'http://127.0.0.1:8000', '--no-autoupdate'],
    cwd=WORKSPACE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

public_url = None
url_pattern = re.compile(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com')

# Read output to capture the public URL
for line in iter(tunnel_proc.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
    match = url_pattern.search(line)
    if match and not public_url:
        public_url = match.group(0)
        print("\n" + "="*70)
        print(f"PUBLIC TUNNEL URL READY FOR n8n CLOUD: {public_url}")
        print("="*70 + "\n")
        
        # Save to file
        with open(os.path.join(WORKSPACE, 'public_url.txt'), 'w') as f:
            f.write(public_url.strip())
            
        # Update workflow files automatically
        for wf_file in ['n8n_eeg_schizophrenia_agent_workflow.json', 'n8n_core_webhook_workflow.json']:
            wf_path = os.path.join(WORKSPACE, wf_file)
            if os.path.exists(wf_path):
                with open(wf_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                updated_content = content.replace('http://127.0.0.1:8000', public_url)
                with open(wf_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"[Launcher] Updated {wf_file} with public URL: {public_url}")

# Keep processes alive
try:
    tunnel_proc.wait()
except KeyboardInterrupt:
    print("[Launcher] Stopping services...")
    tunnel_proc.terminate()
    api_proc.terminate()
