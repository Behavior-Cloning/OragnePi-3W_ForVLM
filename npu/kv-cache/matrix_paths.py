"""Resolve the attempt named by the matrix, preserving failed originals."""
import json
from pathlib import Path


def cpu_result_path(root, case, band):
    root = Path(root)
    state = json.loads((root/'state.json').read_text())
    row = state.get('cases', {}).get(case, {}).get('bands', {}).get(band, {})
    folder = Path(row['cpu_path']) if row.get('cpu_path') else root/(case+'-'+band+'-cpu')
    return folder/'results.json'
