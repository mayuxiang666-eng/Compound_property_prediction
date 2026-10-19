"""
app.py
Continental Mixing Process DOE Workbench & Trial Auto-Evaluator Web Application
A specialized engineering platform for Carbon Black and Silica compound DOE optimization.
"""

import os
import sys

# Ensure workspace root is in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify, request
from web_dashboard.services.recommendation_engine import (
    COMPOUND_REGISTRY,
    get_compound_recipe_and_recommendations
)
from web_dashboard.services.trial_evaluator import (
    evaluate_trial_order,
    get_known_trial_orders,
    load_datasets
)
from web_dashboard.services.energy_profiler import (
    get_compounds_energy_ranking,
    get_compound_material_phr_profile,
    get_compound_step_energy_profile
)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Preload data cache on startup
load_datasets()


@app.route('/api/energy_ranking', methods=['GET'])
def get_energy_ranking():
    """Returns cross-compound energy intensity and savings potential rankings."""
    ranking = get_compounds_energy_ranking()
    return jsonify({'status': 'success', 'data': ranking})


@app.route('/api/material_impact/<compound_id>', methods=['GET'])
def get_material_impact(compound_id):
    """Returns raw material phr formulation breakdown and energy sensitivities."""
    profile = get_compound_material_phr_profile(compound_id)
    return jsonify({'status': 'success', 'data': profile})


@app.route('/api/step_energy_impact/<compound_id>', methods=['GET'])
def get_step_energy_impact(compound_id):
    """Returns step-by-step energy consumption dynamics and sensitivity breakdown."""
    steps = get_compound_step_energy_profile(compound_id)
    return jsonify({'status': 'success', 'data': steps})


@app.route('/')
def index():
    """Renders the main engineering cockpit."""
    return render_template('index.html')


@app.route('/api/compounds', methods=['GET'])
def list_compounds():
    """Returns list of supported compounds with metadata."""
    compounds = []
    for cid, data in COMPOUND_REGISTRY.items():
        compounds.append({
            'id': data['id'],
            'full_name': data['full_name'],
            'system': data['system'],
            'category_zh': data.get('category_zh', data.get('category', '')),
            'category_en': data.get('category_en', data.get('category', '')),
            'mixer': data['mixer'],
            'mixer_alt': data.get('mixer_alt', ''),
            'batch_weight': data['batch_weight'],
            'mooney_spec': data['mooney_spec'],
            'mooney_target': data['mooney_target'],
            'baseline_kwh_ton': data['baseline_kwh_ton'],
            'baseline_kwh_batch': data['baseline_kwh_batch'],
            'description_zh': data.get('description_zh', data.get('description', '')),
            'description_en': data.get('description_en', data.get('description', ''))
        })
    return jsonify({'status': 'success', 'compounds': compounds})


@app.route('/api/recipe/<compound_id>', methods=['GET'])
def get_recipe_doe(compound_id):
    """
    Returns step-by-step recipe baseline and prescriptive DOE recommendation
    with strict rule compliance and calculation rationale.
    """
    result = get_compound_recipe_and_recommendations(compound_id)
    return jsonify({'status': 'success', 'data': result})


@app.route('/api/quick_trial_orders', methods=['GET'])
def quick_trials():
    """Returns curated trial orders for 1-click evaluation."""
    trials = get_known_trial_orders()
    return jsonify({'status': 'success', 'orders': trials})


@app.route('/api/evaluate_trial', methods=['POST'])
def run_trial_evaluation():
    """
    Evaluates an engineer-submitted trial order.
    Body: {"order_id": "...", "notes": "..."}
    """
    req = request.get_json(silent=True) or {}
    order_id = req.get('order_id') or request.form.get('order_id')
    notes = req.get('notes') or request.form.get('notes')
    modifications = req.get('modifications') or []

    if not order_id:
        return jsonify({'status': 'error', 'message': '请输入有效的工单号 (Order ID)。'}), 400

    result = evaluate_trial_order(order_id, notes, modifications)
    return jsonify(result)


@app.route('/api/evaluate_trial/<order_id>', methods=['GET'])
def run_trial_evaluation_get(order_id):
    """GET shorthand for evaluating a specific order ID."""
    notes = request.args.get('notes', '')
    result = evaluate_trial_order(order_id, notes)
    return jsonify(result)


# =============================================================================
# DEDICATED ENERGY OPTIMIZER & RANKING WORKBENCH API ENDPOINTS
# =============================================================================
import json
from flask import send_from_directory

OPTIMIZER_DATA_PATH = os.path.join(PROJECT_ROOT, 'reports', 'compound_energy_optimization_auto', 'compound_energy_optimization_data.json')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'compound_energy_optimization_auto')


@app.route('/api/energy_optimizer/data', methods=['GET'])
def get_energy_optimizer_full_data():
    """Returns the full consolidated optimization ranking and step diagnostic dataset."""
    if not os.path.exists(OPTIMIZER_DATA_PATH):
        # Auto-generate if missing
        import subprocess
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'build_energy_optimization_data.py')], check=False)

    if os.path.exists(OPTIMIZER_DATA_PATH):
        with open(OPTIMIZER_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({'status': 'error', 'message': 'Optimization dataset not found'}), 404


@app.route('/api/energy_optimizer/ranking', methods=['GET'])
def get_energy_optimizer_ranking():
    """Returns plant-wide ranking list with Top 5 Carbon Black & Top 5 Silica."""
    if os.path.exists(OPTIMIZER_DATA_PATH):
        with open(OPTIMIZER_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'status': 'success', 'data': data.get('rankings', []), 'summary': data.get('summary', {})})
    return jsonify({'status': 'error', 'message': 'Dataset not available'}), 404


@app.route('/api/energy_optimizer/diagnostics/<path:compound_id>', methods=['GET'])
def get_energy_optimizer_diagnostics(compound_id):
    """Returns step-level telemetry, Wi variance breakdown, and prescriptive DOE actions for a compound."""
    if os.path.exists(OPTIMIZER_DATA_PATH):
        with open(OPTIMIZER_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        diag = data.get('diagnostics', {}).get(compound_id)
        if diag:
            return jsonify({'status': 'success', 'data': diag})
        # Check by prefix / short code
        for k, v in data.get('diagnostics', {}).items():
            if compound_id.lower() in k.lower() or compound_id.lower() in v.get('short_code', '').lower():
                return jsonify({'status': 'success', 'data': v})
    return jsonify({'status': 'error', 'message': f'Compound {compound_id} not found in diagnostics'}), 404


@app.route('/api/energy_optimizer/run_batch', methods=['POST'])
def run_energy_optimizer_batch():
    """Executes full-process batch recalculation and refreshes the data payload."""
    try:
        import subprocess
        res = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'build_energy_optimization_data.py')], capture_output=True, text=True, check=True)
        return jsonify({
            'status': 'success',
            'message': '全流程一键批处理重算执行成功！',
            'output': res.stdout
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/reports/compound_energy_optimization_auto/<path:filename>')
def serve_optimization_reports(filename):
    """Serves high-resolution diagnostic dashboards and generated CSVs."""
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == '__main__':
    print("=" * 60)
    print("Continental Mixing DOE Workbench & Trial Evaluator")
    print("Server running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)

