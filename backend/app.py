from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import math
import pandas as pd
import json

from energyMatrix import compute_energy_matrix_with_stops, compute_time_matrix, export_matrices_to_excel
from extractLatLong import get_railways_between_coords
from railpathFinder import extract_freight_path
from algorithm import generate_output_excel
from results import parse_results_schedule

# -------------------- Flask App Setup --------------------
# Set static_folder to React build
app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- API Routes --------------------
@app.route('/api', methods=['GET'])
def api():
    return jsonify({"message": "API is working"})

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "inputData.xlsx")
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/generate-energy-matrix', methods=['POST'])
def generate_energy_matrix():
    try:
        data = request.json
        assumptions = data['assumptions']
        stations = [s * 1.0 for s in data['stations']]
        energy_matrix = compute_energy_matrix_with_stops(stations, assumptions)
        time_matrix = compute_time_matrix(stations, assumptions)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "energy_matrix_ui.xlsx")
        export_matrices_to_excel(energy_matrix, time_matrix, stations, output_path)
        return jsonify({"message": "Matrices generated successfully!"})
