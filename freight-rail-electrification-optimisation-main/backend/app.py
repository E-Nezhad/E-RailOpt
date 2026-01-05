from flask import Flask, request, jsonify, send_from_directory
import math
from flask_cors import CORS
import pandas as pd
import os
import json
from energyMatrix import compute_energy_matrix_with_stops, compute_time_matrix, export_matrices_to_excel  # Import relevant functions
from extractLatLong import get_railways_between_coords
from railpathFinder import extract_freight_path
from algorithm import generate_output_excel
from results import parse_results_schedule


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api', methods=['GET'])
def api():
    return jsonify({"message": "API is working"})

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "inputData.xlsx")
    file.save(file_path)  # Save file to uploads folder
    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/generate-energy-matrix', methods=['POST'])
def generate_energy_matrix():
    try:
        data = request.json
        assumptions = data['assumptions']
        stations = data['stations']  # List of station chainages
        stations = [s * 1.0 for s in stations]  # Ensure float type
        print("this is POWERERE")
        print(assumptions["P"])

        # Generate the energy matrix using the energy matrix computation
        energy_matrix = compute_energy_matrix_with_stops(stations, assumptions)
        time_matrix = compute_time_matrix(stations, assumptions)

        # Ensure uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save Excel output to the UPLOAD_FOLDER
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "energy_matrix_ui.xlsx")
        export_matrices_to_excel(energy_matrix, time_matrix, stations, output_path)

        return jsonify({"message": "Matrices for energy consumption and time generated successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['DELETE'])
def reset_input_data():
    try:
        # Delete the uploaded input file
        input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "inputData.xlsx")
        if os.path.exists(input_file_path):
            os.remove(input_file_path)

        # Delete the generated matrix Excel file
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "energy_matrix_ui.xlsx")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        return jsonify({"message": "Reset completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/get-railways', methods=['POST'])
def get_railways():
    content = request.json
    try:
        start_lat = float(content["startLat"])
        start_lon = float(content["startLon"])
        end_lat = float(content["endLat"])
        end_lon = float(content["endLon"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid or missing coordinates"}), 400

    try:
        segments = get_railways_between_coords(start_lat, start_lon, end_lat, end_lon)
        return jsonify({"segments": segments}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-railway-path', methods=['POST'])
def get_railway_path():
    content = request.json
    try:
        start = (float(content["startLat"]), float(content["startLon"]))
        end = (float(content["endLat"]), float(content["endLon"]))
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid coordinates"}), 400

    try:
        segments = get_railways_between_coords(*start, *end)
        path = extract_freight_path(segments, start, end)
        return jsonify({"path": path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-optimisation', methods=['POST'])
def run_algorithm():
    data = request.json
    assumptions = data['assumptions']
    stations = data['stations']  # List of station chainages
    stations = [s * 1.0 for s in stations]  # Ensure float type
    stationDetails = data['stationDetails']
    print("here I am")
    print(stationDetails)
    try:
        source_path = './uploads/energy_matrix_ui.xlsx'
        output_path_xlsx = './FRE/Data/Random/Random Instance 1.xlsx'
        output_path = './FRE/Data/Random/Random Instance 1.xls'
        print("here I am")
        generate_output_excel(assumptions, stations, stationDetails, source_path, output_path_xlsx, output_path)
        print("here after am")
        return jsonify({"message": "Algorithm ran successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/parse-results', methods=['POST'])  # or POST if preferred
def parse_results():
    data = request.get_json()

    if not data or 'stations' not in data:
        return jsonify({'error': 'Missing station chainages'}), 400
    
    all_station_chainages = data['stations']  # passed from frontend
    print("all staitons data")
    print(all_station_chainages)

    filepath = './FRE/Results/Random Instances/Random Instance 1_Delay5_PLARec.xlsx'  # or wherever your file is saved
    # Parse the schedule sheet (list of dicts)
    schedule_results = parse_results_schedule(filepath, all_station_chainages)


    return jsonify({
        "schedule_results": schedule_results
    })


if __name__ == "__main__":
    app.run(port=5002, debug=True)
