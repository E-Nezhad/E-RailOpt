import pandas as pd
import os
import xlwt
import pyexcel

def convert_xlsx_to_xls(input_xlsx, output_xls):
    try:
        # Load the xlsx file (all sheets)
        sheets = pyexcel.get_book(file_name=input_xlsx)
        # Save as xls file
        sheets.save_as(output_xls)
        print(f"✔ Excel .xls successfully written to: {output_xls}")
    except Exception as e:
        print(f"Failed to convert .xlsx to .xls: {e}")


def generate_output_excel(assumptions, stations, stationDetails, source_path, output_path_xlsx, output_path):
    """
    Generates an Excel file with 4 sheets:
    - WaitTime: 1 train with all 0s
    - Power_train1: from existing Excel (sheet: "Energy Matrix")
    - TravelTime_train1: from existing Excel (sheet: "Time Matrix")
    - Stations: from frontend input
    """
    print(f"Starting generate_output_excel with source_path: {source_path}")
    print(xlwt.__VERSION__)


    # === Load precomputed matrices ===
    try:
        power_df = pd.read_excel(source_path, sheet_name='Energy Matrix', index_col=0)
        travel_time_df = pd.read_excel(source_path, sheet_name='Time Matrix', index_col=0)

        # === Convert travel time from seconds to hours ===
        travel_time_df = travel_time_df / 3600

        # === Generate custom station names ===
        num_stations = len(power_df.columns)
        custom_station_names = ["origin"] + \
            [f"station {i}" for i in range(1, num_stations - 1)] + \
            ["destination"]

        # === Rename columns and index for consistency ===
        power_df.columns = custom_station_names
        power_df.index = custom_station_names
        power_df.index.name = "Power (kWh)"
        travel_time_df.columns = custom_station_names
        travel_time_df.index = custom_station_names
        travel_time_df.index.name = "Travel Time (hr)"


    except Exception as e:
        print(f"Failed to read matrix data: {e}")
        return
    

    # === Create WaitTime Sheet ===
    original_names = list(power_df.columns)

    # Rename first and last columns
    station_names = ["origin"]
    station_names += [f"station {i}" for i in range(1, len(original_names) - 1)]
    station_names.append("destination")
    wait_time_values = [0] * len(station_names)
    wait_time_df = pd.DataFrame(
        [wait_time_values],
        columns=station_names,
        index=["train 1"]
    )
    wait_time_df.index.name = "(Train, Station)"


    # === Create Trains Sheet ===
    trains_df = pd.DataFrame(
    {"Number of containers": [assumptions['num_locos']]},
    index=["train 1"]
    )
    trains_df.index.name = "Train"


    # === Create Stations Sheet ===
    print(stationDetails)

    # Generate custom station names
    num_stations = len(stationDetails)
    custom_station_names = ["origin"] + \
        [f"station {i}" for i in range(1, num_stations - 1)] + \
        ["destination"]

    # Convert and rename keys to exact column names expected
    formatted_station_details = []
    for i, d in enumerate(stationDetails):
        next_station = custom_station_names[i + 1] if i + 1 < num_stations else "None"

        # Safely cast to numeric types
        setup_cost = float(d.get('setupCost', 0))
        max_charging = int(d.get('maxCharging', 0))
        max_batteries = int(d.get('maxBatteries', 0))

        formatted_station_details.append({
        "Station": custom_station_names[i],
        "Fixed Setup Cost": setup_cost,
        "Next Station": next_station,
        "Maximum Number of Chargers": max_charging,
        "Maximum Number of Batteries": max_batteries,
        })


    stations_df = pd.DataFrame(formatted_station_details)



    os.makedirs(os.path.dirname(output_path_xlsx), exist_ok=True)
    print("exists!")

    # === Write to output Excel ===
    try:
        with pd.ExcelWriter(output_path_xlsx, engine='openpyxl') as writer:
            print("wwe good")
            wait_time_df.to_excel(writer, sheet_name="WaitTime")
            trains_df.to_excel(writer, sheet_name="Trains")
            power_df.to_excel(writer, sheet_name="Power_train 1")
            travel_time_df.to_excel(writer, sheet_name="TravelTime_train 1")
            stations_df.to_excel(writer, sheet_name="Stations", index=False)

        print(f"✔ Excel successfully written to: {output_path_xlsx}")
    except Exception as e:
        print(f"❌ Failed to write .xlsx file: {e}")


      # === Convert .xlsx to .xls ===
    try:
       convert_xlsx_to_xls(output_path_xlsx, output_path)

    except Exception as e:
        print(f"Failed to convert .xlsx to .xls: {e}")
