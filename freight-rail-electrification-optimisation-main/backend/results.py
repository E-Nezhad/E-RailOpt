import pandas as pd
import math
import json
import re
import numpy as np

def parse_results_schedule(filepath, all_station_chainages):
    df = pd.read_excel(filepath, sheet_name='Results_Schedule', header=None)
    station_row = df.iloc[0]
    time_row = df.iloc[2]
    delay_row = df.iloc[3]

    # Station names with deployment status
    station_names = station_row[1:].tolist()
    times = time_row[1:].tolist()
    delays = delay_row[1:].tolist()

    results = {}

    # Identify number of containers
    container_rows = df[4:]
    container_indices = df[0][df[0].str.contains("container", na=False)].index.tolist()
    

    print(f"stations {station_names}, time {times}, delay {delays}")

    for i, name in enumerate(station_names):
        station_key = re.sub(r" \(deployed\)", "", name)
        deployed = "(deployed)" in name
        try:
            arrival_time, departure_time = eval(str(times[i]))  # e.g., (7.00, 7.55)
        except:
            arrival_time, departure_time = None, None
        delay = delays[i]

        station_data = {
            "name": station_key,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "delay": delay,
            "deployed": deployed,
            "containers": {},
            "chainage": all_station_chainages[i]
        }

        # Parse each container
        for c_idx in container_indices:
            full_label = df.iloc[c_idx, 0].strip()
            match = re.search(r'(container \d+)', full_label.lower())
            container_name = match.group(1) if match else full_label
            action_cell = str(df.iloc[c_idx + 1, i + 1]).lower()
            soc_cell = df.iloc[c_idx + 2, i + 1]

            print(f"container name {container_name}, action cell {action_cell}, soc cell {soc_cell}")

            if "charge" in action_cell:
                action = "charge"
            elif "swap" in action_cell:
                action = "swap"
            else:
                action = "none"

            try:
                # Remove '%' and safely evaluate the tuple
                soc_clean = str(soc_cell).replace("%", "")
                arrival_soc, departure_soc = eval(soc_clean)
            except:
                arrival_soc, departure_soc = None, None

            station_data["containers"][container_name] = {
                "action": action,
                "arrival_soc": arrival_soc,
                "departure_soc": departure_soc
            }

        results[station_key] = station_data
    
    print("station dataaaa")
    print(results)

    return results

