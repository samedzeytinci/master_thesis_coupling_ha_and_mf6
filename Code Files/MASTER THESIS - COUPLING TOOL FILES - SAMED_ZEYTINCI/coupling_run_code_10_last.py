# IMPORT ALL PACKAGES BEFOREHAND
import numpy as np
import math
import subprocess
import os
import pandas as pd
from pathlib import Path
from datetime import datetime


# MAKE SURE MAXBOUND VALUE IN RIVER PACKAGE INPUT FILE IS EITHER EQUAL TO THE NUMBER OF CELLS 
# WITH RIVER PACKAGE OR BIGGER THAN THAT



# BEGINNING OF THE COUPLING LOOP 


# # defining total time for hydroas
# BEST WAY WOULD BE CREATE THE INITIAL FILES AND DATA BEFORE RUNNING HYDRO AS ONCE, AND THEN JUST MAKE THE PROCESSES
# IN BETWEEN LATER ON AND WRITE THEM IN THE LOOP



# Define file paths and settings
hydro_as_path = r"C:\Hydro_As\6.0.0\CPU\hydro_as.exe"
working_directory = r"MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI"
depth_file = Path("MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/AS/depth.dat")

# File names for processing
nodes_with_cells_file = "nodes_with_cells.txt"
weights_file = "nodes_with_weights.txt"
output_idw_file = "cell_IDW_results.txt"
output_sorted_file = "cell_IDW_results_sorted.txt"

# Simulation parameters

total_time = 5000 # Total simulation time
time_step = 1000  # Interval in seconds # COUPLING FREQUENCY
current_time = 0  # Start time
run_count = 0  # Counter for number of runs
coupling_run = 0  # Counter for runs



# defining total time for hydroas

# Define the file name
file_name = "hydro_as-2d.inp"

# Define the new value for the parameter
new_value = time_step  # Example: Replace the first number with this

try:
    # Read the file
    with open(file_name, 'r') as file:
        lines = file.readlines()  # Read all lines into a list

    # Variable to track if we're in the right section
    in_target_section = False

    # Iterate through the lines
    for i, line in enumerate(lines):
        # Check if we reached the section after 'RUN CONTROL DATA SET'
        if "RUN CONTROL DATA SET" in line:
            in_target_section = True  # We've entered the section
            continue  # Move to the next line

        # If we're in the target section, find the line with 3 numbers
        if in_target_section:
            # Split the line into parts and check if it's numeric
            parts = line.split()
            if len(parts) == 3 and all(p.replace('.', '', 1).isdigit() for p in parts):
                # Replace the first value with the new value
                parts[0] = f"{new_value:.6f}"
                
                # Adjust the spacing: Add one less space before the first value
                lines[i] = f" {parts[0]}     {parts[1]}     {parts[2]}\n"
                break  # Exit the loop after replacing the value

    # Write the modified lines back to the file
    with open(file_name, 'w') as file:
        file.writelines(lines)

    print(f"Value successfully updated to {new_value} in {file_name}")

except FileNotFoundError:
    print(f"Error: The file {file_name} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")



















# MODFLOW time discretization settings

# Define values
perlen = time_step  # Period length
nstp = 10      # Number of time steps
tsmult = 1.0   # Time step multiplier

# Calculate the time range
start_time = -1.000000000000E+000
end_time = f"{(perlen + start_time):.12E}"  # Adjusted top limit to match required format

# File content
file_content = f"""# TDIS file created on 2/25/2025 by ModelMuse version 5.3.1.0.

BEGIN OPTIONS
  TIME_UNITS seconds
END OPTIONS

BEGIN DIMENSIONS
  NPER     1
END DIMENSIONS

BEGIN PERIODDATA
    {perlen:.12E}     {nstp}  {tsmult:.12E}  # perlen nstp tsmult, Stress period      1 Time = {start_time}  -  {end_time} 
END PERIODDATA
"""

# Write to file
with open("model_2_Riv_3.tdis", "w") as file:
    file.write(file_content)

print("TDIS file generated successfully.")

# Start looping until reaching total_time
while current_time < total_time:

    run_count += 1  # Increment counter for each iteration
    coupling_run += 1  

    print(f"🚀 Running HYDRO AS simulation (Run #{run_count}) for {current_time + time_step}/{total_time} sec...")

    try:
        # Run HYDRO AS
        process = subprocess.Popen(
            hydro_as_path,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream the output in real-time
        for line in process.stdout:
            print(line, end="")

        process.wait()

        # Check process success
        if process.returncode == 0:
            print(f"\n✅ Simulation Run #{run_count} completed successfully for {current_time + time_step} sec.")
        else:
            print(f"\n❌ Simulation failed with exit code {process.returncode}. Stopping process.")
            break  # Stop if an error occurs

    except FileNotFoundError:
        print("Error: Hydro_AS executable not found.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break


    # COPY THE RESULTS INTO A NEW FOLDER 

    import shutil
    import os


    # Construct new folder name
    new_folder_name = f"result_{current_time + time_step}"

    # Source and destination paths
    src_folder = r"MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI\Data-out\AS"
    dst_root = r"MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI\Data-out"
    dst_folder = os.path.join(dst_root, new_folder_name)

    # Copy the folder
    shutil.copytree(src_folder, dst_folder)

    print(f"Folder copied successfully to: {dst_folder}")





    # PART: Update depths from HYDRO AS
    print("🔄 Updating depth values...")
    depth_values = []
    found_ts = False

    with open(depth_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("TS") and str(time_step) in line:  # Match the current timestep
                found_ts = True
                continue
            if found_ts:
                try:
                    depth_values.append(float(line))
                except ValueError:
                    continue

    # Update nodes_with_cells.txt
    with open(nodes_with_cells_file, "r") as f:
        lines = f.readlines()

    header = lines[0]  # Keep the header
    data_lines = lines[1:]

    if len(data_lines) != len(depth_values):
        print("⚠ Error: Mismatch between depth values and nodes. Skipping update.")
        break

    updated_lines = [header]
    for i, line in enumerate(data_lines):
        parts = line.strip().split(", ")
        parts[2] = f"{depth_values[i]:.6f}"  # Replace Z value
        updated_lines.append(", ".join(parts) + "\n")

    with open(nodes_with_cells_file, "w") as f:
        f.writelines(updated_lines)

    print(f"✅ Depth values updated in {nodes_with_cells_file}")


  


    # PART: Compute IDW for each cell
    print("⚙️ Computing IDW Z-values...")

    # Load weights
    weights_map = {}  # (row, col) -> List of (x, y, weight)
    with open(weights_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(", ")
            x, y, row, col, weight = float(parts[0]), float(parts[1]), int(parts[3]), int(parts[4]), float(parts[6])

            if (row, col) not in weights_map:
                weights_map[(row, col)] = []
            weights_map[(row, col)].append((x, y, weight))

    # Load updated depths
    node_depths = {}  # (x, y) -> Z value
    with open(nodes_with_cells_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(", ")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            node_depths[(x, y)] = z

    # Compute IDW for each cell
    with open(output_idw_file, "w") as f:
        f.write("Row, Column, IDW_Z\n")

        for (row, col), nodes in weights_map.items():
            total_weight = sum(w for _, _, w in nodes)
            idw_z = sum(node_depths[(x, y)] * w for x, y, w in nodes) / total_weight if total_weight > 0 else 0
            f.write(f"{row}, {col}, {idw_z:.6f}\n")

    print(f"✅ IDW results saved in {output_idw_file}")

    # PART: Sort IDW results
    print("🔄 Sorting IDW results by (Row, Column)...")

    # Read and sort IDW results
    with open(output_idw_file, "r") as f:
        lines = f.readlines()

    header = lines[0]  # Keep header
    data_lines = lines[1:]  # Skip header for sorting

    # Sort by Row first, then Column
    sorted_data = sorted(data_lines, key=lambda line: (int(line.split(", ")[0]), int(line.split(", ")[1])))

    # Write sorted data to a new file
    with open(output_sorted_file, "w") as f:
        f.write(header)  # Keep header
        f.writelines(sorted_data)

    print(f"✅ Sorted IDW results saved in {output_sorted_file}")




    # Define file paths
    base_dir = Path(__file__).parent
    idw_file = base_dir / "cell_IDW_results_sorted.txt"
    top_layer_file = base_dir / "top_layer_extracted.txt"
    output_file = base_dir / "model_2_Riv_3.riv"

    # Default values
    layer = 1
    conductance = 1.000000E-03  # Constant value as per specifications
    zero_value = 0  # Default zero value

    # Step 1: Read top_layer_extracted.txt (store row, column, top level)
    top_levels = {}
    with open(top_layer_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            row, col, top_level = int(parts[0]), int(parts[1]), float(parts[2])
            top_levels[(row, col)] = top_level

    # Step 2: Read cell_IDW_results.txt (store row, column, IDW_Z values)
    riv_data = []
    with open(idw_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            row, col = int(parts[0]), int(parts[1])
            idw_z = float(parts[2])  # 3rd column is IDW_Z
            
            # Match with top_layer_extracted.txt
            if (row, col) in top_levels:
                top_level = top_levels[(row, col)]
                stage = top_level + idw_z  # Compute (top level + IDW_Z)
                riv_data.append((layer, row, col, stage, conductance, top_level, zero_value))



    output_file = "model_2_Riv_3.riv"

    # Step 3: Write to model_2_Riv.riv file in the required format
    with open(output_file, "w") as f:
        today_date = datetime.today().strftime("%m/%d/%Y")
        
        # Write header
        f.write(f"# RIV: River package file created on {today_date} by ModelMuse version 5.3.1.0.\n")
        f.write("BEGIN OPTIONS\n")
        f.write("    AUXILIARY IFACE\n")
        f.write("    BOUNDNAMES\n")
        f.write("    PRINT_INPUT\n")
        f.write("    SAVE_FLOWS\n")
        f.write("END OPTIONS\n\n")
        
        # Write dimensions
        f.write("BEGIN DIMENSIONS\n")
        f.write("  MAXBOUND  100000\n")
        f.write("END DIMENSIONS\n\n")
        
        # Write period data
        f.write("BEGIN PERIOD      1\n")
        
        for layer, row, col, stage, conductance, bottom, zero_value in riv_data:
            # Properly format the line with fixed-width spacing
            line = f"{layer:6d}{row:6d}{col:6d}  {stage:12.6E}   {conductance:12.6E}   {bottom:12.6E}      {zero_value:6d}\n"
            f.write(line)
        
        # Write END PERIOD immediately after the last row
        f.write("END PERIOD\n")


        print(f"Output file saved as: {output_file}")




# COPYING HA INPUT FILES

# COPYING HA INP FILES


# THIS INPUT FILE IS THE ONE THAT INCLUDES CURVES EXACTLY USED FOR THE TIME FRAME DEFINED

    # Original file name
    original_file = "hydro_as-2d.inp"

    # New file name based on the changing variable
    new_file = f"HA_input_file_version_{current_time}_{current_time + time_step}.lst"

    # Copy and rename the file
    shutil.copyfile(original_file, new_file)

    print(f"File copied and renamed to: {new_file}")

















# # PART 7.5 SHIFTING ZULAUF CURVES   


# # PART 7.5 SHIFTING ZULAUF CURVES


    from pathlib import Path

    # === SETTINGS ===
    input_file_path = Path("hydro_as-2d.inp")
    time_step = time_step  # Threshold in seconds
    output_txt_path = input_file_path.with_name("processed_zulauf_curves.txt")
    output_inp_path = input_file_path.with_name("hydro_as-2d.inp")

    # === STEP 1: Read file and locate ZULAUF ===
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    zulauf_index = next(i for i, line in enumerate(lines) if line.strip() == "ZULAUF")
    i = zulauf_index + 1

    # Skip blank lines after ZULAUF
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    zulauf_curve_count_line = lines[i].strip()

    if zulauf_curve_count_line == "0":
        print("⚠️  No ZULAUF curves (0 curves). Nothing will be processed.")
    else:
        num_curves = int(zulauf_curve_count_line)
        i += 1

        # === STEP 2: Extract all curves ===
        curves = []
        while i < len(lines):
            line = lines[i].strip()
            if line == "":
                i += 1
                continue
            if "AUSLAUF" in line:
                break
            try:
                header_line = lines[i].strip()
                i += 1
                n_lines = int(lines[i].strip())
                i += 1
                curve_data = []
                for _ in range(n_lines):
                    x, y = map(float, lines[i].strip().split())
                    curve_data.append([x, y])
                    i += 1
                curves.append(((header_line, n_lines), curve_data))
            except Exception:
                i += 1  # skip on error

        # === STEP 3: Shift and interpolate curves ===
        def process_curve(curve, threshold_x):
            new_curve = []
            interpolated_point = None

            for i in range(1, len(curve)):
                x1, y1 = curve[i - 1]
                x2, y2 = curve[i]
                if x1 <= threshold_x < x2:
                    # Interpolate only if y changes
                    if y1 != y2:
                        interpolated_y = y1 + (threshold_x - x1) * (y2 - y1) / (x2 - x1)
                    else:
                        interpolated_y = y1
                    interpolated_point = [0, round(interpolated_y, 6)]
                    break

            # Keep only data after threshold
            cut_data = [pt for pt in curve if pt[0] > threshold_x]

            # Shift x-values
            shifted_data = []
            for x, y in cut_data:
                shifted_x = x - threshold_x
                if shifted_x <= 0:
                    continue  # Only allow one [0, y] point
                shifted_data.append([shifted_x, y])

            final_curve = [interpolated_point] + shifted_data if interpolated_point else shifted_data
            return final_curve

        processed_curves = []
        for (header_line, _), data in curves:
            processed = process_curve(data, time_step)
            processed_curves.append((header_line, processed))

        # === STEP 4: Save processed curves to text ===
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("ZULAUF\n")
            f.write(f" {len(processed_curves)}\n\n")
            for header, curve in processed_curves:
                f.write(f"{header}\n")
                f.write(f"{len(curve)}\n")
                for x, y in curve:
                    f.write(f" {int(x)} {y:.6f}\n")
                f.write("\n")
        print(f"✅ Processed ZULAUF curves saved to: {output_txt_path.name}")

        # === STEP 5: Merge updated ZULAUF into .inp file ===
        auslauf_index = next(i for i in range(zulauf_index + 1, len(lines)) if lines[i].strip() == "AUSLAUF")
        before_zulauf = lines[:zulauf_index]
        after_auslauf = lines[auslauf_index:]

        with open(output_txt_path, "r", encoding="utf-8") as f:
            zulauf_lines = f.readlines()

        if before_zulauf and before_zulauf[-1].strip() != "":
            before_zulauf.append("\n")

        new_content = before_zulauf + zulauf_lines + after_auslauf

        with open(output_inp_path, "w", encoding="utf-8") as f:
            f.writelines(new_content)

        print(f"✅ Final .inp file updated and saved as: {output_inp_path.name}")







        # PART 8 
    #  SHIFTING RAINFALL CURVES FOR NEXT RUNS





    # Define file paths
    input_file_path = "hydro_as-2d.inp"
    output_file_path = "precipitation_series.txt"

    # Read the input file and extract relevant lines
    start_marker = "Precipitation-Series"
    end_marker = "-----------------------------------------------------\nSources at Nodes\n           0"

    # Initialize variables
    capture = False
    extracted_lines = []

    # Read the file and extract lines
    with open(input_file_path, "r", encoding="utf-8") as file:
        for line in file:
            if start_marker in line:
                capture = True  # Start capturing lines
            elif end_marker in line:
                break  # Stop capturing when the end marker is found
            elif capture:
                extracted_lines.append(line)

    # Write the extracted lines to a new file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(extracted_lines)

    # Return the output file path
    output_file_path




    # Reload and process the new file while ignoring unrelated values
    input_file_path =  "precipitation_series.txt"

    # Read file contents
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Define the shift value
    shift_value = time_step

    # Process sections while keeping unrelated values unchanged
    new_lines = []
    current_section = []

    for line in lines:
        stripped_line = line.strip()

        # If line is purely a number (unrelated value), keep it unchanged
        if stripped_line.isdigit():
            if current_section:
                # Process the previous section before adding the unrelated value
                section_data = []
                for row in current_section:
                    try:
                        x, y = map(float, row.split())
                        section_data.append([x, y])
                    except ValueError:
                        section_data.append(row)  # Keep non-numeric lines unchanged

                # Apply interpolation if necessary
                for i in range(len(section_data) - 1):
                    if isinstance(section_data[i], list) and isinstance(section_data[i + 1], list):
                        x1, y1 = section_data[i]
                        x2, y2 = section_data[i + 1]

                        if x1 < shift_value < x2:
                            y_shift = y1 + (shift_value - x1) * (y2 - y1) / (x2 - x1)
                            section_data[i][1] = y_shift

                # Add processed section
                new_lines.extend([" ".join(map(str, row)) if isinstance(row, list) else row for row in section_data])
                new_lines.append("")  # Preserve blank line

            # Add the unrelated value unchanged
            new_lines.append(stripped_line)
            current_section = []

        elif stripped_line:
            current_section.append(stripped_line)

    # Write the modified output to a new file
    output_file_path = "modified_values_to_new_interpolated.txt"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(new_lines))

    # Return the path of the modified file
    output_file_path

    # Read the extracted file
    with open(output_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Process lines to round numeric values to 2 decimal places
    processed_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                # Convert second value to float, round to 2 decimals, and format as a string
                rounded_value = f"{float(parts[1]):.6f}"
                processed_line = f"{parts[0]} {rounded_value}\n"
            except ValueError:
                # If conversion fails, keep the original line
                processed_line = line
        else:
            processed_line = line

        processed_lines.append(processed_line)

    # Write the processed lines back to the file
    processed_output_file_path = "precipitation_series_processed.txt"
    with open(processed_output_file_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(processed_lines)

    # Return the new output file path
    processed_output_file_path






    # Define file paths
    original_input_file_path = "hydro_as-2d.inp"
    updated_output_file_path = "hydro_as-2d_updated.inp"

    # Read the original input file
    with open(original_input_file_path, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    # Read the processed precipitation series data
    with open(processed_output_file_path, "r", encoding="utf-8") as file:
        precipitation_series_lines = file.readlines()

    # Identify the start marker
    start_marker = "Precipitation-Series"

    # Find the index where "Precipitation-Series" appears
    start_index = None
    for i, line in enumerate(original_lines):
        if start_marker in line:
            start_index = i
            break

    # If the start marker is found, keep only lines up to that point and add the new data
    if start_index is not None:
        updated_lines = original_lines[:start_index + 1] + precipitation_series_lines

        # Write the updated content to a new file
        with open(updated_output_file_path, "w", encoding="utf-8") as output_file:
            output_file.writelines(updated_lines)

        # Return the updated file path
        updated_output_file_path
    else:
        updated_output_file_path = None

    updated_output_file_path






    # Read the updated file
    with open(updated_output_file_path, "r", encoding="utf-8") as file:
        updated_lines = file.readlines()

    # Process the file to remove any space between "Sources at Nodes" and "0"
    processed_lines = []
    skip_next = False

    for i, line in enumerate(updated_lines):
        if "Sources at Nodes" in line and i < len(updated_lines) - 1:
            next_line = updated_lines[i + 1].strip()
            if next_line == "0":
                processed_lines.append(line.strip() + "\n")
                processed_lines.append("0\n")
                skip_next = True  # Skip the next line since it's being handled here
            else:
                processed_lines.append(line)
        elif not skip_next:
            processed_lines.append(line)
        else:
            skip_next = False  # Reset the flag after skipping one line

    # Save the final cleaned-up file
    final_output_file_path = "hydro_as-2d_final.inp"
    with open(final_output_file_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(processed_lines)

    # Return the final updated file path
    final_output_file_path





    # Load the file and modify the specific section
    file_path = "hydro_as-2d_final.inp"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace the unwanted format with the correct one
    content = content.replace(
        "-----------------------------------------------------\nSources at Nodes\n\n0",
        "-----------------------------------------------------\nSources at Nodes\n0"
    )

    # Overwrite the file with the corrected content
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    # Return the updated file path
    file_path






    import numpy as np

    # Define file paths
    inp_file_path = "hydro_as-2d_final.inp"
    output_file_path = "hydro_as-2d_adjusted.inp"

    # Read the input file
    with open(inp_file_path, "r", encoding="ISO-8859-1") as file:
        lines = file.readlines()

    # Define shift time
    shift_time = shift_value

    # Identify curves after "Precipitation-Series"
    processing_curves = False
    curve_lines = []
    header_lines = []
    sources_at_nodes_lines = []
    temp_storage = []
    found_precipitation_series = False

    # Extract the first three lines after "Precipitation-Series" from the original file
    precipitation_series_index = None
    for i, line in enumerate(lines):
        if "Precipitation-Series" in line:
            precipitation_series_index = i
            break

    extracted_lines = lines[precipitation_series_index + 1 : precipitation_series_index + 4] if precipitation_series_index is not None else []

    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line == "Precipitation-Series":
            if found_precipitation_series:
                continue  # Remove duplicate "Precipitation-Series"
            found_precipitation_series = True
            processing_curves = True  # Start collecting curves from this point
            header_lines.append(line)  # Keep the header
            # Add extracted lines after "Precipitation-Series"
            header_lines.extend(extracted_lines)
            continue

        if not processing_curves:
            header_lines.append(line)  # Store header part
        elif "Sources at Nodes" in stripped_line or stripped_line == "0" or "-" in stripped_line:
            sources_at_nodes_lines.append(line)  # Collect these lines separately
        else:
            temp_storage.append(line)

    # Process and shift curves
    shifted_curve_lines = []
    current_curve = []

    for line in temp_storage:
        stripped_line = line.strip()
        if stripped_line == "":
            continue  # Skip empty lines

        # Try parsing time-value pairs
        try:
            time, value = map(float, stripped_line.split())
            time -= shift_time  # Apply shifting
            time = max(time, 0)  # Ensure non-negative times
            current_curve.append(f"{time:.1f} {value:.6f}\n")
        except ValueError:
            if current_curve:
                shifted_curve_lines.extend(current_curve)
                current_curve = []
            shifted_curve_lines.append(line)  # Keep non-time-value lines

    if current_curve:
        shifted_curve_lines.extend(current_curve)

    # Combine everything and append "Sources at Nodes" at the end
    final_lines = header_lines + shifted_curve_lines + sources_at_nodes_lines

    # Write the adjusted output file
    with open(output_file_path, "w", encoding="ISO-8859-1") as file:
        file.writelines(final_lines)

    print("File adjusted and saved successfully!")


    import os

    # Define file paths
    file_path = "hydro_as-2d_adjusted.inp"

    # Read file content
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Identify the start of precipitation curves
    start_index = None
    for i, line in enumerate(lines):
        if "Precipitation-Series" in line:
            start_index = i
            break

    # Ensure "Precipitation-Series" line is retained
    if start_index is not None:
        processed_lines = lines[:start_index + 1]  # Include the "Precipitation-Series" line
        relevant_lines = lines[start_index + 1:]  # Process the following lines
    else:
        processed_lines = []
        relevant_lines = lines  # If the section is not found, process the entire file

    # Processing the precipitation curves
    current_curve = []
    curve_started = False

    for line in relevant_lines:
        stripped_line = line.strip()

        if stripped_line == "":
            continue

        # Identify new curve start
        if stripped_line.isdigit():
            if current_curve:
                # Process and add the cleaned curve
                non_zero_found = False
                latest_zero_rows = []
                cleaned_curve = []
                
                for row in current_curve:
                    try:
                        value = float(row.split()[0])  # Check first column
                    except ValueError:
                        # If conversion fails, keep the line as it is (header or separator)
                        cleaned_curve.append(row)
                        continue

                    if value == 0.0 and not non_zero_found:
                        latest_zero_rows.append(row)  # Store latest zero rows
                    elif value != 0.0:
                        non_zero_found = True
                        if latest_zero_rows:
                            cleaned_curve.extend(latest_zero_rows[-1:])  # Keep at least two zero rows
                            latest_zero_rows = []  # Reset after adding
                        cleaned_curve.append(row)
                    elif non_zero_found:
                        cleaned_curve.append(row)  # Keep remaining non-zero values

                # Ensure at least 2 rows exist in the curve
                while len(cleaned_curve) < 1:
                    cleaned_curve.append("0 0.0\n")  # Append 0 rows if needed

                processed_lines.extend(cleaned_curve)

            # Start new curve
            processed_lines.append(line)
            current_curve = []
            curve_started = True

        elif curve_started:
            current_curve.append(line)

    # Save the updated data back into the original input file, keeping "Precipitation-Series"
    with open(file_path, "w", encoding="utf-8") as updated_file:
        updated_file.writelines(processed_lines)

    print("File corrected successfully to maintain at least 2 rows per curve without extra spaces.")










    import pandas as pd

    # Load the file and read its content
    file_path = "hydro_as-2d_adjusted.inp"

    # Read the file as a list of lines
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Identify where the precipitation curves start
    precipitation_start = None
    for i, line in enumerate(lines):
        if "Precipitation-Series" in line:
            precipitation_start = i
            break

    # Process precipitation curves and correct row counts
    if precipitation_start is not None:
        i = precipitation_start + 2  # Skip the header line
        while i < len(lines):
            if lines[i].strip().isdigit():  # Identify curve number
                curve_number_index = i
                num_rows_index = i + 1

                # Ensure there is a number of rows entry
                if num_rows_index >= len(lines) or not lines[num_rows_index].strip().isdigit():
                    break

                num_rows_actual = 0
                j = num_rows_index + 1

                # Count actual number of rows in the curve
                while j < len(lines) and not lines[j].strip().isdigit():
                    if lines[j].strip():  # Ensure the line is not empty
                        num_rows_actual += 1
                    j += 1

                # Update the number of rows if necessary
                lines[num_rows_index] = f"           {num_rows_actual}\n"

                # Move to the next curve
                i = j
            else:
                break

    # Overwrite the original file with the corrected content
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("File updated successfully.")






    # last STEP TO REMOVE UNWANTED LINES




    file_path = "hydro_as-2d_adjusted.inp"

    # Read file content
    with open(file_path, "r") as file:
        lines = file.readlines()



    # Adjustments
    # 1. Remove the last two rows (12 and 2) after "Precipitation-Series"
    precipitation_index = None
    for i, line in enumerate(lines):
        if "Precipitation-Series" in line:
            precipitation_index = i

    if precipitation_index is not None:
        # Ensure we have at least 5 lines after "Precipitation-Series" to modify
        if len(lines) > precipitation_index + 5:
            del lines[precipitation_index + 3: precipitation_index + 5]

    # 2. Modify the last curve by removing the last "0" before the dashed line
    dash_index = None
    for i, line in enumerate(lines):
        if "-----------------------------------------------------" in line:
            dash_index = i
            break

    if dash_index is not None:
        # Remove the last "0" before the dashed line
        for i in range(dash_index - 1, -1, -1):
            if lines[i].strip() == "0":
                del lines[i]
                break

    # Save the adjusted file
    adjusted_file_path = "hydro_as-2d_adjusted.inp"
    with open(adjusted_file_path, "w") as file:
        file.writelines(lines)

    # Provide the modified file
    adjusted_file_path




    # Load the file and delete the line two lines before "Sources at Nodes"

    file_path = "hydro_as-2d_adjusted.inp"
    # Read file content
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find "Sources at Nodes" and remove the line two lines before it
    sources_index = None
    for i, line in enumerate(lines):
        if "Sources at Nodes" in line:
            sources_index = i
            break

    if sources_index is not None and sources_index >= 2:
        del lines[sources_index - 2]  # Delete the line two lines before "Sources at Nodes"

    # Save the adjusted file
    adjusted_file_path = "hydro_as-2d_adjusted.inp"
    with open(adjusted_file_path, "w") as file:
        file.writelines(lines)

    # Provide the modified file
    adjusted_file_path










    # Define file paths
    original_file_path = "hydro_as-2d.inp"
    adjusted_file_path = "hydro_as-2d_adjusted.inp"
    output_file_path = "hydro_as-2d_merged_updated.inp"

    # Read the original file
    with open(original_file_path, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    # Read the adjusted file
    with open(adjusted_file_path, "r", encoding="utf-8") as file:
        adjusted_lines = file.readlines()

    # Find the index of "nodeniederschlag" in both files
    nodeniederschlag_index_original = next(
        (i for i, line in enumerate(original_lines) if "nodeniederschlag" in line.lower()), None
    )

    nodeniederschlag_index_adjusted = next(
        (i for i, line in enumerate(adjusted_lines) if "nodeniederschlag" in line.lower()), None
    )

    # Extract lines before "nodeniederschlag" from the original file
    lines_before_nodeniederschlag = original_lines[:nodeniederschlag_index_original]

    # Extract lines from "nodeniederschlag" onwards from the adjusted file
    lines_after_nodeniederschlag = adjusted_lines[nodeniederschlag_index_adjusted:]

    # Combine both parts
    final_lines = lines_before_nodeniederschlag + lines_after_nodeniederschlag

    # Save the modified file
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.writelines(final_lines)

    print(f"Modified file saved at: {output_file_path}")


    import os

    # Define file paths
    old_file = "hydro_as-2d.inp"
    new_file = "hydro_as-2d_merged_updated.inp"
    renamed_file = "hydro_as-2d.inp"

    # Delete the old file if it exists
    if os.path.exists(old_file):
        os.remove(old_file)
        print(f"Deleted: {old_file}")
    else:
        print(f"File not found: {old_file}")

    # Rename the new file to the old file's name
    if os.path.exists(new_file):
        os.rename(new_file, renamed_file)
        print(f"Renamed {new_file} to {renamed_file}")
    else:
        print(f"File not found: {new_file}")











    # Define the MODFLOW 6 executable name
    mf6_exe = "mf6.exe"

    # Define the workspace where the MODFLOW 6 input files are located
    workspace = r"MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI"
    # Check if the executable exists
    if not os.path.isfile(mf6_exe):
        raise FileNotFoundError(f"{mf6_exe} not found in the current directory.")


    # Run the MODFLOW 6 executable and capture output
    try:
        process = subprocess.Popen(
            [mf6_exe],  # Command to run
            cwd=workspace,  # Directory where the command is run
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture errors
            text=True,  # Output as strings instead of bytes
        )

        # Print the stdout and stderr output in real time
        print("Running MODFLOW 6...\n")
        for line in iter(process.stdout.readline, ""):
            print(line, end="")  # Print standard output lines

        for err_line in iter(process.stderr.readline, ""):
            print("ERROR:", err_line, end="")  # Print error output lines

        process.wait()  # Wait for the process to complete
        if process.returncode == 0:
            print("\nSimulation completed successfully!")
        else:
            print(f"\nSimulation failed with return code {process.returncode}.")
    except Exception as e:
        print(f"An error occurred: {e}")



# PART 6 CREATING HEAD CHANGES FILE and WTIEFE_0

    import re
    import pandas as pd

    
    time_step_mf = time_step / nstp


    # Load the file
    file_path = "model_2_Riv_3.lst"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    data = []
    current_period = None
    current_step = None
    inside_block = False

    # Define regex patterns
    header_pattern = re.compile(r"RIV PACKAGE \(RIV-1\) FLOW RATES\s+PERIOD\s+(\d+)\s+STEP\s+(\d+)")
    data_pattern = re.compile(r"\s*\d+\s+\((\d+),(\d+),(\d+)\)\s+([\d.Ee+-]+)")
    end_block_pattern = re.compile(r"-{20,}")

    # Parse the file
    for line in lines:
        header_match = header_pattern.search(line)
        if header_match:
            current_period = int(header_match.group(1))
            current_step = int(header_match.group(2))
            inside_block = True
            continue

        if inside_block:
            if end_block_pattern.match(line):
                inside_block = False
                continue
            data_match = data_pattern.match(line)
            if data_match:
                layer = int(data_match.group(1))
                row = int(data_match.group(2))
                col = int(data_match.group(3))
                rate = float(data_match.group(4))
                data.append([current_period, current_step, layer, row, col, rate])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Period", "Step", "Layer", "Row", "Column", "Rate"])

    # Filter to only include steps 1 to 10 of period 1
    df_filtered = df[(df["Period"] == 1) & (df["Step"] >= 1) & (df["Step"] <= nstp)]

    # Save to CSV
    csv_output_path = "riv_flow_rates_period1_steps1to10.csv"
    df_filtered.to_csv(csv_output_path, index=False)

    csv_output_path
















    import pandas as pd
    import numpy as np


    # Paths to your files
    riv_csv_path = "riv_flow_rates_period1_steps1to10.csv"
    cell_coords_path = "cell_coordinates_with_corners.txt"
    node_data_path = "processed_data.txt"
    output_csv_path = "riv_flow_rates_with_node_areas.csv"

    # 1. Load RIV cell data
    df_riv = pd.read_csv(riv_csv_path)

    # 2. Load cell coordinates with corners
    df_corners = pd.read_csv(cell_coords_path)
    df_corners.columns = [col.strip().replace(" ", "") for col in df_corners.columns]
    df_corners = df_corners.rename(columns={"Row": "Row", "Column": "Column"})

    # Ensure coordinate columns are floats
    float_columns = ["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"]
    for col in float_columns:
        df_corners[col] = pd.to_numeric(df_corners[col], errors="coerce")

    # 3. Load node coordinates and area
    df_processed = pd.read_csv(node_data_path, delim_whitespace=True)

    # 4. Precompute bounding boxes for each cell
    df_corners["X_min"] = df_corners[["X1", "X2", "X3", "X4"]].min(axis=1)
    df_corners["X_max"] = df_corners[["X1", "X2", "X3", "X4"]].max(axis=1)
    df_corners["Y_min"] = df_corners[["Y1", "Y2", "Y3", "Y4"]].min(axis=1)
    df_corners["Y_max"] = df_corners[["Y1", "Y2", "Y3", "Y4"]].max(axis=1)

    # Create a lookup dictionary
    cell_bounds = {
        (row["Row"], row["Column"]): (row["X_min"], row["X_max"], row["Y_min"], row["Y_max"])
        for _, row in df_corners.iterrows()
    }

    # 5. Function to calculate total area of nodes inside a cell's bounding box
    def sum_areas_in_box(cell_row, cell_col):
        bounds = cell_bounds.get((cell_row, cell_col))
        if bounds is None:
            return np.nan
        x_min, x_max, y_min, y_max = bounds
        in_box = df_processed[
            (df_processed["X"] > x_min) & (df_processed["X"] < x_max) &
            (df_processed["Y"] > y_min) & (df_processed["Y"] < y_max)
        ]
        return in_box["area"].sum()

    # 6. Apply to each RIV cell
    df_riv["Total_Node_Area"] = df_riv.apply(lambda row: sum_areas_in_box(row["Row"], row["Column"]), axis=1)

    # 7. Save to CSV
    df_riv.to_csv(output_csv_path, index=False)

    print(f"Saved output to: {output_csv_path}")





    # CALCULATING DELTA Y FOR EACH CELL





    import pandas as pd

    # Path to the original CSV file
    file_path = "riv_flow_rates_with_node_areas.csv"

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Create a new column 'Delta Y' by dividing 6th column by 7th column and multiplying by -100
    # Note: Python uses 0-based indexing, so 6th column is index 5, 7th is index 6
    df['Delta Y'] = (df.iloc[:, 5] / df.iloc[:, 6]) * time_step_mf

    # Overwrite the original file with the new data
    df.to_csv(file_path, index=False)
















    # # TAKING NODE DATA WITH THEIR ROW COLUMN DATA

    import pandas as pd

    # File names
    csv_file = "riv_flow_rates_with_node_areas.csv"
    nodes_file = "nodes_with_weights.txt"
    output_file = "final_merged_riv_and_nodes.csv"

    # Step 1: Read the original CSV file with proper headers
    csv_df = pd.read_csv(csv_file)

    # Step 2: Read the node file and skip the first row (it's not real data)
    nodes_df = pd.read_csv(nodes_file, delim_whitespace=True, header=None, skiprows=1)

    # Step 3: Select only columns: Node_X, Node_Y, Row, Column (index 0,1,3,4)
    selected_nodes = nodes_df.iloc[:, [0, 1, 3, 4]]

    # Step 4: Rename columns
    selected_nodes.columns = ['Node_X', 'Node_Y', 'Node_Row', 'Node_Column']

    # Step 5: Merge and clean trailing commas if any
    merged_df = pd.concat([csv_df, selected_nodes], axis=1)
    merged_df = merged_df.applymap(lambda x: str(x).rstrip(',') if isinstance(x, str) else x)

    # Step 6: Save the cleaned merged file
    merged_df.to_csv(output_file, index=False)







    # # FINDING DELTA Y SUM FOR NODES






    # Load the CSV file
    df = pd.read_csv("final_merged_riv_and_nodes.csv")

    # Ensure proper column names for reference
    df.columns = df.columns.str.strip()

    # Assume column 4 is 'Row', column 5 is 'Column', column 8 is 'Delta_Y',
    # columns 11 and 12 are 'Match_Row', 'Match_Col', and column 13 is 'Delta_Y_Sum'

    # Add 'Delta_Y_Sum' column if it doesn't exist
    if 'Delta_Y_Sum' not in df.columns:
        df['Delta_Y_Sum'] = 0.0

    # Iterate over all rows and sum Delta_Y where (Match_Row, Match_Col) matches (Row, Column)
    for idx, row in df.iterrows():
        match_row = row.iloc[10]  # 11th column
        match_col = row.iloc[11]  # 12th column
        mask = (df.iloc[:, 3] == match_row) & (df.iloc[:, 4] == match_col)  # columns 4 and 5 are Row, Column
        matched_deltas = df.loc[mask, df.columns[7]]  # 8th column: Delta_Y
        df.at[idx, 'Delta_Y_Sum'] = matched_deltas.sum()

    # Save updated file
    output_path = "riv_flow_rates_with_node_areas_updated.csv"
    df.to_csv(output_path, index=False)


    output_path


    # TAKE LATEST DEPTHS OF PREV HA RUN AND PREPARE FOR WTIEFE_0

    depth_file_path = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/AS/depth.dat"



    import pandas as pd

    # Step 1: Read depth.dat and extract the values after the last 'TS'
    depth_file_path = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/AS/depth.dat"
    with open(depth_file_path, 'r') as file:
        lines = file.readlines()

    # Find last TS line
    ts_indices = [i for i, line in enumerate(lines) if line.strip().startswith('TS')]
    latest_ts_index = ts_indices[-1] if ts_indices else None

    # Extract depth values after the last TS
    depth_values_lines = lines[latest_ts_index + 1:] if latest_ts_index is not None else []
    depth_last_step_values = [float(line.strip()) for line in depth_values_lines if line.strip()]

    # Step 2: Load the CSV file
    csv_file_path = "riv_flow_rates_with_node_areas_updated.csv"
    df = pd.read_csv(csv_file_path)

    # Step 3: Truncate the DataFrame to match depth values
    df_truncated = df.iloc[:len(depth_last_step_values)].copy()

    # Step 4: Add the 'depth last step' column
    df_truncated['depth last step'] = depth_last_step_values

    # Step 5: Compute the 'wtiefe_0' column
    df_truncated['wtiefe_0'] = df_truncated['depth last step'] - df_truncated['Delta_Y_Sum']

    # Step 6: Create output DataFrame with only 3 columns
    output_df = df_truncated[['Delta_Y_Sum', 'depth last step', 'wtiefe_0']]

    # Step 7: Save to new CSV file
    output_file_path = f"depth_wtiefe_output_for_HA_run_{current_time + time_step}.csv"
    output_df.to_csv(output_file_path, index=False)

    print(f"Saved output to: {output_file_path}")




    # PREPARING WTIEFE_0.dat file 

    # Try again now that the original wtiefe_0.dat file is uploaded
    wtiefe_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/wtiefe_0.dat"
    csv_input_file = output_file_path
    output_wtiefe_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/wtiefe_0.dat"

    # Step 1: Extract header lines from original wtiefe_0.dat
    header_lines = []
    with open(wtiefe_file, "r") as f:
        for line in f:
            header_lines.append(line.strip())
            if line.strip().startswith("TS 0 0"):
                break  # Stop after TS 0 0 line

    # Step 2: Load the CSV and extract only the 3rd column (wtiefe_0)
    df = pd.read_csv(csv_input_file)
    wtiefe_values = df.iloc[:, 2].astype(str).tolist()  # 3rd column (index 2)

    # Step 3: Write the updated wtiefe_0.dat file
    with open(output_wtiefe_file, "w") as f:
        f.write("\n".join(header_lines) + "\n")       # Write the header lines
        f.write("\n".join(wtiefe_values) + "\n")      # Write the 3rd column values
        f.write("ENDDS\n")                            # Append ENDDS at the end

    output_wtiefe_file




############# END OF PART 6 #############





# # PART 7 CHANGING INITIAL AQUIFER HEADS AFTER THE RUN FOR NEXT MODFLOW RUN 




    # # PART WITH DIRECTLY APPLYING THOSE VALUES

    import os

    # Define file paths
    lst_file_path = "model_2_Riv_3.lst"
    ic_file_path = "model_2_Riv_3.ic"
    output_ic_file_path = "model_2_Riv_3.ic"

    # Read the .lst file content
    with open(lst_file_path, "r") as file:
        lst_content = file.readlines()

    # Locate the start and end of the relevant section in the .lst file
    start_index = None
    end_index = None

    for i, line in enumerate(lst_content):
        if "HEAD IN LAYER   1 AT END OF TIME STEP  10 IN STRESS PERIOD    1" in line:
            start_index = i + 1  # Move to the next line
        if "HEAD WILL BE SAVED ON UNIT 1014 AT END OF TIME STEP   10, STRESS PERIOD    1" in line:
            end_index = i
            break

    # Ensure both start and end indices were found
    if start_index is None or end_index is None:
        raise ValueError("Could not find the required section in the .lst file.")

    # Extract the relevant section
    head_values_section = lst_content[start_index:end_index]

    # Process extracted values into a structured format
    heads_list = []
    row_counter = 1
    col_counter = 1
    ncols = 128  # Define number of columns per row

    for line in head_values_section:
        values = line.strip().split()

        # Ignore lines that contain only dashes, dots, or headers
        if not values or all(c in "-." for c in values[0]):
            continue

        # Check if the first value is a row number and remove it
        if values[0].isdigit():
            values = values[1:]

        # Store the values with row and column numbers
        for value in values:
            heads_list.append((row_counter, col_counter, float(value)))
            col_counter += 1
            if col_counter > ncols:
                col_counter = 1
                row_counter += 1

    # Read the original .ic file
    with open(ic_file_path, "r") as f:
        ic_lines = f.readlines()

    # Identify the section with numerical values
    start_idx = None
    for i, line in enumerate(ic_lines):
        if "STRT LAYERED" in line:
            start_idx = i + 2  # Move to the data row
            break

    if start_idx is None:
        raise ValueError("Could not locate the numerical values section in the .ic file.")

    # Process the numerical values while keeping the structure and spacing
    processed_lines = ic_lines[:start_idx]  # Preserve headers

    current_row = 1
    current_col = 1
    new_heads_dict = {(row, col): value for row, col, value in heads_list}

    for i in range(start_idx, len(ic_lines)):
        line = ic_lines[i]
        values = line.strip().split()

        if not values:  # Preserve empty lines
            processed_lines.append("\n")
            continue

        new_values = []
        for val in values:
            if (current_row, current_col) in new_heads_dict:
                formatted_value = f"{new_heads_dict[(current_row, current_col)]:.12E}"  # Ensure scientific notation
            else:
                try:
                    formatted_value = f"{float(val):.12E}"  # Convert and format existing value
                except ValueError:
                    formatted_value = val  # Keep non-numeric values as they are

            new_values.append(formatted_value)

            # Move to the next column/row
            current_col += 1
            if current_col > ncols:
                current_col = 1
                current_row += 1

        # Format new line with proper spacing
        formatted_line = "        " + "   ".join(new_values) + "\n"
        processed_lines.append(formatted_line)

    # Save the modified .ic file
    with open(output_ic_file_path, "w") as f:
        f.writelines(processed_lines)

    print(f"Updated IC file saved as: {output_ic_file_path}")


# END FOR PART WITH DIRECTLY APPLYING WITH 1E30 VALUES 
######################################################
######################################################

# PART 8.5 COPYING MODFLOW OUTPUT LST FILE


# PART 8.5 COPYING MODFLOW OUTPUT LST FILE


    # Example value that changes in each run

    # Original file name
    original_file = "model_2_Riv_3.lst"

    # New file name based on the changing variable
    new_file = f"GW_results_for_HA_run_{current_time + time_step}.lst"

    # Copy and rename the file
    shutil.copyfile(original_file, new_file)

    print(f"File copied and renamed to: {new_file}")



# END OF COPYING MODFLOW LST





















    




############################################################################################

    # PART 9
    # ADDING geschw_0 for next runs


    import chardet

    # Define file paths
    veloc_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/AS/veloc.dat"
    nodeniederschlag_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/nodeniederschlag.dat"
    geschw_0_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/geschw_0.dat"

    # Read nodeniederschlag.dat to extract ND and NC values
    nd_value = None
    nc_value = None

    with open(nodeniederschlag_file, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[0] == "ND":
                nd_value = int(parts[1])
            elif parts[0] == "NC":
                nc_value = int(parts[1])
            if nd_value is not None and nc_value is not None:
                break  # Stop reading once both values are found

    # Ensure nd_value and nc_value are extracted correctly
    if nd_value is None or nc_value is None:
        raise ValueError("ND or NC value could not be determined from nodeniederschlag.dat")

    # Read the veloc.dat file to extract TS values and velocities
    with open(veloc_file, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    # Find the last TS line and extract values after it
    ts_indices = [i for i, line in enumerate(lines) if line.startswith("TS")]
    latest_ts_value = None
    velocity_values = []

    if ts_indices:
        last_ts_index = ts_indices[-1]  # Last TS line index
        latest_ts_value = int(lines[last_ts_index].split()[1])  # Get TS value
        velocity_values = lines[last_ts_index + 1 : last_ts_index + 1 + nd_value]  # Extract values for ND count

    # Ensure the extracted velocity values match the expected ND count
    if len(velocity_values) != nd_value:
        raise ValueError(f"Mismatch in velocity values count: Expected {nd_value}, got {len(velocity_values)}")

    # Format velocities properly to match the required spacing
    def format_velocity_line(line):
        values = line.split()
        if len(values) < 2:
            return ""  # Skip if fewer than 2 values
        return f"{float(values[0]):>9f}          {float(values[1]):>10.5f}\n"

    velocity_values = [format_velocity_line(line) for line in velocity_values if format_velocity_line(line)]

    # Write the Geschw_0.dat file
    with open(geschw_0_file, "w", encoding="utf-8") as file:
        file.write("DATASET\n")
        file.write('OBJTYPE "mesh2d"\n')
        file.write("BEGVEC\n")
        file.write(f"ND {nd_value}\n")
        file.write(f"NC {nc_value}\n")
        file.write('NAME "Geschw_0"\n')
        file.write("TIMEUNITS seconds\n")
        file.write(f"TS 0 {latest_ts_value}\n")
        file.writelines(velocity_values)
        file.write("ENDDS\n")  # Append ENDDS at the last line

    print(f"Successfully created {geschw_0_file} with ND={nd_value}, NC={nc_value}, TS={latest_ts_value}")





# PART 10 - UPDATING SOURCES_IN TIME INTERVALS AND SHIFTING 



    from pathlib import Path

    def process_sources_file(file_path, output_path, time_step):
        # Read original lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract header before first TS line
        first_ts_index = next(i for i, line in enumerate(lines) if line.startswith("TS "))
        header = lines[:first_ts_index]

        # Split into TS blocks
        ts_blocks = []
        block = []
        for line in lines[first_ts_index:]:
            if line.startswith("TS "):
                if block:
                    ts_blocks.append(block)
                block = [line]
            else:
                block.append(line)
        if block:
            ts_blocks.append(block)

        # Adjust TS values and prepare result
        result_blocks = []
        keep_previous_ts0 = True
        ts0_index = None

        for i, block in enumerate(ts_blocks):
            ts_line = block[0]
            try:
                original_ts = int(ts_line.split()[1])
            except (IndexError, ValueError):
                continue

            # First block is assumed to be TS 0
            if i == 0:
                result_blocks.append(block)
                ts0_index = 0
                continue

            new_ts = original_ts - time_step
            if new_ts <= 0:
                new_ts = 0
                if keep_previous_ts0:
                    # Remove all blocks before this one (including first TS 0)
                    result_blocks = [[f"TS 0\n"] + block[1:]]
                    keep_previous_ts0 = False
                else:
                    result_blocks.append([f"TS 0\n"] + block[1:])
            else:
                result_blocks.append([f"TS {new_ts}\n"] + block[1:])

        # Save to file
        with open(output_path, "w") as out_file:
            out_file.writelines(header + [line for block in result_blocks for line in block])

        print(f"✅ Processed file saved to: {output_path}")


    # ==== Run the function ====
    # Define your paths and step
    input_file = Path("MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/sources-in.dat")  # Replace with your actual file path
    output_file = Path("MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-in/AS/sources-in.dat")
    time_step = time_step

    process_sources_file(input_file, output_file, time_step)




# END OF UPDATING ALL THE FILES 
#####################################
#####################################
#####################################
#####################################
#####################################






    ## **STEP 5: Update Time Counter**
    current_time += time_step
    print(f"🕒 Time Updated: {current_time}/{total_time} sec\n")

print(f"✅ **Simulation Completed: {coupling_run} runs, {total_time} sec**")





print(" Post Processing of Results starts")

# PART 11

# POST-PROCESSING
#################
#################


# CHANGE THE PARAMETER initial_time_step TO THE COUPLING FREQUENCY YOU CHOSE


import os
import numpy as np

# === PARAMETERS ===
initial_time_step = 1000
max_time_step = total_time
results_dir_merge = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out"
results_dir_max = results_dir_merge

# === FUNCTION TO EXTRACT ND LINE FROM depth.dat ===
def extract_nd_line(results_dir):
    for folder in sorted(os.listdir(results_dir)):
        if folder.startswith("result_"):
            depth_file = os.path.join(results_dir, folder, "depth.dat")
            if os.path.exists(depth_file):
                with open(depth_file, "r") as f:
                    for line in f:
                        if line.strip().startswith("ND"):
                            return line
    raise ValueError("No 'ND' line found in any depth.dat file.")

nd_line = extract_nd_line(results_dir_merge)


# === FUNCTION TO MERGE TIME-STEP FILES (depth, veloc, wspl) ===
def process_file_type(file_name, output_name):
    time_steps = list(range(initial_time_step, max_time_step + 1, initial_time_step))
    all_blocks = []

    for ts in time_steps:
        folder_path = os.path.join(results_dir_merge, f"result_{ts}")
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()

            block = []
            current = []
            for line in lines:
                if line.strip().startswith("TS"):
                    if current:
                        block.append(current)
                        current = []
                    current.append(line)
                else:
                    current.append(line)
            if current:
                block.append(current)
            all_blocks.append(block)

    new_lines = []

    for i, blocks in enumerate(all_blocks):
        ts_offset = i * initial_time_step

        for j, block in enumerate(blocks):
            block_lines = block.copy()

            if i == 0:
                new_lines.extend(block_lines)
            else:
                if j == 0:
                    new_ts = ts_offset + 10
                    scalar_index = next((k for k, line in enumerate(block_lines) if line.strip().startswith("SCALAR")), None)
                    if scalar_index is not None:
                        block_lines = block_lines[scalar_index:]
                        for k in range(len(block_lines)):
                            if block_lines[k].strip().startswith("TS"):
                                block_lines[k] = f"TS {new_ts}\n"
                                break
                    new_lines.extend(block_lines)
                elif block_lines[0].strip() == f"TS {ts_offset}":
                    continue
                else:
                    old_ts = int(block_lines[0].split()[1])
                    new_ts = old_ts + ts_offset
                    block_lines[0] = f"TS {new_ts}\n"
                    new_lines.extend(block_lines)

    # === Post-process to rename second TS XXXX lines ===
    important_ts_vals = list(range(initial_time_step, max_time_step, initial_time_step))
    ts_seen = {}
    for i in range(len(new_lines)):
        if new_lines[i].strip().startswith("TS"):
            ts_val = int(new_lines[i].split()[1])
            if ts_val in important_ts_vals:
                if ts_val not in ts_seen:
                    ts_seen[ts_val] = 1
                else:
                    corrected_ts = ts_val + 10
                    new_lines[i] = f"TS {corrected_ts}\n"

    with open(output_name, "w") as f:
        f.writelines(new_lines)

    print(f"✅ Created: {output_name}")


# === MERGE ALL 3 DATA TYPES ===
process_file_type("depth.dat", "depth_data_merged.dat")
process_file_type("veloc.dat", "veloc_data_merged.dat")
process_file_type("wspl.dat", "wspl_data_merged.dat")


# END OF MERGING RESULTS


# CREATING MAX OF ALL TIMES

import os
import numpy as np

# === PARAMETERS ===
results_dir = results_dir_merge  # Folder containing result_XXXX folders
ts_value = max_time_step  # Final TS to write in the header

# === Function to get ND value from a file ===
def get_nd_value(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith("ND"):
                return int(line.strip().split()[1])
    raise ValueError(f"ND value not found in file: {file_path}")

# === Function to process max values from single-column files (depth_max, wspl_max) ===
def process_scalar_max(file_name, output_name, scalar_header):
    result_folders = sorted([f for f in os.listdir(results_dir) if f.startswith("result_")])
    blocks = []

    for folder in result_folders:
        path = os.path.join(results_dir, folder, file_name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            block = []
            for line in lines:
                if line.strip().startswith("TS"):
                    if block:
                        blocks.append(block)
                    block = []
                else:
                    block.append(line)
            if block:
                blocks.append(block)

    parsed_blocks = []
    for block in blocks:
        data = []
        for line in block:
            try:
                val = float(line.strip().split()[0])
                data.append(val)
            except (ValueError, IndexError):
                continue
        if data:
            parsed_blocks.append(np.array(data))

    min_length = min(len(b) for b in parsed_blocks)
    trimmed = [b[:min_length] for b in parsed_blocks]
    max_values = np.max(np.stack(trimmed), axis=0)

    nd_val = get_nd_value(os.path.join(results_dir, result_folders[0], file_name))

    with open(output_name, 'w') as out:
        out.writelines([line for line in scalar_header])
        out.write(f"ND     {nd_val}\n")
        out.writelines(["ST  0\n", "TIMEUNITS Seconds\n", f"TS         {ts_value}\n"])
        for val in max_values:
            out.write(f"{val:.6f}\n")

    print(f"✅ Created: {output_name}")

# === Function to process velocity_max.dat (vector with 2 columns) ===
def process_vector_max(file_name, output_name):
    result_folders = sorted([f for f in os.listdir(results_dir) if f.startswith("result_")])
    blocks = []

    for folder in result_folders:
        path = os.path.join(results_dir, folder, file_name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            block = []
            for line in lines:
                if line.strip().startswith("TS"):
                    if block:
                        blocks.append(block)
                    block = []
                else:
                    block.append(line)
            if block:
                blocks.append(block)

    parsed_blocks = []
    for block in blocks:
        data = []
        for line in block:
            try:
                x, y = map(float, line.strip().split()[:2])
                data.append([x, y])
            except:
                continue
        if data:
            parsed_blocks.append(np.array(data))

    min_length = min(len(b) for b in parsed_blocks)
    trimmed = [b[:min_length] for b in parsed_blocks]
    magnitudes = [np.linalg.norm(b, axis=1) for b in trimmed]
    stacked_mags = np.stack(magnitudes)
    max_indices = np.argmax(stacked_mags, axis=0)
    max_vectors = np.array([trimmed[max_indices[i]][i] for i in range(len(max_indices))])

    nd_val = get_nd_value(os.path.join(results_dir, result_folders[0], file_name))

    with open(output_name, "w") as out:
        out.writelines(["VECTOR\n", f"ND     {nd_val}\n", "ST  0\n", "TIMEUNITS Seconds\n", f"TS         {ts_value}\n"])
        np.savetxt(out, max_vectors, fmt="%.6f")

    print(f"✅ Created: {output_name}")

# === Run all three ===
process_scalar_max("depth_max.dat", "final_max_depth_with_correct_header.dat", ["SCALAR\n"])
process_scalar_max("wspl_max.dat", "final_max_wspl_with_correct_header.dat", ['SCALAR "NaN= 0.0000000E+00"\n'])
process_vector_max("velocity_max.dat", "final_max_velocity_with_proper_header.dat")



# AUTHOR: SAMED ZEYTINCI





























































































