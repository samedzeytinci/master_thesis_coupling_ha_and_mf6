# IMPORT ALL PACKAGES BEFOREHAND
import numpy as np
import math
import subprocess
import os
import pandas as pd
from pathlib import Path
from datetime import datetime






########## PRELIMINARY STEPS FOR MODFLOW

# PRELIMINARY STEPS
# STEPS BEFORE THE COUPLING RUN
# INCLUDES CREATION OF DELR, DELC DATA, CELL CENTERS WITH BOUNDING BOXES, CREATION OF IDW CELLS AND NODES ETC.

# PART 1

# # PART TO CREATE CELLS WITH THEIR CENTER COORDINATES




import numpy as np

# File paths
dis_file_path = "model_2_Riv_3.dis"

output_file = "cell_coordinates_with_corners.txt"

# Read the .dis file
with open(dis_file_path, "r") as file:
    dis_lines = file.readlines()

# Extract origin from "Upper left corner" line
x_origin, y_origin = None, None
for line in dis_lines:
    if "Upper left corner:" in line:
        parts = line.strip().split(":")[1].strip().strip("()").split(", ")
        x_origin = float(parts[0])
        y_origin = float(parts[1])
        break

# Extract number of rows and columns
nrow = ncol = None
for line in dis_lines:
    if "NROW" in line:
        nrow = int(line.split()[1])
    elif "NCOL" in line:
        ncol = int(line.split()[1])

# Extract DELR and DELC constant values
delr_value = delc_value = None
for i, line in enumerate(dis_lines):
    if line.strip().startswith("DELR"):
        for j in range(i+1, i+4):
            if "CONSTANT" in dis_lines[j]:
                delr_value = float(dis_lines[j].split()[1])
                break
    elif line.strip().startswith("DELC"):
        for j in range(i+1, i+4):
            if "CONSTANT" in dis_lines[j]:
                delc_value = float(dis_lines[j].split()[1])
                break

# Create arrays for DELR and DELC
DELR_array = np.full(ncol, delr_value)
DELC_array = np.full(nrow, delc_value)


# Generate cell center and corner coordinates
cell_coordinates = []
y = y_origin
for row_idx, delc in enumerate(DELC_array):
    x = x_origin
    for col_idx, delr in enumerate(DELR_array):
        x_center = x + delr / 2
        y_center = y - delc / 2
        X1, Y1 = x, y
        X2, Y2 = x + delr, y
        X3, Y3 = x + delr, y - delc
        X4, Y4 = x, y - delc
        cell_coordinates.append((
            row_idx + 1, col_idx + 1, x_center, y_center,
            X1, Y1, X2, Y2, X3, Y3, X4, Y4
        ))
        x += delr
    y -= delc

# Write coordinates to file
with open(output_file, "w") as f:
    f.write("Row, Column, X_Center, Y_Center, X1, Y1, X2, Y2, X3, Y3, X4, Y4\n")
    for coord in cell_coordinates:
        f.write(
            f"{coord[0]}, {coord[1]}, {coord[2]:.6f}, {coord[3]:.6f}, "
            f"{coord[4]:.6f}, {coord[5]:.6f}, {coord[6]:.6f}, {coord[7]:.6f}, "
            f"{coord[8]:.6f}, {coord[9]:.6f}, {coord[10]:.6f}, {coord[11]:.6f}\n"
        )

output_file


# PART 1.1 EXTRACTING TOP AND BOT LEVELS FROM DIS FILE


# # FOR TOP LEVELS


# Read the .dis file
dis_file_path = "model_2_Riv_3.dis"
with open(dis_file_path, "r") as file:
    dis_lines = file.readlines()

# Function to find the start index of the top layer values
def find_top_start_index(lines, keyword):
    for i, line in enumerate(lines):
        if keyword in line:
            return i + 1  # Move to the data row
    return None

# Function to extract numerical values from space-separated lines
# Stops when encountering non-numeric values
def extract_top_values(lines, start_idx):
    values = []
    for line in lines[start_idx:]:
        try:
            values.extend([float(val) for val in line.split()])
        except ValueError:
            break
    return np.array(values)

# Locate the top layer values
top_start_idx = find_top_start_index(dis_lines, "INTERNAL IPRN     12 # TOP")

# Extract the top values
nrows, ncols = 85, 132  # Defined grid size
top_values = extract_top_values(dis_lines, top_start_idx)
top_values = top_values[: nrows * ncols].reshape(nrows, ncols)

# Prepare the output format
output_lines = ["row, column, top level"]
for row in range(nrows):
    for col in range(ncols):
        output_lines.append(f"{row+1}, {col+1}, {top_values[row, col]:.6E}")

# Write the results to a file
output_file_path = "top_layer_extracted.txt"
with open(output_file_path, "w") as file:
    file.write("\n".join(output_lines))

print(f"Top layer values saved to {output_file_path}")


#ADDING RIV IN THE NAM FILE 

# Define the file path
file_path = "model_2_Riv_3.nam"

# Read the file contents
with open(file_path, "r") as file:
    lines = file.readlines()

# Find the index of the line containing "OC6"
insert_index = None
for i, line in enumerate(lines):
    if "OC6" in line:
        insert_index = i + 1
        break

# If "OC6" is found, insert the required lines
if insert_index is not None:
    lines.insert(insert_index, "  RIV6         model_2_Riv_3.riv RIV-1\n")

# Write the modified contents back to the file
modified_file_path = "model_2_Riv_3.nam"
with open(modified_file_path, "w") as file:
    file.writelines(lines)

# Provide the modified file to the user
modified_file_path


# PRELIMINARY STEPS OF HYDROAS

# PRELIMINARY OF IDW Z 

import math
from pathlib import Path

# Define file paths
cell_file = "cell_coordinates_with_corners.txt"
node_file = "nodes_with_cells.txt"

p = 2  # Power parameter for IDW

# Load cells
cells = {}
with open(cell_file, "r") as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split(", ")
        row, col = int(parts[0]), int(parts[1])
        x_center, y_center = float(parts[2]), float(parts[3])
        xmin = min(float(parts[4]), float(parts[6]), float(parts[8]), float(parts[10]))
        xmax = max(float(parts[4]), float(parts[6]), float(parts[8]), float(parts[10]))
        ymin = min(float(parts[5]), float(parts[7]), float(parts[9]), float(parts[11]))
        ymax = max(float(parts[5]), float(parts[7]), float(parts[9]), float(parts[11]))
        cells[(row, col)] = (x_center, y_center, xmin, xmax, ymin, ymax)

# import pandas as pd
import numpy as np




## CREATING THE FILE initial_depths.txt from hydroas 2dm file

# Define the input and output file paths
input_file_path = "hydro_as-2d.2dm"
output_file_path = "initial_depths.txt"

# Read the file using an appropriate encoding (latin-1 for special characters)
with open(input_file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Filter lines starting with 'ND' and extract columns 3, 4, and 5
data = []
for line in lines:
    if line.strip().startswith("ND"):
        parts = line.strip().split()
        if len(parts) >= 5:
            x, y, z = parts[2], parts[3], parts[4]
            data.append(f"{x} {y} {z}\n")

# Write the extracted data to a new file
with open(output_file_path, 'w') as out_file:
    out_file.writelines(data)

print("Extraction complete. Data written to:", output_file_path)


# File paths
cell_file = "cell_coordinates_with_corners.txt"
node_file = "initial_depths.txt"
nodes_with_weights_file = "nodes_with_weights.txt"
output_file = "cell_IDW_results.txt"
nodes_with_cells_output = "nodes_with_cells.txt"

p = 2  # Power parameter for IDW

# Load cells
cells = {}
with open(cell_file, "r") as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split(", ")
        row, col = int(parts[0]), int(parts[1])
        x_center, y_center = float(parts[2]), float(parts[3])
        x1, y1 = float(parts[4]), float(parts[5])
        x2, y2 = float(parts[6]), float(parts[7])
        x3, y3 = float(parts[8]), float(parts[9])
        x4, y4 = float(parts[10]), float(parts[11])
        xmin, xmax = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        ymin, ymax = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        cells[(row, col)] = (x_center, y_center, xmin, xmax, ymin, ymax)

# Load nodes (each node belongs to exactly one cell)
node_cell_map = {}  # Key: (x, y) -> Value: (z, row, col, distance, weight)
ordered_node_data = []

with open(node_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        
        # Find which cell this node belongs to
        for (row, col), (x_center, y_center, xmin, xmax, ymin, ymax) in cells.items():
            if xmin <= x <= xmax and ymin <= y <= ymax:
                distance = math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                weight = 1 / (distance ** p) if distance > 0 else 0
                node_cell_map[(x, y)] = (z, row, col, distance, weight)
                ordered_node_data.append((x, y, z, row, col))  # Preserve order for extra file
                break  # Exit after finding the correct cell (each node belongs to only one)

# Normalize weights within each cell
cell_weights = {}  # Key: (row, col) -> List of (z, weight)
for (x, y), (z, row, col, distance, weight) in node_cell_map.items():
    if (row, col) not in cell_weights:
        cell_weights[(row, col)] = []
    cell_weights[(row, col)].append((z, weight))

# Write nodes with weights to a file
with open(nodes_with_weights_file, "w") as f:
    f.write("Node_X, Node_Y, Node_Z, Row, Column, Distance_to_Center, IDW_Weight, Normalized_Weight\n")
    for (x, y), (z, row, col, distance, weight) in node_cell_map.items():
        total_weight = sum(w for _, w in cell_weights[(row, col)])
        normalized_weight = weight / total_weight if total_weight > 0 else 0
        f.write(f"{x}, {y}, {z}, {row}, {col}, {distance:.6f}, {weight:.6f}, {normalized_weight:.6f}\n")





# Compute IDW for each cell
idw_results = {}
with open(output_file, "w") as f:
    f.write("Row, Column, X_Center, Y_Center, IDW_Z, Weights\n")
    
    for (row, col), (x_center, y_center, xmin, xmax, ymin, ymax) in cells.items():
        if (row, col) in cell_weights:
            nodes = cell_weights[(row, col)]
            total_weight = sum(w for _, w in nodes)
            idw_z = sum(z * w for z, w in nodes) / total_weight if total_weight > 0 else 0
            weight_values = [f"{w / total_weight:.6f}" for _, w in nodes]  # Normalize weights
            f.write(f"{row}, {col}, {x_center:.6f}, {y_center:.6f}, {idw_z:.6f}, {', '.join(weight_values)}\n")

# Write the extra file with nodes and their assigned weights
with open(nodes_with_cells_output, "w") as f:
    f.write("X, Y, Z, Row, Column, Weight\n")
    for x, y, z, row, col in ordered_node_data:
        weight = node_cell_map[(x, y)][4]  # Retrieve weight from node_cell_map
        f.write(f"{x}, {y}, {z}, {row}, {col}, {weight:.6f}\n")

print(f"Node weights saved to {nodes_with_weights_file}")
print(f"IDW results written to {output_file}")
print(f"Node-to-cell mapping with weights saved to {nodes_with_cells_output}")





# MATCHING NODES WITH AREAS BEFORE APPLYING HEAD CHANGE TO THE NODES






# # PART MAKING NODES MATCHING WITH VOLUMES
# # ALSO IN PRELIMINARY PART TO PUT


import pandas as pd

# File paths
initial_depths_file = "initial_depths.txt"
volumen_file = "MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/volumen.dat"
output_file = "processed_data.txt"

# Load the initial_depths.txt file and extract the first two columns (X, Y)
df_initial_depths = pd.read_csv(initial_depths_file, delim_whitespace=True, header=None, usecols=[0, 1])
df_initial_depths.columns = ["X", "Y"]

# Read volumen.dat and extract values after "TS 0.0"
with open(volumen_file, "r") as f:
    lines = f.readlines()

# Find the index where "TS 0.0" appears
start_idx = None
for i, line in enumerate(lines):
    if "TS 0.0" in line:
        start_idx = i + 1  # Data starts right after this line
        break

# Extract the data after "TS 0.0"
if start_idx is not None:
    volumen_data = [float(line.strip()) for line in lines[start_idx:] if line.strip()]
else:
    volumen_data = []

# Ensure the number of extracted values matches the number of X, Y coordinates
min_length = min(len(df_initial_depths), len(volumen_data))
df_initial_depths = df_initial_depths.iloc[:min_length].copy()
df_initial_depths["area"] = volumen_data[:min_length]

# Save the result to a new text file
df_initial_depths.to_csv(output_file, sep=" ", index=False, header=["X", "Y", "area"])

print(f"Processed data saved to {output_file}")

# ### END ###


# CREATE PROCESSED_DATA for VOLUME FORMULA TO APPLY

# Required imports
from pathlib import Path

# File paths
initial_depths_path = Path("initial_depths.txt")
volumen_path = Path("MASTER THESIS - COUPLING TOOL FILES - SAMED_ZEYTINCI/Data-out/volumen.dat")
output_path = Path("processed_data.txt")

# Step 1 & 2: Read X and Y from initial_depths.txt
with open(initial_depths_path, "r") as f:
    initial_lines = f.readlines()
xy_data = [f"{line.split()[0]} {line.split()[1]}" for line in initial_lines]

# Step 3: Read volumen.dat and extract lines after 'TS 0.0'
with open(volumen_path, "r", encoding="latin-1") as f:
    lines = f.readlines()
ts_start_index = next(i for i, line in enumerate(lines) if "TS 0.0" in line)
area_values = [line.strip() for line in lines[ts_start_index + 1:]]

# Ensure lengths match
if len(xy_data) != len(area_values):
    raise ValueError("Mismatch between XY coordinates and area values.")

# Step 4: Write to processed_data.txt
with open(output_path, "w") as f:
    f.write("X Y area\n")
    for xy, area in zip(xy_data, area_values):
        f.write(f"{xy} {area}\n")

output_path.name  # Return the name of the created file


##################################################
##################################################
################################################## 
# END OF PRELIMINARY STEPS ######################


