import json
import csv
import os

# 1. JSON file reading
json_file = "test_analysis_result.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. geometry_elements extraction
geometry_elements = data.get("geometry_elements", [])

# 3. CSV file generation
csv_file = "storage/labels/labels_005.csv"

# Create directory if it does not exist
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # CSV header
    writer.writerow(["source_handle", "label", "length", "area", "radius"])

    #  entity_handle and layer extraction
    for entity in geometry_elements:

        handle = entity.get("source_handle")
        layer = entity.get("layer")
        length = entity.get("length")
        area = entity.get("area")
        radius = entity.get("radius")

        writer.writerow([handle, layer, length, area, radius])

print(f"CSV file generation finished: {csv_file}")
print(f"extracted entities: {len(geometry_elements)}")