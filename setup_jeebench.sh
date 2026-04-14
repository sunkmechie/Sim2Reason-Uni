#!/bin/bash

# JEEBench dataset preparation script

# Global variables
PHO_DATA_DIR="${PHO_DATA_DIR:-/home/mprabhud/datasets/physics_sim_data}"
JEEBENCH_DIR="${PHO_DATA_DIR}/JEEBench"
DATASET_FILE="${JEEBENCH_DIR}/dataset.json"
TEMP_DIR="/tmp"
JEEBENCH_REPO_URL="https://github.com/dair-iitd/jeebench/archive/refs/heads/main.zip"

echo "🚀 Starting JEEBench dataset preparation..."
echo "📂 Target directory: ${JEEBENCH_DIR}"

# Create target directory
echo "📁 Creating target directory..."
mkdir -p "${JEEBENCH_DIR}"

# Download JEEBench repository
cd "${TEMP_DIR}"
echo "📥 Downloading JEEBench repository..."
wget -q "${JEEBENCH_REPO_URL}"

# Extract repository
echo "📦 Extracting repository..."
unzip -q main.zip
cd jeebench-main
unzip -q data.zip

# Copy dataset file to target location
echo "📋 Copying dataset file..."
cp data/dataset.json "${DATASET_FILE}"

# Verify dataset
echo "🔍 Verifying dataset..."
python3 -c "
import json
with open('${DATASET_FILE}', 'r') as f:
    data = json.load(f)
print(f'✅ Total questions: {len(data)}')
subjects = {}
types = {}
for item in data:
    subjects[item['subject']] = subjects.get(item['subject'], 0) + 1
    types[item['type']] = types.get(item['type'], 0) + 1
print('📊 Subject distribution:', subjects)
print('📝 Question type distribution:', types)
"

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf "${TEMP_DIR}/jeebench-main" "${TEMP_DIR}/main.zip"

echo "✅ JEEBench dataset preparation completed!"
echo "📍 Dataset location: ${DATASET_FILE}" 