# data/script_download_all_datasets.sh

#!/bin/bash

# Define the base URL for downloads
BASE_URL="https://data.dgl.ai/dataset/graphforming/benchmarking-gnns"

# List of datasets to download
DATASETS=(
  "AQSOL.zip"
  "ZINC.zip"
  "SBM_CLUSTER.zip"
  "SBM_PATTERN.zip"
  "CYCLES.zip"
  "GraphTheoryProp.zip"
  "CSL.zip"
  "MNIST.zip"
  "CIFAR10.zip"
  "TUs.zip"
  "TSP.zip"
  "COLLAB.zip"
  "WikiCS.zip"
)

# Create data directory if it doesn't exist
mkdir -p data
cd data

echo "Starting download of all datasets..."

# Loop through and download each dataset
for ds in "${DATASETS[@]}"; do
  echo "Downloading ${ds}..."
  if [ ! -f "$ds" ]; then
    wget "${BASE_URL}/${ds}"
    if [ $? -eq 0 ]; then
      echo "${ds} downloaded successfully."
      # Unzip the file, removing the .zip extension for the folder name
      unzip -q "$ds" -d "${ds%.zip}" && rm "$ds"
      echo "Extracted ${ds}."
    else
      echo "Failed to download ${ds}."
    fi
  else
    echo "${ds} already exists. Skipping download."
  fi
done

echo "All datasets downloaded and extracted."
