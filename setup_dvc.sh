#!/bin/bash

# Initialize DVC
echo "Initializing DVC..."
dvc init

# Add data directory to DVC
echo "Adding data to DVC tracking..."
dvc add data/raw/customer_churn.csv
dvc add data/processed/train.csv
dvc add data/processed/val.csv
dvc add data/processed/test.csv

# Add model directory to DVC
echo "Adding models to DVC tracking..."
dvc add models/random_forest_model.pkl
dvc add models/preprocessor.pkl

# Configure remote storage (example with local directory)
# In production, use S3, GCS, or Azure Blob
echo "Configuring DVC remote storage..."
dvc remote add -d myremote /tmp/dvc-storage

# Or for Google Drive:
# dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID

# Or for S3:
# dvc remote add -d s3remote s3://mybucket/path

echo "DVC setup complete!"
echo "Remember to commit .dvc files to git:"
echo "git add data/.gitignore data/raw/customer_churn.csv.dvc"
echo "git add models/.gitignore models/*.dvc"
echo "git commit -m 'Add data and models to DVC'"
echo ""
echo "To push data to remote:"
echo "dvc push"