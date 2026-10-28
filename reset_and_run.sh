rm -rf /home/bishop/ML_CBDP/models
echo "Removed all the previous model files, ingesting data file ............"
python3 src/data_ingestion_pipeline_xgboost.py