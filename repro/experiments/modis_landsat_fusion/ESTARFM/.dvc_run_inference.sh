# Define main path variables
ROOT=data/patches/modis_landsat_ESTARFM/
OUTPUT=data/experiments_outputs/modis_landsat_fusion/ESTARFM/predictions

# Run dvc pipeline
dvc run -v -f -n run_modis_landsat_fusion_ESTARFM \
-d $ROOT \
"python run_ESTARFM.py --root=$ROOT"
