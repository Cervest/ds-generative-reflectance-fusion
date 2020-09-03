# Define main path variables
CONFIG=src/deep_reflectance_fusion/config/modis_landsat_fusion/generative/cgan_fusion_unet.yaml
PREDICTIONS=data/experiments_outputs/modis_landsat_fusion/ESTARFM/predictions
GROUNDTRUTH=data/patches/modis_landsat_ESTARFM/
OUTPUT=data/experiments_outputs/modis_landsat_fusion/ESTARFM/eval

# Run dvc pipeline
dvc run -v -f -n test_modis_landsat_fusion_ESTARFM \
-d $CONFIG \
-d $PREDICTIONS \
-d $GROUNDTRUTH \
-o $OUTPUT \
"python run_ESTARFM_evaluation.py --cfg=$CONFIG --root=$PREDICTIONS --target=$GROUNDTRUTH --o=$OUTPUT"
