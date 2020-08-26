# Define main path variables
CONFIG=src/rsgan/config/modis_landsat_fusion/generative/cgan_fusion_unet.yaml
PREDICTIONS=data/experiments_outputs/modis_landsat_fusion/ESTARFM/predictions
GROUNDTRUTH=data/not-so-toy/patches/landsat_modis_ESTARFM/
OUTPUT=data/experiments_outputs/modis_landsat_fusion/ESTARFM/eval

# Run dvc pipeline on specified device
dvc run -v -f -n test_modis_landsat_fusion_ESTARFM \
-d $CONFIG \
-d $PREDICTIONS \
-d $GROUNDTRUTH \
-o $OUTPUT \
"python src/rsgan/evaluation/ESTARFM.py --cfg=$CONFIG --root=$PREDICTIONS --target=$GROUNDTRUTH --o=$OUTPUT"
