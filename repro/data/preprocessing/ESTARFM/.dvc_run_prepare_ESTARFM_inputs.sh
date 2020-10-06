# Root directories
PATCH_DIRECTORY=data/patches/modis_landsat
OUTPUT=data/patches/modis_landsat_ESTARFM
ESTARFM_OUTPUT=data/experiments_outputs/modis_landsat_fusion/ESTARFM/predictions

dvc run -v -f -n prepare_ESTARFM_inputs \
-d src/prepare_data/preprocessing/ESTARFM/prepare_inputs.py \
-d $PATCH_DIRECTORY \
-o $OUTPUT \
"python src/prepare_data/preprocessing/ESTARFM/prepare_inputs.py --patch_dir=$PATCH_DIRECTORY --o=$OUTPUT --estarfm_out=$ESTARFM_OUTPUT"
