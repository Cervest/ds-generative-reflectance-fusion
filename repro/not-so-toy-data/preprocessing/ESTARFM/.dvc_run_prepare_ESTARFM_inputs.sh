# Root directories
PATCH_DIRECTORY=data/not-so-toy/patches/landsat_modis
OUTPUT=data/not-so-toy/patches/landsat_modis_ESTARFM
ESTARFM_OUTPUT=data/experiments_outputs/modis_landsat_fusion/ESTARFM

dvc run -v -f -n prepare_ESTARFM_inputs \
-d src/notsotoygeneration/preprocessing/ESTARFM/prepare_inputs.py \
-d $PATCH_DIRECTORY \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/ESTARFM/prepare_inputs.py --patch_dir=$PATCH_DIRECTORY --o=$OUTPUT --estarfm_out=$ESTARFM_OUTPUT"
