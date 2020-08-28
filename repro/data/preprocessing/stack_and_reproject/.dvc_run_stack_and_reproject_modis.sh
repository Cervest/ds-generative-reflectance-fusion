# MODIS
ROOT=/data/raw_data/modis/mcd43a4_aws/
OUTPUT=data/reprojected/modis
SCENES=src/prepare_data/config/scenes/modis.yaml

dvc run -v -f -n stack_bands_and_reproject_modis \
-d src/prepare_data/preprocessing/stack_and_reproject/stack_and_reproject_modis.py \
-d $ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/prepare_data/preprocessing/stack_and_reproject/stack_and_reproject_modis.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
