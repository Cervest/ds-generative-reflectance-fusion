# LANDSAT
ROOT=/data/temporary/shahine/raw_data/landsat/landsat-8/france
OUTPUT=data/reprojected/landsat
SCENES=src/prepare_data/config/scenes/landsat.yaml

dvc run -v -f -n stack_bands_and_reproject_landsat \
-d src/prepare_data/preprocessing/stack_and_reproject/stack_and_reproject_landsat.py \
-d $ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/prepare_data/preprocessing/stack_and_reproject/stack_and_reproject_landsat.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
