# LANDSAT
ROOT=/data/temporary/shahine/raw_data/landsat/landsat-8/france
OUTPUT=data/not-so-toy/reprojected/landsat
SCENES=src/notsotoygeneration/config/scenes/landsat.yaml

dvc run -v -f -n stack_bands_and_reproject_landsat \
-d src/notsotoygeneration/preprocessing/stack_and_reproject/stack_and_reproject_landsat.py \
-d $ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/stack_and_reproject/stack_and_reproject_landsat.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
