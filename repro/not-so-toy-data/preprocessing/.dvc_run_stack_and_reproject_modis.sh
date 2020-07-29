# MODIS
ROOT=/data/raw_data/modis/mcd43a4_aws/
OUTPUT=data/not-so-toy/reprojected/modis
SCENES=src/notsotoygeneration/config/scenes/modis.yaml

dvc run -v -n stack_bands_and_reproject \
-d src/notsotoygeneration/preprocessing/stack_and_reproject.py \
-d $ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/stack_and_reproject.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
