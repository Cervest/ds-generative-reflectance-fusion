# Sentinel-2
ROOT=/data/raw_data/s2_aws/downloaded
OUTPUT=data/not-so-toy/reprojected/s2
SCENES=src/notsotoygeneration/config/scenes/s2.yaml

dvc run -v -n stack_bands_and_reproject \
-d src/notsotoygeneration/preprocessing/stack_and_reproject/stack_and_reproject.py \
-d $ROOT/31UEQ \
-d $ROOT/31UFQ \
-d $SCENES \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/stack_and_reproject/tack_and_reproject.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
