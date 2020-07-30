# Sentinel-1
ROOT=/data/raw_data/s1_aws/france_farm/clipped_tif_2018/
OUTPUT=data/not-so-toy/reprojected/s1
SCENES=src/notsotoygeneration/config/scenes/s1.yaml

dvc run -v -n stack_bands_and_reproject \
-d src/notsotoygeneration/preprocessing/stack_and_reproject/stack_and_reproject.py \
-d $ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/stack_and_reproject/stack_and_reproject.py --root=$ROOT --o=$OUTPUT --scenes=$SCENES"
