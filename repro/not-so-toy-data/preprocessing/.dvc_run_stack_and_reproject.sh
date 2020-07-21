# MODIS
ROOT_MODIS=/data/raw_data/modis/mcd43a4_aws/
OUTPUT_MODIS=data/not-so-toy/reprojected/modis
SCENES_MODIS=src/notsotoygeneration/config/scenes/modis.yaml

dvc run -v -n stack_bands_and_reproject_modis \
-d src/notsotoygeneration/preprocessing/stack_and_reproject_modis.py \
-d $ROOT_MODIS \
-d $SCENES_MODIS \
-o $OUTPUT_MODIS \
"python src/notsotoygeneration/preprocessing/stack_and_reproject_modis.py --root=$ROOT_MODIS --o=$OUTPUT_MODIS --scenes=$SCENES_MODIS"


# Sentinel-2
ROOT_S2=/data/raw_data/s2_aws/downloaded
OUTPUT_S2=data/not-so-toy/reprojected/s2
SCENES_S2=src/notsotoygeneration/config/scenes/s2.yaml

dvc run -v -f -n stack_bands_and_reproject_s2 \
-d src/notsotoygeneration/preprocessing/stack_and_reproject_s2.py \
-d $ROOT_S2/31UEQ \
-d $ROOT_S2/31UFQ \
-d $SCENES_S2 \
-o $OUTPUT_S2 \
"python src/notsotoygeneration/preprocessing/stack_and_reproject_s2.py --root=$ROOT_S2 --o=$OUTPUT_S2 --scenes=$SCENES_S2"


# Sentinel-1
ROOT_S1=/data/raw_data/s1_aws/france_farm/clipped_tif_2018/
OUTPUT_S1=data/not-so-toy/reprojected/s1
SCENES_S1=src/notsotoygeneration/config/scenes/s1.yaml

dvc run -v -n stack_bands_and_reproject_s1 \
-d src/notsotoygeneration/preprocessing/stack_and_reproject_s1.py \
-d $ROOT_S1 \
-d $SCENES_S1 \
-o $OUTPUT_S1 \
"python src/notsotoygeneration/preprocessing/stack_and_reproject_s1.py --root=$ROOT_S1 --o=$OUTPUT_S1 --scenes=$SCENES_S1"
