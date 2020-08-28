# Root directories
LANDSAT_ROOT=data/reprojected/landsat
MODIS_ROOT=data/reprojected/modis
OUTPUT=data/patches/modis_landsat
SCENES=src/prepare_data/config/patch_extraction/modis_landsat.yaml

dvc run -v -f -n patch_extraction_modis_landsat \
-d src/prepare_data/preprocessing/patch_extraction/extract_patches_modis_landsat.py \
-d $LANDSAT_ROOT \
-d $MODIS_ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/prepare_data/preprocessing/patch_extraction/extract_patches_modis_landsat.py --modis_root=$MODIS_ROOT --landsat_root=$LANDSAT_ROOT --scenes_specs=$SCENES --o=$OUTPUT"
