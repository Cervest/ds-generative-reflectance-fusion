# Root directories
LANDSAT_ROOT=data/not-so-toy/reprojected/landsat
MODIS_ROOT=data/not-so-toy/reprojected/modis
OUTPUT=data/not-so-toy/patches/landsat_modis
SCENES=src/notsotoygeneration/config/patch_extraction/landsat_modis.yaml

dvc run -v -f -n patch_extraction_landsat_modis \
-d src/notsotoygeneration/preprocessing/patch_extraction/extract_patches_modis_landsat.py \
-d $LANDSAT_ROOT \
-d $MODIS_ROOT \
-d $SCENES \
-o $OUTPUT \
"python src/notsotoygeneration/preprocessing/patch_extraction/extract_patches_modis_landsat.py --modis_root=$MODIS_ROOT --landsat_root--o=$LANDSAT_ROOT --scenes=$SCENES --o=$OUTPUT"
