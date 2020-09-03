# Retrive device id
for i in "$@"
do
case $i in
    --device=*)
    DEVICE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Define main path variables
CONFIG=src/deep_reflectance_fusion/config/modis_landsat_fusion/plain/residual_fusion_unet.yaml
EXPERIMENT=src/deep_reflectance_fusion/experiments/modis_landsat_fusion/early_fusion_modis_landsat.py
DATASET=data/patches/modis_landsat
ROOT=data/experiments_outputs/modis_landsat_fusion/residual_unet
TRAIN_DIR=$ROOT/dvc_run/run/
TEST_DIR=$ROOT/dvc_run/eval/

# Run dvc pipeline on specified device
dvc run -v -f -n test_modis_landsat_fusion_residual_unet \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-d $TRAIN_DIR \
-o $TEST_DIR \
"python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE"
