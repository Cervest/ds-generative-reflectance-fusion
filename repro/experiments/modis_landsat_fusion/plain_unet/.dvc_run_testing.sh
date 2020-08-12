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
CONFIG=src/rsgan/config/modis_landsat_fusion/plain/fusion_unet.yaml
EXPERIMENT=src/rsgan/experiments/modis_landsat_fusion/early_fusion_modis_landsat.py
DATASET=data/not-so-toy/patches/landsat_modis
ROOT=data/experiments_outputs/modis_landsat_fusion/plain_unet
TRAIN_DIR=$ROOT/dvc_run/run/
TEST_DIR=$ROOT/dvc_run/eval/

# Run dvc pipeline on specified device
dvc run -v -n test_modis_landsat_fusion_plain_unet \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-d $TRAIN_DIR \
-o $TEST_DIR \
"python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE"
