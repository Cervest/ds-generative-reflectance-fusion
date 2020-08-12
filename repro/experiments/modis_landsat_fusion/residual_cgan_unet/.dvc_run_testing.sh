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
CONFIG=src/rsgan/config/modis_landsat_fusion/generative/residual_cgan_fusion_unet.yaml
EXPERIMENT=src/rsgan/experiments/modis_landsat_fusion/residual_cgan_fusion_modis_landsat.py
DATASET=data/not-so-toy/patches/landsat_modis
ROOT=data/experiments_outputs/modis_landsat_fusion/residual_cgan_unet
TRAIN_DIR=$ROOT/dvc_run/run/
TEST_DIR=$ROOT/dvc_run/eval/

# Run dvc pipeline on specified device
dvc run -v -n test_modis_landsat_fusion_residual_cgan_unet \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-d $TRAIN_DIR \
-o $TEST_DIR \
"python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE"
