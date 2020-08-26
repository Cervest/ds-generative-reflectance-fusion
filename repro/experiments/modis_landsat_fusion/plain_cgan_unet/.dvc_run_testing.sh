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
CONFIG=src/rsgan/config/modis_landsat_fusion/generative/cgan_fusion_unet.yaml
EXPERIMENT=src/rsgan/experiments/modis_landsat_fusion/cgan_fusion_modis_landsat.py
DATASET=data/not-so-toy/patches/landsat_modis
ROOT=data/experiments_outputs/modis_landsat_fusion/plain_cgan_unet
TRAIN_DIR=$ROOT/dvc_run/run/
TEST_DIR=$ROOT/dvc_run/eval/


# Run dvc pipeline on specified device
for SEED in {17, 37, 43, 73, 101}:
do
  NAME=seed_$SEED
  TRAIN_DIR=$ROOT/$NAME/run
  TEST_DIR=$ROOT/$NAME/eval
  dvc run -v -f -n test_modis_landsat_fusion_plain_cgan_unet_$NAME \
  -d $CONFIG \
  -d $EXPERIMENT \
  -d $DATASET \
  -d $TRAIN_DIR \
  -o $TEST_DIR \
  "python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE"
done
