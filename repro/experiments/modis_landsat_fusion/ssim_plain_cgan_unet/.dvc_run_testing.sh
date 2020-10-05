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
CONFIG=src/deep_reflectance_fusion/config/modis_landsat_fusion/generative/ssim_cgan_fusion_unet.yaml
EXPERIMENT=src/deep_reflectance_fusion/experiments/modis_landsat_fusion/cgan_fusion_modis_landsat.py
DATASET=data/patches/modis_landsat
ROOT=data/experiments_outputs/modis_landsat_fusion/ssim_plain_cgan_unet
# SEEDS=(17 43 73)
# CHKPTS=("seed_17/checkpoints/epoch=???.ckpt"
#         "seed_43/checkpoints/epoch=???.ckpt"
#         "seed_73/checkpoints/epoch=426.ckpt")
SEEDS=(17)
CHKPTS=("seed_17/checkpoints/epoch=437.ckpt")


# Run dvc pipeline on specified device
for (( i=0; i<${#SEEDS[*]}; ++i));
do
  NAME=seed_${SEEDS[$i]}
  CHKPT=$ROOT/${CHKPTS[$i]}
  TRAIN_DIR=$ROOT/$NAME/run
  TEST_DIR=$ROOT/$NAME/eval
  dvc run -v -f -n test_modis_landsat_fusion_ssim_plain_cgan_unet_$NAME \
  -d $CONFIG \
  -d $EXPERIMENT \
  -d $DATASET \
  -d $TRAIN_DIR \
  -o $TEST_DIR \
  "python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE --chkpt=$CHKPT"
done
