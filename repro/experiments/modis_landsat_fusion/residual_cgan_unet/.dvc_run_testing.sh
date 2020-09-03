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
CONFIG=src/deep_reflectance_fusion/config/modis_landsat_fusion/generative/residual_cgan_fusion_unet.yaml
EXPERIMENT=src/deep_reflectance_fusion/experiments/modis_landsat_fusion/cgan_fusion_modis_landsat.py
DATASET=data/patches/modis_landsat
ROOT=data/experiments_outputs/modis_landsat_fusion/residual_cgan_unet
SEEDS=(17 37 43 73 101)
CHKPTS=("seed_17/checkpoints/epoch=42.ckpt"
        "seed_37/checkpoints/epoch=62.ckpt"
        "seed_43/checkpoints/epoch=60.ckpt"
        "seed_73/checkpoints/epoch=63.ckpt"
        "seed_101/checkpoints/epoch=61.ckpt")


# Run dvc pipeline on specified device
for (( i=0; i<${#SEEDS[*]}; ++i));
do
  NAME=seed_${SEEDS[$i]}
  CHKPT=$ROOT/${CHKPTS[$i]}
  TRAIN_DIR=$ROOT/$NAME/run
  TEST_DIR=$ROOT/$NAME/eval
  dvc run -v -f -n test_modis_landsat_fusion_residual_cgan_unet_$NAME \
  -d $CONFIG \
  -d $EXPERIMENT \
  -d $DATASET \
  -d $TRAIN_DIR \
  -o $TEST_DIR \
  "python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE --chkpt=$CHKPT"
done
