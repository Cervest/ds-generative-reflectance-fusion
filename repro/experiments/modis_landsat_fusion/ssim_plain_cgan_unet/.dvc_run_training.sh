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


# Run dvc pipeline on specified device
#for SEED in 17 43 73 ;
for SEED in 17 43 ;
do
  NAME=seed_$SEED
  TRAIN_DIR=$ROOT/$NAME/run
  dvc run -v -f -n train_modis_landsat_fusion_ssim_plain_cgan_unet_$NAME \
  -d $CONFIG \
  -d $EXPERIMENT \
  -d $DATASET \
  -o $TRAIN_DIR \
  "python run_training.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE --experiment_name=$NAME"
done
