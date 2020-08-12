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


# Run dvc pipeline on specified device
dvc run -v -n train_modis_landsat_fusion_plain_unet \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-o $TRAIN_DIR \
"python run_training.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE --experiment_name=dvc_run"
