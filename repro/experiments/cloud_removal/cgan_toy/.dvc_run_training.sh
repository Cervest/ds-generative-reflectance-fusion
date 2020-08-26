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
CONFIG=src/rsgan/config/cloud_removal/cgan_toy.yaml
EXPERIMENT=src/rsgan/experiments/cloud_removal/cgan_toy_cloud_removal.py
DATASET=data/toy/cloud_removal
ROOT=data/experiments_outputs/cgan_toy_cloud_removal
TRAIN_DIR=$ROOT/dvc_run/run/


# Run dvc pipeline on specified device
dvc run -v -n train_cgan_toy_cloud_removal \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-o $TRAIN_DIR \
"python run_training.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE --experiment_name=dvc_run"
