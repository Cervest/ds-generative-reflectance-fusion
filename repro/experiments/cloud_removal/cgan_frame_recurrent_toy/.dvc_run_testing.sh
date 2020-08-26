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
CONFIG=src/rsgan/config/cloud_removal/cgan_frame_recurrent_toy.yaml
EXPERIMENT=src/rsgan/experiments/cloud_removal/cgan_frame_recurrent_toy_cloud_removal.py
DATASET=data/toy/cloud_removal
ROOT=data/experiments_outputs/cgan_frame_recurrent_toy_cloud_removal
TRAIN_DIR=$ROOT/dvc_run/run/
TEST_DIR=$ROOT/dvc_run/eval/

# Run dvc pipeline on specified device
dvc run -v -n test_cgan_frame_recurrent_toy_cloud_removal \
-d $CONFIG \
-d $EXPERIMENT \
-d $DATASET \
-d $TRAIN_DIR \
-o $TEST_DIR \
"python make_baseline_classifier.py --cfg=$CONFIG --o=$TEST_DIR/baseline_classifier/classifier.pickle \
&& python run_testing.py --cfg=$CONFIG --o=$ROOT --device=$DEVICE"
