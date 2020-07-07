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

# Run dvc pipeline on specified device
dvc run -v -n test_cgan_frame_recurrent_toy_cloud_removal \
-d src/rsgan/experiments/cloud_removal/cgan_frame_recurrent_toy_cloud_removal.py \
-d src/rsgan/config/cloud_removal/cgan_frame_recurrent_toy.yaml \
-d data/toy/sar_to_optical \
-o data/experiments_outputs/cgan_frame_recurrent_toy_cloud_removal/dvc_run/eval/ \
"python make_baseline_classifier.py --cfg=src/rsgan/config/cloud_removal/cgan_frame_recurrent_toy.yaml --o=data/experiments_outputs/cgan_frame_recurrent_toy_cloud_removal/dvc_run/eval/baseline_classifier/classifier.pickle \
&& python run_testing.py --cfg=src/rsgan/config/cloud_removal/cgan_frame_recurrent_toy.yaml --o=data/experiments_outputs/cgan_frame_recurrent_toy_cloud_removal --device=$DEVICE"
