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
dvc run -v -n train_cgan_toy_cloud_removal \
-d src/rsgan/experiments/cloud_removal/cgan_toy_cloud_removal.py \
-d src/rsgan/config/cloud_removal/cgan_toy.yaml \
-d data/toy/cloud_removal \
-o data/experiments_outputs/cgan_toy_cloud_removal/dvc_run/run/ \
"python run_training.py --cfg=src/rsgan/config/cloud_removal/cgan_toy.yaml --o=data/experiments_outputs/cgan_toy_cloud_removal --device=$DEVICE --experiment_name=dvc_run"
