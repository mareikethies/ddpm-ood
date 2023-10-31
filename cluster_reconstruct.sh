#!/bin/bash -l
#SBATCH --job-name=ddpm_ood_reconstruct
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks-per-node=1
# if chained after another job, enter the job number here
# #SBATCH --dependency=afterany:146054
#SBATCH --mail-user=mareike.thies@fau.de
#SBATCH --mail-type=ALL
# do not export environment variables (see https://hpc.fau.de/systems-services/systems-documentation-instructions/clusters/tinygpu-cluster)
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load cuda/11.6.2
source /home/hpc/iwi5/iwi5046h/.bashrc
conda activate motion

# do this to have internet access for wandb logging
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# do this to prevent wandb from filling .cache directory in home
export WANDB_CACHE_DIR="/home/woody/iwi5/iwi5046h/Code/3d_moco_gradients/wandb_cache"

# copy training data to local temp dir on the current compute node and unpack there
WORKDIR="$TMPDIR/$SLURM_JOBID"
mkdir "$WORKDIR"
echo "Start copying and unpacking data to compute node."
echo "$WORKDIR"
mkdir "$WORKDIR"/Decathlon
cp -r $WORK/Data/OOD/input_data/Decathlon/Task01_BrainTumour.tar $WORKDIR/Decathlon/Task01_BrainTumour.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task02_Heart.tar $WORKDIR/Decathlon/Task02_Heart.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task03_Liver.tar $WORKDIR/Decathlon/Task03_Liver.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task04_Hippocampus.tar $WORKDIR/Decathlon/Task04_Hippocampus.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task05_Prostate.tar $WORKDIR/Decathlon/Task05_Prostate.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task06_Lung.tar $WORKDIR/Decathlon/Task06_Lung.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task07_Pancreas.tar $WORKDIR/Decathlon/Task07_Pancreas.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task08_HepaticVessel.tar $WORKDIR/Decathlon/Task08_HepaticVessel.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task09_Spleen.tar $WORKDIR/Decathlon/Task09_Spleen.tar
cp -r $WORK/Data/OOD/input_data/Decathlon/Task10_Colon.tar $WORKDIR/Decathlon/Task10_Colon.tar
cd $WORKDIR/Decathlon
tar xf Task01_BrainTumour.tar
tar xf Task02_Heart.tar
tar xf Task03_Liver.tar
tar xf Task04_Hippocampus.tar
tar xf Task05_Prostate.tar
tar xf Task06_Lung.tar
tar xf Task07_Pancreas.tar
tar xf Task08_HepaticVessel.tar
tar xf Task09_Spleen.tar
tar xt Task10_Colon.tar
echo "Done copying and unpacking data to compute node."

export output_root=$WORK/Data/OOD/output_data
export data_root=$WORK/Data/OOD/input_data

cd /home/woody/iwi5/iwi5046h/Code/ddpm-ood/

python adapt_paths_on_cluster.py --new_data_root $WORKDIR

export data_root=$WORKDIR

python reconstruct.py \
  --output_dir=${output_root} \
  --model_name=ddpm_decathlon \
  --vqvae_checkpoint=${output_root}/vqvae_decathlon/checkpoint.pth \
  --validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
  --in_ids=${data_root}/data_splits/Task01_BrainTumour_test.csv \
  --out_ids=${data_root}/data_splits/Task02_Heart_test.csv,${data_root}/data_splits/Task03_Liver_test.csv,${data_root}/data_splits/Task04_Hippocampus_test.csv,${data_root}/data_splits/Task05_Prostate_test.csv,${data_root}/data_splits/Task06_Lung_test.csv,${data_root}/data_splits/Task07_Pancreas_test.csv,${data_root}/data_splits/Task08_HepaticVessel_test.csv,${data_root}/data_splits/Task09_Spleen_test.csv\
  --is_grayscale=1 \
  --batch_size=32 \
  --cache_data=0 \
  --prediction_type=epsilon \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128 \
  --num_inference_steps=100 \
  --inference_skip_factor=2 \
  --run_val=1 \
  --run_in=1 \
  --run_out=1

