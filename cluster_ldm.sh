#!/bin/bash -l
#SBATCH --job-name=ddpm_ood_ldm
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:4
#SBATCH --ntasks-per-node=4
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
cd $WORKDIR/Decathlon
tar xf Task01_BrainTumour.tar
echo "Done copying and unpacking data to compute node."

export output_root=$WORK/Data/OOD/output_data
export data_root=$WORK/Data/OOD/input_data

cd /home/woody/iwi5/iwi5046h/Code/ddpm-ood/

python adapt_paths_on_cluster.py --new_data_root $WORKDIR

export data_root=$WORKDIR

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=ddpm_decathlon \
  --vqvae_checkpoint=${output_root}/vqvae_decathlon/checkpoint.pth \
  --training_ids=${data_root}/data_splits/Task01_BrainTumour_train.csv \
  --validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
  --is_grayscale=1 \
  --n_epochs=12000 \
  --batch_size=8 \
  --eval_freq=25 \
  --checkpoint_every=1000 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=small \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128

