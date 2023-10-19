#!/bin/bash -l
#SBATCH --job-name=ddpm_ood
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
cp -r $WORK/Data/OOD/input_data/data_splits $WORKDIR
cp -r $WORK/Data/OOD/input_data/Decathlon $WORKDIR
cd $WORKDIR/Decathlon
tar xf Task01_BrainTumour.tar
echo "Done copying and unpacking data to compute node."

cd /home/woody/iwi5/iwi5046h/Code/ddpm-ood/

export output_root=$WORK/Data/OOD/output_data
export data_root=$WORKDIR

srun python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_decathlon \
--training_ids=${data_root}/data_splits/Task01_BrainTumour_train.csv \
--validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
--is_grayscale=1 \
--n_epochs=300 \
--batch_size=8  \
--eval_freq=10 \
--cache_data=0  \
--vqvae_downsample_parameters=[[2,4,1,1],[2,4,1,1],[2,4,1,1],[2,4,1,1]] \
--vqvae_upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0]] \
--vqvae_num_channels=[256,256,256,256] \
--vqvae_num_res_channels=[256,256,256,256] \
--vqvae_embedding_dim=128 \
--vqvae_num_embeddings=2048 \
--vqvae_decay=0.9  \
--vqvae_learning_rate=3e-5 \
--spatial_dimension=3 \
--image_roi=[160,160,128] \
--image_size=128


