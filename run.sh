OMP_NUM_THREADS=1 /home/dut/.conda/envs/spider/bin/torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:20745 \
  --nnodes=1 \
  --nproc_per_node=1 \
  train_ema.py
