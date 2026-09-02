# Workloads
## LLama 3 8B
Single node benchmark script with SDMA and mock data
```
ssh -o BatchMode=yes -o ConnectTimeout=10 $NODE_ID 'if [ "$(docker ps -q | wc -l)" -ne 0 ]; then echo NODE_BUSY; exit 9; fi; docker run --rm --name llama8-standalone-nofile-test --network host --ipc host --privileged --device /dev/kfd --device /dev/dri --group-add video --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --shm-size 128g rocm/primus:v26.5 bash -lc '"'"'set -eo pipefail; cd /workspace/Primus; git fetch -q https://github.com/Z-Y00/Primus.git feature/megatron-rccl-sdma-direct-gather; git checkout -q --detach FETCH_HEAD; export PYTHONPATH=${PYTHONPATH:-}; source examples/mlperf/llama3.1_8b/config_MI355X_1x8x1.sh; /opt/venv/bin/python -c "from pathlib import Path; p=Path(\"$EXP\"); s=p.read_text().replace(\"mock_data: false\", \"mock_data: true\").replace(\"train_data_path: \\\"/data/c4-train.en_6_text_document\\\"\", \"train_data_path: null\").replace(\"valid_data_path: \\\"/data/c4-validation-91205-samples.en_text_document\\\"\", \"valid_data_path: null\"); Path(\"/tmp/llama8-mock.yaml\").write_text(s)"; \
 export EXP=/tmp/llama8-mock.yaml PRIMUS_TRAIN_ITERS=500 PRIMUS_EVAL_ITERS=0 SYNTH_WARMUP_STEPS=10 \
  SYNTH_WARMUP_VALID_STEPS=0 PRIMUS_EVAL_INTERVAL=1000000 PRIMUS_MLPERF_PERF_MOCK=1 DATA_PATH=/tmp/mock_data \
  MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma MEGATRON_RCCL_SDMA_DIRECT=1 MEGATRON_RCCL_SDMA_EAGER_INIT=1 \
  NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo RCCL_MSCCL_ENABLE=0 RCCL_DDA_ENABLE=0 NCCL_DEBUG=VERSION; \
  mkdir -p /results /tmp/mock_data; timeout -k 30s 900s ./primus-cli direct -- train pretrain --config "$EXP"'"'"'' > llama3.8b.sdma.log
```
without SDMA

```
ssh -o BatchMode=yes -o ConnectTimeout=10 $NODE_ID 'if [ "$(docker ps -q | wc -l)" -ne 0 ]; then echo NODE_BUSY; exit 9; fi; docker run --rm --name llama8-standalone-nofile-test --network host --ipc host --privileged --device /dev/kfd --device /dev/dri --group-add video --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --shm-size 128g rocm/primus:v26.5 bash -lc '"'"'set -eo pipefail; cd /workspace/Primus; export PYTHONPATH=${PYTHONPATH:-}; source examples/mlperf/llama3.1_8b/config_MI355X_1x8x1.sh; /opt/venv/bin/python -c "from pathlib import Path; p=Path(\"$EXP\"); s=p.read_text().replace(\"mock_data: false\", \"mock_data: true\").replace(\"train_data_path: \\\"/data/c4-train.en_6_text_document\\\"\", \"train_data_path: null\").replace(\"valid_data_path: \\\"/data/c4-validation-91205-samples.en_text_document\\\"\", \"valid_data_path: null\"); Path(\"/tmp/llama8-mock.yaml\").write_text(s)"; \
 export EXP=/tmp/llama8-mock.yaml PRIMUS_TRAIN_ITERS=500 PRIMUS_EVAL_ITERS=0 SYNTH_WARMUP_STEPS=10 \
  SYNTH_WARMUP_VALID_STEPS=0 PRIMUS_EVAL_INTERVAL=1000000 PRIMUS_MLPERF_PERF_MOCK=1 DATA_PATH=/tmp/mock_data \
  MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma MEGATRON_RCCL_SDMA_DIRECT=1 MEGATRON_RCCL_SDMA_EAGER_INIT=1 \
  NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo RCCL_MSCCL_ENABLE=0 RCCL_DDA_ENABLE=0 NCCL_DEBUG=VERSION; \
  mkdir -p /results /tmp/mock_data; timeout -k 30s 900s ./primus-cli direct -- train pretrain --config "$EXP"'"'"'' > llama3.8b.no.sdma.log
```
