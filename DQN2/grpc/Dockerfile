FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY DQN2/grpc/requirements.txt /tmp/requirements-grpc.txt
RUN grep -v '^torch' /tmp/requirements-grpc.txt > /tmp/requirements-runtime.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt

COPY . /app

EXPOSE 50051
CMD ["python", "DQN2/grpc/dqn_scheduler_server.py", \
     "--model", "DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth", \
     "--host", "0.0.0.0", \
     "--port", "50051", \
     "--hidden-dim", "512", \
     "--action-rerank-topk", "32", \
     "--action-rerank-balance-weight", "1.0", \
     "--action-rerank-q-weight", "1.0"]
