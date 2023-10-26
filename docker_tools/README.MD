# WARNING:

**THIS FUNCTIONALITY IS STILL UNDER DEVELOPMENT**, WE ARE TRYING TO:

- MAKE THE DOCKER DEPLOYMENT COMPATIBLE WITH CUDA (FOR NVIDIA GPUS)
- MAKE THE DOCKER DEPLOYMENT COMPATIBLE WITH ROCm (FOR AMD GPUs)
- MAKE IT COMPATIBLE WITH CPU ONLY TRAINING.



# RUN MODIFIED CONTAINER

```
sudo docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G -p 8888:8888 -v /home/felipera/projects/udeai_forest:/mnt rocm-pytorch-felipe-modified:latest bash
```

# Open jupyter service

```
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

# NOTE: Remember to access through the port 8888 while using the token displayed on the bash terminal
