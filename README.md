# EIFFEL
Code for the paper - EIFFEL

Need to install tensorboard - pip install tensorboardX<br>

Other dependencies:
```
Package                  Version
------------------------ -----------
filelock                 3.13.1
fsspec                   2024.2.0
Jinja2                   3.1.3
joblib                   1.3.2
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.2.1
numpy                    1.26.4
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        8.9.2.26
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.19.3
nvidia-nvjitlink-cu12    12.4.99
nvidia-nvtx-cu12         12.1.105
packaging                24.0
pillow                   10.2.0
pip                      22.0.2
protobuf                 4.25.3
scikit-learn             1.4.1.post1
scipy                    1.12.0
setuptools               59.6.0
sympy                    1.12
tensorboardX             2.6.2.2
threadpoolctl            3.3.0
torch                    2.2.1
torchvision              0.17.1
triton                   2.2.0
typing_extensions        4.10.0
```

! git clone repository

cd EIFFEL/

!python main.py --num_clients 10 --optimizer Adam --lr 0.1 --momentum 0.5 --AR fedavg --dataset cifar --loader_type dirichlet --experiment_name "Battery" --device cpu --batch_size 64 --test_batch_size 2048  --collection_size 400 --collection_battery 0.005 --training_battery 0.002 --alpha 0.5 --beta 0.5 --gamma 0.5 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --upload_battery 0.3 --download_battery 3 --collection_success_chance 0.95 --sample_selection loss --collection_budget 10 --training_budget 10
