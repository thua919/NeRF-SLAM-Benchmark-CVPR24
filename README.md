
<h1 align="center">
[CVPR 2024]

Benchmarking Implicit Neural Representation and Geometric Rendering in Real-Time RGB-D SLAM

[Project Page](https://vlis2022.github.io/nerf-slam-benchmark/) | [Paper](https://arxiv.org/abs/2403.19473)
</h1>

<br>

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

For linux, you need to install **libopenexr-dev** before creating the environment:
```bash
sudo apt-get install libopenexr-dev
```
You can then create an anaconda environment called `benchmark-release`:
```bash
git clone https://github.com/thua919/NeRF-SLAM-Benchmark-CVPR24.git
cd NeRF-SLAM-Benchmark-CVPR24
conda env create -f environment.yaml
conda activate benchmark-release
```

Alternatively, if you run into installation errors, we recommend the following commands:

```bash
# Create conda environment
conda create -n benchmark-release python=3.7
conda activate benchmark-release

# Install the pytorch first (Please check the cuda version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install all the dependencies via pip (Note here pytorch3d and tinycudann requires ~10min to build)
pip install -r requirements.txt

# For tinycudann, if you cannot access network when you use GPUs, you can also try build from source as below:
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# Build extension (marching cubes from neuralRGBD)
cd external/NumpyMarchingCubes
python setup.py install
```


## Quick Run

Download the data as below and the data is saved into the `Datasets/` folder,
```bash
mkdir -p Datasets
cd Datasets
# Replica Dataset for Lab Scenario
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
# NeuralRGBD Dataset for Practical Scenario
wget http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
unzip neural_rgbd_data.zip
```
and then you can run benchmark for, e.g. `room0` sequence of Replica Dataset by:
```bash
python run_benchmark.py configs/Replica/room0.yaml
```

## Evaluation

### ATE/PSNR/DepthL1/Time
To monitor the evaluation metrics throughout SLAM running, please use tensorboard:
```bash
tensorboard --logdir output/Replica/room0/ --port=8008
```

### Acc./Comp.
Upon finishing the SLAM process, you will see the final reconstruction in the folder with `.ply` ending. To measure the quality of the reconstruction, we recommend you to evaluate the meshes using the repository [here](https://github.com/JingwenWang95/neural_slam_eval). Note that our results are evaluated by its `Neural-RGBD/GO-Surf` mesh culling strategy:

```bash
## Replica ##
# mesh culling
INPUT_MESH=/home/ps/data/tongyanhua/NeRF-SLAM-Benchmark-CVPR24/Replica/room0/mesh_track1999.ply # path_to_your_mesh
python cull_mesh.py --config configs/Replica/room0.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose
# mesh evaluation
REC_MESH=/home/ps/data/tongyanhua/NeRF-SLAM-Benchmark-CVPR24/Replica/room0/mesh_track1999_cull_occlusion.ply # path_to_your_culled_mesh
GT_MESH=/home/ps/data/tongyanhua/Datasets/Replica/room0_mesh.ply # path_to_gt_mesh: we use gt mesh without any culling for replica evaluation
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d


## Neural RGBD ##
INPUT_MESH=/home/ps/data/tongyanhua/NeRF-SLAM-Benchmark-CVPR24/synthetic/gr_mesh_track1441.ply
python cull_mesh.py --config configs/Synthetic/gr.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose
REC_MESH=/home/ps/data/tongyanhua/NeRF-SLAM-Benchmark-CVPR24/synthetic/gr_mesh_track1441_cull_occlusion.ply
GT_MESH=/home/ps/data/tongyanhua/Datasets/neural_rgbd_data/green_room/gt_mesh_cull_virt_cams.ply # path_to_gt_mesh: we use culled gt mesh for neuralrgbd evaluation
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type RGBD -3d

```
You may need to download ground truth meshes by youself, following the instruction and data directories of section `Run Evaluation` in the [repository](https://github.com/JingwenWang95/neural_slam_eval).


### ScanNet for Explicit Hybrid Encoding

Please find our implementation [here](https://github.com/thua919/explicit_hybrid_encoding).


## Related Repositories
We would like to extend our gratitude to the authors of 
[NICE-SLAM](https://github.com/cvg/nice-slam), 
[ESLAM](https://github.com/idiap/ESLAM), 
[COSLAM](https://github.com/HengyiWang/Co-SLAM), 
[TensoRF](https://github.com/apchenstu/TensoRF), and 
[NeuS](https://github.com/Totoro97/NeuS) 
for their exceptional work.


## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{nerfslam24hua,
  author = {Johari, M. M. and Carta, C. and Fleuret, F.},
  title = {Benchmarking Implicit Neural Representation and Geometric Rendering in Real-Time RGB-D SLAM},
  booktitle = {Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2023},
}
```

### Acknowledgement
This paper is supported by the National Natural Science Foundation of China (NSF).

