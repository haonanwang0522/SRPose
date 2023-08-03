# Lightweight Super-Resolution Head for Human Pose Estimation [arxiv](https://arxiv.org/abs/2307.16765)

> [**Lightweight Super-Resolution Head for Human Pose Estimation**](https://arxiv.org/abs/2307.16765)<br>
> Accepted by **ACM MM 2023**<br>
> [Haonan Wang](https://github.com/haonanwang0522), Jie Liu, Jie Tang, [Gangshan Wu](http://mcg.nju.edu.cn/member/gswu/en/index.html)

## News!
- [2023.08.03] The pretrained models are released in [Google Drive](https://drive.google.com/drive/folders/1ErxLJjrvgXNuNaflN62fvS6nhJfmBEjH?usp=drive_link)!
- [2023.07.30] The codes for SRPose are released!
- [2023.07.29] Our paper ''Lightweight Super-Resolution Head for Human Pose Estimation'' has been accpeted by **ACM MM 2023**. If you find this repository useful please give it a star 🌟. 


## Introduction
This is the official implementation of [Lightweight Super-Resolution Head for Human Pose Estimation](https://arxiv.org/abs/2307.16765). We present a Lightweight Super-Resolution Head , which predicts heatmaps with a spatial resolution higher than the input feature maps (or even consistent with the input image) by super-resolution, to effectively reduce the quantization error and the dependence on further post-processing. Besides, we propose SRPose to gradually recover the HR heatmaps from LR heatmaps and degraded features in a coarse-to-fine manner. To reduce the training difficulty of HR heatmaps, SRPose applies SR heads to supervise the intermediate features in each stage. In addition, the SR head is a lightweight and generic head that applies to top-down and bottom-up methods. 

<img width="1183" alt="image" src="https://github.com/haonanwang0522/SRPose/blob/main/overall.png">

## Experiments

### Results on COCO validation set
<!-- <details> -->
<table>
	<tr>
	    <th rowspan="2">Backbone</th>
	    <th rowspan="2">Scheme</th>
	    <th rowspan="2">GFLOPs</th>  
	    <th colspan="2">Params</th>
	    <th colspan="2">w/ Post.</th>
	    <th colspan="2">w/o Post.</th>
	</tr >
	<tr >
	    <th>Backbone</th>
	    <th>Other</th>
	    <th>AP</th>
	    <th>AR</th>
	    <th>AP</th>
	    <th>AR</th>	
	</tr>
	<tr >
	    <th colspan="9">Top-down methods</th>
	</tr>
	<tr >
	    <td rowspan="3"><a href ="https://arxiv.org/abs/1804.06208">Resnet-50</a></td>
	    <td>Simple head</td>
	    <td>5.46</td>
	    <td>23.51M</td>
	    <td>10.49M</td>
	    <td>71.7</td>
	    <td>77.3</td>
	    <td>69.8</td>
		<td>75.8</td>		
	</tr>
	<tr>
	    <td>SR head (ours)</td>
	    <td>5.77</td>
	    <td>23.51M</td>
	    <td>10.59M</td>
	    <td>72.4</td>
	    <td>77.9</td>
	    <td>72.2</td>
		<td>77.7</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>4.61</td>
	    <td>23.51M</td>
	    <td>1.29M</td>
	    <td><b>73.3</b></td>
	    <td><b>78.8</b></td>
		<td>73.1</td>
	    <td>78.6</td>
	</tr>
	<tr >
	    <td rowspan="3"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>Simple head</td>
	    <td>7.70</td>
	    <td>28.54M</td>
	    <td>0.00M</td>
	    <td>74.5</td>
	    <td>79.9</td>
	    <td>72.3</td>
		<td>78.2</td>		
	</tr>
	<tr>
	    <td>SR head (ours)</td>
	    <td>7.98</td>
	    <td>28.54M</td>
	    <td>0.09M</td>
	    <td>75.6</td>
	    <td>80.6</td>
	    <td>75.4</td>
		<td>80.5</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>8.28</td>
	    <td>29.30M</td>
	    <td>0.65M</td>
	    <td><b>75.9</b></td>
	    <td><b>81.0</b></td>
		<td>75.7</td>
	    <td>80.9</td>
	</tr>
	<tr >
	    <td rowspan="3"><a href ="https://arxiv.org/abs/2012.14214">TransPose-R-A4</a></td>
	    <td>Simple head</td>
	    <td>8.91</td>
	    <td>4.93M</td>
	    <td>1.06M</td>
	    <td>71.8</td>
	    <td>77.3</td>
	    <td>69.7</td>
		<td>75.5</td>		
	</tr>
	<tr>
	    <td>SR head (ours)</td>
	    <td>9.23</td>
	    <td>4.93M</td>
	    <td>1.16M</td>
	    <td>73.2</td>
	    <td>78.4</td>
	    <td>73.1</td>
		<td>78.3</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>6.26</td>
	    <td>4.93M</td>
	    <td>0.55M</td>
	    <td><b>73.5</b></td>
	    <td><b>78.9</b></td>
		<td>73.4</td>
	    <td>78.7</td>
	</tr>
	<tr >
	    <td rowspan="3"><a href ="https://arxiv.org/abs/2110.09408">HRFormer-S</a></td>
	    <td>Simple head</td>
	    <td>2.82</td>
	    <td>7.89M</td>
	    <td>0.00M</td>
	    <td>74.0</td>
	    <td>79.2</td>
	    <td>72.1</td>
		<td>77.6</td>		
	</tr>
	<tr>
	    <td>SR head (ours)</td>
	    <td>3.09</td>
	    <td>7.89M</td>
	    <td>0.09M</td>
	    <td>75.0</td>
	    <td>80.1</td>
	    <td>74.8</td>
		<td>80.0</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>3.34</td>
	    <td>8.21M</td>
	    <td>0.65M</td>
	    <td><b>75.6</b></td>
	    <td><b>80.7</b></td>
		<td>75.5</td>
	    <td>80.6</td>
	</tr>
	<tr >
	    <th colspan="9">Bottpm-up methods</th>
	</tr>
	<tr >
	    <td rowspan="2"><a href ="https://arxiv.org/abs/1804.06208">Resnet-50</a></td>
	    <td>Simple head</td>
	    <td>29.20</td>
	    <td>23.51M</td>
	    <td>10.49M</td>
	    <td>46.7</td>
	    <td>55.1</td>
	    <td>-</td>
		<td>-</td>		
	</tr>
	<tr>
	    <td><b>SR head (ours)</b></td>
	    <td>30.86</td>
	    <td>23.51M</td>
	    <td>10.60M</td>
	    <td><b>48.4</b></td>
	    <td><b>56.6</b></td>
	    <td>-</td>
		<td>-</td>
	</tr>
	<tr >
	    <td rowspan="2"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>Simple head</td>
	    <td>41.10</td>
	    <td>28.54M</td>
	    <td>0.00M</td>
	    <td>65.3</td>
	    <td>70.9</td>
	    <td>-</td>
		<td>-</td>		
	</tr>
	<tr>
	    <td><b>SR head (ours)</b></td>
	    <td>42.57</td>
	    <td>28.54M</td>
	    <td>0.09M</td>
	    <td><b>67.1</b></td>
	    <td><b>71.7</b></td>
	    <td>-</td>
		<td>-</td>
	</tr>
</table>
</details>

#### Note:
* The resolution of input is 256x192 for top-down methods, 512x512 for bottom-up methods.
* Flip test is used.
* Person detector has person AP of 56.4 on COCO val2017 dataset for top-down methods.
* Post. = extra post-processing (empirical shift) towards refining the predicted keypoint coordinate.

### Results on MPII val set
<table>
	<tr>
	    <th>Method</th>
	    <th>Backbone</th>
	    <th>PCKh@0.5</th>
	</tr >	
	<tr >
	    <td><a href ="https://arxiv.org/abs/1804.06208">SimBa</a></td>
	    <td>Resnet-50</td>
	    <td>88.2</td>	
	</tr>
	<tr >
	    <td><a href ="https://arxiv.org/abs/1902.09212">HRNet</a></td>
	    <td>HRNet-W32</td>
	    <td>90.1</td>	
	</tr>
	<tr >
	    <td><a href ="https://arxiv.org/abs/2107.03332">SimCC</a></td>
	    <td>HRNet-W32</td>
	    <td>90.0</td>	
	</tr>
	<tr>
	    <td>SRPose (ours)</td>
	    <td>Resnet-50</td>
	    <td>89.1</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>HRNet-W32</td>
	    <td><b>90.5</b></td>
	</tr>
</table>

#### Note:
* Flip test is used.


### Results on CrowdPose
<table>
	<tr>
	    <th>Method</th>
	    <th>Backbone</th>
	    <th>AP</th>
	    <th>AP_E</th>
	    <th>AP_M</th>
	    <th>AP_H</th> 		
	</tr >
	<tr >
	    <td><a href ="https://arxiv.org/abs/1804.06208">SimBa</a></td>
	    <td>Resnet-50</td>
	    <td>63.7</td>
	    <td>73.9</td>
	    <td>65.0</td>
	    <td>50.6</td>		
	</tr>
	<tr>
	    <td><a href ="https://arxiv.org/abs/1902.09212">HRNet</a></td>
	    <td>HRNet-W32</td>
	    <td>66.4</td>
	    <td>74.0</td>
	    <td>67.4</td>
	    <td>55.6</b></td>		
	</tr>
	<tr>
	    <td><a href ="https://arxiv.org/abs/2107.03332">SimCC</a></td>
	    <td>HRNet-W32</td>
	    <td>66.7</td>	
	    <td>74.1</td>
	    <td>67.8</td>
	    <td><b>56.2</b></td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>Resnet-50</td>
	    <td>64.7</td>
	    <td>74.9</td>
	    <td>65.8</td>
	    <td>52.3</td>
	</tr>
	<tr>
	    <td><b>SRPose (ours)</b></td>
	    <td>HRNet-W32</td>
	    <td><b>67.8</b></td>
	    <td><b>77.5</b></td>
	    <td><b>69.1</b></td>
	    <td>55.6</td>
	</tr>
</table>

#### Note:
* Flip test is used.

## Start to use
### 1. Dependencies installation & data preparation
Please refer to [THIS](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) to prepare the environment step by step.

### 2. Model Zoo
Pretrained models are provided in our [model zoo](https://drive.google.com/drive/folders/1ErxLJjrvgXNuNaflN62fvS6nhJfmBEjH?usp=drive_link).

### 3. Trainging
```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch --seed 0
```

### 4. Testing
To test the pretrained models performance, please run 

```bash
bash tools/dist_test.sh <Config PATH> <Checkpoint PATH> <NUM GPUs>
```

## Acknowledgement
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [HRFormer](https://github.com/HRNet/HRFormer).

## Citations
If you use our code or models in your research, please cite with:
```
@article{wang2023lightweight,
  title={Lightweight Super-Resolution Head for Human Pose Estimation},
  author={Wang, Haonan and Liu, Jie and Tang, Jie and Wu, Gangshan},
  journal={arXiv preprint arXiv:2307.16765},
  year={2023}
}
```
