# Awesome Crowd Counting

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Misc](#misc)
* [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard)

## Misc

### Code
- [[C^3 Framework](https://github.com/gjy3035/C-3-Framework)] An open-source PyTorch code for crowd counting, which is released.

### Technical blog
- [2019.05] [Chinese Blog] C^3 Framework系列之一：一个基于PyTorch的开源人群计数框架 [[Link](https://zhuanlan.zhihu.com/p/65650998)]
- [2019.04] Crowd counting from scratch [[Link](https://github.com/CommissarMa/Crowd_counting_from_scratch)]
- [2017.11] Counting Crowds and Lines with AI [[Link1](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/)] [[Link2](https://count.dimroc.com/)] [[Code](https://github.com/dimroc/count)]

###  GT generation
- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)] [[Fast Python Code](https://github.com/vlad3996/computing-density-maps)] [[Pytorch CUDA Code]](https://github.com/gjy3035/NWPU-Crowd-Sample-Code/blob/master/misc/dot_ops.py)


## Datasets

### Free-view

| Name | Year | Attributes | Avg. Resolution | No. Samples | No. Instenaces | Avg. Cnt | Link | 
| --- | --- |  --- | --- |--- | --- | --- | --- |
| NWPU-Crowd | 2020 | Congested | 2311\*3383 | 5,109 | 2,133,238 | 418 | [[Homepage](https://www.crowdbenchmark.com/)]   [[Download](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EsubMp48wwJDiH0YlT82NYYBmY9L0s-FprrBcoaAJkI1rw?e=e2JLgD)] [[Code](https://github.com/gjy3035/NWPU-Crowd-Sample-Code/)]  |
| JHU-CROWD | 2019 | Congested | 1450\*900 | 4,250 | 1,114,785 | 262 | Unreleased |
| UCF-QNRF | 2018 | Congested | 2013\*2902 | 1,535 | 1,251,642 | 815 | [[Homepage](http://crcv.ucf.edu/data/ucf-qnrf/)] [[Download](https://drive.google.com/open?id=1fLZdOsOXlv2muNB_bXEW6t-IS9MRziL6)] |
| ShanghaiTech Part A | 2016 |  Congested | 589\*868 | 482 | 241,677 | 501 | Download: [[Link1](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)] [[Link2](https://pan.baidu.com/s/1nuAYslz)] |
| UCF_CC_50 | 2013 | Congested | 2101\*2888 | 50 | 63,974 | 1,279 | [[Homepage](http://crcv.ucf.edu/data/ucf-cc-50/)]|



### Surveillance-view

| Name | Year | Attributes | Avg. Resolution | No. Samples | No. Instenaces | Avg. Cnt | Link | 
| --- | --- |  --- | --- |--- | --- | --- | --- |
| Crowd Surveillance | 2019 | Free scenes | 840\*1342 | 13,945 | 386,513 | 28 | [[Homepage](https://ai.baidu.com/broad/introduction)] |
| ShanghaiTechRGBD | 2019 | Depth | - | - | - | - | [[Homepage](https://github.com/svip-lab/RGBD-Counting)] |
| Fudan-ShanghaiTech  | 2019 | Video | 1080\*1920 | 15,000 | 394,081 | 27 | [[Homepage](https://github.com/sweetyy83/Lstn_fdst_dataset)] [[Download](https://pan.baidu.com/share/init?surl=NNaJ1vtsxCPJUjDNhZ1sHA) (pwd:**sgt1**)] |
| GCC | 2019 | 400 Fixed Scenes, Synthetic | 1080\*1920 | 15,211 | 7,625,843 | 501 | Download: [[Link1](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/Eo4L82dALJFDvUdy8rBm6B0BuQk6n5akJaN1WUF1BAeKUA?e=ge2cRg)] [[Link2](https://v2.fangcloud.com/share/4625d2bfa9427708060b5a5981)] [[Link3](https://pan.baidu.com/s/1OtKqmw84TFbxAiN0H2xBtQ) (pwd:**utdo**)]|
| Venice | 2019 | 4 Fixed Scenes  | 720\*1280 | 167 | - | - |  [[Download](https://drive.google.com/file/d/15PUf7C3majy-BbWJSSHaXUlot0SUh3mJ/view)] |
| CityStreet | 2019 | Multi-view | 1520\*2704 | 500 | - | - |  [[Homepage](http://visal.cs.cityu.edu.hk/research/citystreet/)]  |
| Beijing-BRT | 2019 | 1 Fixed Scene | 640\*360 | 1,280 | 16,795 | 13 | [[Homepage](https://github.com/XMU-smartdsp/Beijing-BRT-dataset)] |
| SmartCity | 2018 | - | 1080\*1920 | 50 | 369 | 7 | Download: [[Link1](https://drive.google.com/file/d/1xqflSQv9dZ0A93_lP34pSIfcpheT2Fi8/view?usp=sharing)] [[Link2](https://pan.baidu.com/s/1pMuGyNp)] |
| CityUHK-X | 2017 | 55 Fixed Scenes | 384\*512 | 3,191 | 106,783 | 33 | [[Homepage](http://visal.cs.cityu.edu.hk/downloads/#cityuhk-x)] |
| ShanghaiTech Part B | 2016 |  Free Scenes | 768\*1024 | 716 | 88,488 | 123 | Download: [[Link1](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)] [[Link2](https://pan.baidu.com/s/1nuAYslz)] |
| AHU-Crowd | 2016 |  - | 720\*576 | 107 | 45,000 | 421 | [[Homepage](http://cs-chan.com/downloads_crowd_dataset.html)] |
| WorldExpo'10 | 2015 | 108 Fixed Scenes | 576\*720 | 3,980 | 199,923 | 50 | [[Homepage](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)] |
| Mall | 2012 | 1 Fixed Scene | 480\*640 | 2,000 | 62,325 | 31 | [[Homepage](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)] |
| UCSD | 2008 | 1 Fixed Scene  | 158\*238 | 2,000 | 49,885 | 25 | [[Homepage](http://www.svcl.ucsd.edu/projects/peoplecnt/)] |


### Drone-view
| Name | Year | Attributes | Avg. Resolution | No. Samples | No. Instenaces | Avg. Cnt | Link | 
| --- | --- |  --- | --- |--- | --- | --- | --- |
| DroneVehicle | 2020 | Vehicle | 840\*712 |  31,064 |  441,642 | 14.2 | [[Homepage](https://github.com/VisDrone/DroneVehicle)] |
| DroneCrowd | 2019 | Video | 1080\*1920 |  33,600 | 4,864,280 | 145 | [[Homepage](https://github.com/VisDrone/VisDrone-Dataset)] |
| DLR-ACD | 2019 | - | - | 33 |  226,291 | 6,857 | [[Homepage](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-58354/)] |



## Papers

### arXiv papers
Note that all unpublished arXiv papers are not included in [the leaderboard of performance](#performance).

- <a name=""></a>Efficient Crowd Counting via Structured Knowledge Transfer [[paper](https://arxiv.org/abs/2003.10120)]
- <a name=""></a>Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting [[paper](https://arxiv.org/abs/2003.05586)]
- <a name=""></a>Drone Based RGBT Vehicle Detection and Counting: A Challenge [[paper](https://arxiv.org/abs/2003.02437)]
- <a name=""></a>NAS-Count: Counting-by-Density with Neural Architecture Search [[paper](https://arxiv.org/abs/2003.00217)]
- <a name=""></a>Towards Using Count-level Weak Supervision for Crowd Counting [[paper](https://arxiv.org/abs/2003.00164)]
- <a name=""></a>ZoomCount: A Zooming Mechanism for Crowd Counting in Static Images [[paper](https://arxiv.org/abs/2002.12256)]
- <a name=""></a>NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting [[paper](https://arxiv.org/abs/2001.03360)][[code](https://github.com/gjy3035/NWPU-Crowd-Sample-Code)]
- <a name=""></a>PDANet: Pyramid Density-aware Attention Net for Accurate Crowd Counting [[paper](https://arxiv.org/abs/2001.05643)]
- <a name=""></a>From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting [[paper](https://arxiv.org/abs/2001.01886)](extension of [S-DCNet](#S-DCNet))
- <a name=""></a>AutoScale: Learning to Scale for Crowd Counting [[paper](https://arxiv.org/abs/1912.09632)](extension of [L2SM](#L2SM))
- <a name=""></a>Domain-adaptive Crowd Counting via Inter-domain Features Segregation and Gaussian-prior Reconstruction [[paper](https://arxiv.org/abs/1912.03677)]
- <a name=""></a>Feature-aware Adaptation and Structured Density Alignment for Crowd Counting in Video Surveillance [[paper](https://arxiv.org/abs/1912.03672)]
- <a name=""></a>Drone-based Joint Density Map Estimation, Localization and Tracking with Space-Time Multi-Scale Attention Network [[paper](https://arxiv.org/abs/1912.01811)][[code](https://github.com/VisDrone)]
<details>
<summary>Earlier ArXiv Papers</summary>
  
- Using Depth for Pixel-Wise Detection of Adversarial Attacks in Crowd Counting [[paper](https://arxiv.org/abs/1911.11484)]
- Estimating People Flows to Better Count them in Crowded Scenes [[paper](https://arxiv.org/abs/1911.10782)]
- Segmentation Guided Attention Network for Crowd Counting via Curriculum Learning [[paper](https://arxiv.org/abs/1911.07990)]
- Deep Density-aware Count Regressor [[paper](https://arxiv.org/abs/1908.03314)][[code](https://github.com/GeorgeChenZJ/deepcount)]
- Video Crowd Counting via Dynamic Temporal Modeling [[paper](https://arxiv.org/abs/1907.02198)]
- Dense Scale Network for Crowd Counting [[paper](https://arxiv.org/abs/1906.09707)][unofficial code: [PyTorch](https://github.com/rongliangzi/Dense-Scale-Network-for-Crowd-Counting)]
- Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection [[paper](https://arxiv.org/abs/1906.07538)][[code](https://github.com/val-iisc/lsc-cnn)]
- Content-aware Density Map for Crowd Counting and Density Estimation [[paper](https://arxiv.org/abs/1906.07258)]
- Crowd Transformer Network [[paper](https://arxiv.org/abs/1904.02774)]
- W-Net: Reinforced U-Net for Density Map Estimation [[paper](https://arxiv.org/abs/1903.11249)][[code](https://github.com/ZhengPeng7/W-Net-Keras)]
- Improving Dense Crowd Counting Convolutional Neural Networks using Inverse k-Nearest Neighbor Maps and Multiscale Upsampling [[paper](https://arxiv.org/abs/1902.05379)]
- Dual Path Multi-Scale Fusion Networks with Attention for Crowd Counting [[paper](https://arxiv.org/pdf/1902.01115.pdf)]
- Scale-Aware Attention Network for Crowd Counting [[paper](https://arxiv.org/pdf/1901.06026.pdf)]
- Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance [[paper](https://arxiv.org/abs/1808.07456)][[code](http://github.com/siyuhuang/crowdcount-stackpool)]
- Attention to Head Locations for Crowd Counting [[paper](https://arxiv.org/abs/1806.10287)]
- Crowd Counting with Density Adaption Networks [[paper](https://arxiv.org/abs/1806.10040)]
- Improving Object Counting with Heatmap Regulation [[paper](https://arxiv.org/abs/1803.05494)][[code](https://github.com/littleaich/heatmap-regulation)]
- Structured Inhomogeneous Density Map Learning for Crowd Counting [[paper](https://arxiv.org/pdf/1801.06642.pdf)]
- Image Crowd Counting Using Convolutional Neural Network and Markov Random Field [[paper](https://arxiv.org/abs/1706.03686)] [[code](https://github.com/hankong/crowd-counting)]
</details>

### Methods dealing with the lack of labelled data
- <a name="FSC"></a> **[FSC]** Focus on Semantic Consistency for Cross-domain Crowd Understanding (**ICASSP**) [[paper](https://arxiv.org/abs/2002.08623)]
- <a name="CCWld"></a> **[CCWld, SFCN]** Learning from Synthetic Data for Crowd Counting in the Wild (**CVPR2019**) [[paper](http://gjy3035.github.io/pdf/CC_Wild_0308_cvpr2019.pdf)] [[Project](https://gjy3035.github.io/GCC-CL/)] [[arxiv](https://arxiv.org/abs/1903.03303)]
- <a name="SL2R"></a>  **[SL2R]** Exploiting Unlabeled Data in CNNs by Self-supervised Learning to Rank (**T-PAMI**) [[paper](https://arxiv.org/abs/1902.06285)](extension of [L2R](#L2R))
- <a name="GWTA-CCNN"></a> **[GWTA-CCNN]** Almost Unsupervised Learning for Dense Crowd Counting (**AAAI2019**) [[paper](http://val.serc.iisc.ernet.in/valweb/papers/AAAI_2019_WTACNN.pdf)]
- <a name="CAC"></a>**[CAC]** Class-Agnostic Counting (**ACCV2018**) [[paper](https://arxiv.org/abs/1811.00472)] [[code](https://github.com/erikalu/class-agnostic-counting)]
- <a name="L2R"></a>  **[L2R]** Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (**CVPR2018**) [[paper](https://arxiv.org/abs/1803.03095)] [[code](https://github.com/xialeiliu/CrowdCountingCVPR18)] 
- <a name="SSR"></a> **[SSR]** From Semi-Supervised to Transfer Counting of Crowds (**ICCV2013**) [[paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Loy_From_Semi-supervised_to_2013_ICCV_paper.pdf)]

### Survey
- <a name=""></a> Beyond Counting：Comparisons of Density Maps for Crowd Analysis Tasks (**T-CSVT2018**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8360001)][[arxiv](https://arxiv.org/abs/1705.10118)]
- <a name=""></a> A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation (**PR Letters2018**) [[paper](https://arxiv.org/abs/1707.01202)]
- <a name=""></a> Advances and Trends in Visual Crowd Analysis: A Systematic Survey and Evaluation of Crowd Modelling Techniques (**Neurocomputing2016**) [[paper](https://home.isr.uc.pt/~jorge/wp-content/uploads/85.pdf)]
- <a name=""></a> An Evaluation of Crowd Counting Methods, Features and Regression Models (**CVIU2015**) [[paper](https://eprints.qut.edu.au/75845/4/75845.pdf)]
- <a name=""></a> Crowded Scene Analysis：A Survey (**T-CSVT2015**) [[paper](https://arxiv.org/pdf/1502.01812.pdf)]
- <a name=""></a> Recent survey on crowd density estimation and counting for visual surveillance (**Artificial Intelligence2015**) [[paper](https://www.sciencedirect.com/science/article/pii/S0952197615000081)]
- <a name=""></a> A Survey of Human-Sensing: Methods for Detecting Presence, Count, Location, Track, and Identity (**CSUR2010**) [[paper](https://papers.ger.sh/Teixeira-SurveyHumanSensing-2010.pdf)]

### 2020

- <a name="HSRNet"></a> **[HSRNet]** Crowd Counting via Hierarchical Scale Recalibration Network (**ECAI**) [[paper](https://arxiv.org/abs/2003.03545)]
- <a name="MSPNET"></a> **[MSPNET]** Multi-supervised Parallel Network for Crowd Counting (**ICASSP**) [[paper](https://crabwq.github.io/pdf/2020%20MSPNET%20Multi-supervised%20Parallel%20Network%20for%20Crowd%20Counting.pdf)]
- <a name="ASPDNet"></a> **[ASPDNet]** Counting dense objects in remote sensing images (**ICASSP**) [[paper](https://arxiv.org/abs/2002.05928)]
- <a name="FSC"></a> **[FSC]** Focus on Semantic Consistency for Cross-domain Crowd Understanding (**ICASSP**) [[paper](https://arxiv.org/abs/2002.08623)]
- <a name="C-CNN"></a> **[C-CNN]** A Real-Time Deep Network for Crowd Counting (**ICASSP**) [[paper](https://arxiv.xilesou.top/abs/2002.06515)]
- <a name="HyGnn"></a> **[HyGnn]** Hybrid  Graph  Neural  Networks  for  Crowd  Counting (**AAAI**) [[paper](https://arxiv.org/abs/2002.00092)]
- <a name="DUBNet"></a> **[DUBNet]** Crowd Counting with Decomposed Uncertainty (**AAAI**) [[paper](https://arxiv.org/abs/1903.07427)]
- <a name="SDANet"></a> **[SDANet]** Shallow  Feature  based  Dense  Attention  Network  for  Crowd  Counting (**AAAI**) [[paper](http://wrap.warwick.ac.uk/130173/1/WRAP-shallow-feature-dense-attention-crowd-counting-Han-2019.pdf)]
- <a name="3DCC"></a> **[3DCC]** 3D Crowd Counting via Multi-View Fusion with 3D Gaussian Kernels (**AAAI**) [[paper](https://arxiv.org/abs/2003.08162)][[Project](http://visal.cs.cityu.edu.hk/research/aaai20-3d-counting/)]
- <a name="FFSA"></a> **[FSSA]** Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning (**WACV**) [[paper](https://arxiv.org/abs/2002.00264)]
- <a name="CC-Mod"></a> **[CC-Mod]** Plug-and-Play Rescaling Based Crowd Counting in Static Images (**WACV**) [[paper](https://arxiv.org/abs/2001.01786)]
- <a name="CLPNet"></a> **[CLPNet]** Cross-Level Parallel Network for Crowd Counting (**TII**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798674)]
- <a name="HA-CCN"></a> **[HA-CCN]** HA-CCN: Hierarchical Attention-based Crowd Counting Network (**TIP**) [[paper](https://arxiv.org/abs/1907.10255)]
- <a name="PaDNet"></a> **[PaDNet]** PaDNet: Pan-Density Crowd Counting (**TIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8897143)]
- <a name="MS-GAN"></a> **[MS-GAN]** Adversarial Learning for Multiscale Crowd Counting Under Complex Scenes (**TCYB**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8949751)]
- <a name="MLSTN"></a> **[MLSTN]** Multi-level feature fusion based Locality-Constrained Spatial Transformer network for video crowd counting (**Neurocomputing**) [[paper](https://sciencedirect.xilesou.top/science/article/abs/pii/S0925231220301454)](extension of [LSTN](#LSTN))
- <a name="SRN+PS"></a> **[SRN+PS]** Scale-Recursive Network with point supervision for crowd scene analysis (**Neurocomputing**) [[paper](https://sciencedirect.xilesou.top/science/article/abs/pii/S0925231219317795)]
- <a name="ASDF"></a> **[ASDF]** Counting crowds with varying densities via adaptive scenario discovery framework (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220302356)](extension of [ASD](#ASD))
- <a name="CAT-CNN"></a> **[CAT-CNN]** Crowd counting with crowd attention convolutional neural network (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219316662)]

### 2019

- <a name="D-ConvNet"></a> **[D-ConvNet]** Nonlinear Regression via Deep Negative Correlation Learning (**T-PAMI**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8850209)](extension of [D-ConvNet](#D-ConvNet))[[Project](https://mmcheng.net/dncl/)]
- <a name=""></a>Generalizing semi-supervised generative adversarial networks to regression using feature contrasting (**CVIU**)[[paper](https://arxiv.org/abs/1811.11269)]
- <a name="CG-DRCN"></a> **[CG-DRCN]** Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and
Benchmark Method (**ICCV**)[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sindagi_Pushing_the_Frontiers_of_Unconstrained_Crowd_Counting_New_Dataset_and_ICCV_2019_paper.pdf)]
- <a name="ADMG"></a> **[ADMG]** Adaptive Density Map Generation for Crowd Counting (**ICCV**)[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wan_Adaptive_Density_Map_Generation_for_Crowd_Counting_ICCV_2019_paper.pdf)]
- <a name="DSSINet"></a> **[DSSINet]** Crowd Counting with Deep Structured Scale Integration Network (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Crowd_Counting_With_Deep_Structured_Scale_Integration_Network_ICCV_2019_paper.pdf)][[code](https://github.com/Legion56/Counting-ICCV-DSSINet)] 
- <a name="RANet"></a> **[RANet]** Relational Attention Network for Crowd Counting (**ICCV**)[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Relational_Attention_Network_for_Crowd_Counting_ICCV_2019_paper.pdf)]
- <a name="ANF"></a> **[ANF]** Attentional Neural Fields for Crowd Counting (**ICCV**)[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Attentional_Neural_Fields_for_Crowd_Counting_ICCV_2019_paper.pdf)]
- <a name="SPANet"></a> **[SPANet]** Learning Spatial Awareness to Improve Crowd Counting (**ICCV(oral)**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Learning_Spatial_Awareness_to_Improve_Crowd_Counting_ICCV_2019_paper.pdf)]
- <a name="MBTTBF"></a> **[MBTTBF]** Multi-Level Bottom-Top and Top-Bottom Feature Fusion for Crowd Counting (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sindagi_Multi-Level_Bottom-Top_and_Top-Bottom_Feature_Fusion_for_Crowd_Counting_ICCV_2019_paper.pdf)]
- <a name="CFF"></a> **[CFF]** Counting with Focus for Free (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Counting_With_Focus_for_Free_ICCV_2019_paper.pdf)][[code](https://github.com/shizenglin/Counting-with-Focus-for-Free)] 
- <a name="L2SM"></a> **[L2SM]** Learn to Scale: Generating Multipolar Normalized Density Map for Crowd Counting (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Learn_to_Scale_Generating_Multipolar_Normalized_Density_Maps_for_Crowd_ICCV_2019_paper.pdf)]
- <a name="S-DCNet"></a> **[S-DCNet]** From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_From_Open_Set_to_Closed_Set_Counting_Objects_by_Spatial_ICCV_2019_paper.pdf)][[code](https://github.com/xhp-hust-2018-2011/S-DCNet)]
- <a name="BL"></a> **[BL]** Bayesian Loss for Crowd Count Estimation with Point Supervision (**ICCV(oral)**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ma_Bayesian_Loss_for_Crowd_Count_Estimation_With_Point_Supervision_ICCV_2019_paper.pdf)][[code](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)] 
- <a name="PGCNet"></a> **[PGCNet]** Perspective-Guided Convolution Networks for Crowd Counting (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Perspective-Guided_Convolution_Networks_for_Crowd_Counting_ICCV_2019_paper.pdf)][[code](https://github.com/Zhaoyi-Yan/PGCNet)]
- <a name="SACANet"></a> **[SACANet]** Crowd Counting on Images with Scale Variation and Isolated Clusters (**ICCVW**) [[paper](https://arxiv.org/abs/1909.03839)]
- <a name="McML"></a> **[McML]** Improving the Learning of Multi-column Convolutional Neural Network for Crowd Counting (**ACM MM**) [[paper](https://dl.acm.org/citation.cfm?doid=3343031.3350898)]
- <a name="DADNet"></a> **[DADNet]** DADNet: Dilated-Attention-Deformable ConvNet for Crowd Counting (**ACM MM**) [[paper](https://dl.acm.org/citation.cfm?doid=3343031.3350881)] 
- <a name="MRNet"></a> **[MRNet]** Crowd Counting via Multi-layer Regression (**ACM MM**) [[paper](https://dl.acm.org/citation.cfm?doid=3343031.3350914)]
- <a name="MRCNet"></a> **[MRCNet]** MRCNet: Crowd Counting and Density Map Estimation in Aerial and Ground Imagery (**BMVCW**)[[paper](https://arxiv.org/abs/1909.12743)]
- <a name="E3D"></a> **[E3D]** Enhanced 3D convolutional networks for crowd counting (**BMVC**) [[paper](https://arxiv.org/abs/1908.04121)]
- <a name="OSSS"></a> **[OSSS]** One-Shot Scene-Specific Crowd Counting (**BMVC**) [[paper](https://bmvc2019.org/wp-content/uploads/papers/0209-paper.pdf)]
- <a name="RAZ-Net"></a> **[RAZ-Net]** Recurrent Attentive Zooming for Joint Crowd Counting and Precise Localization (**CVPR**) [[paper](http://www.muyadong.com/paper/cvpr19_0484.pdf)]
- <a name="RDNet"></a> **[RDNet]** Density Map Regression Guided Detection Network for RGB-D Crowd Counting and Localization (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lian_Density_Map_Regression_Guided_Detection_Network_for_RGB-D_Crowd_Counting_CVPR_2019_paper.pdf)][[code](https://github.com/svip-lab/RGBD-Counting)] 
- <a name="RRSP"></a> **[RRSP]** Residual Regression with Semantic Prior for Crowd Counting (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wan_Residual_Regression_With_Semantic_Prior_for_Crowd_Counting_CVPR_2019_paper.pdf)][[code](https://github.com/jia-wan/ResidualRegression-pytorch)] 
- <a name="MVMS"></a> **[MVMS]** Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Wide-Area_Crowd_Counting_via_Ground-Plane_Density_Maps_and_Multi-View_Fusion_CVPR_2019_paper.pdf)] [[Project](http://visal.cs.cityu.edu.hk/research/cvpr2019wacc/)] [[Dataset&Code](http://visal.cs.cityu.edu.hk/research/citystreet/)]
- <a name="AT-CFCN"></a> **[AT-CFCN]** Leveraging Heterogeneous Auxiliary Tasks to Assist Crowd Counting (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Leveraging_Heterogeneous_Auxiliary_Tasks_to_Assist_Crowd_Counting_CVPR_2019_paper.pdf)]
- <a name="TEDnet"></a> **[TEDnet]** Crowd Counting and Density Estimation by Trellis Encoder-Decoder Networks (**CVPR**) [[paper](https://arxiv.org/abs/1903.00853)]
- <a name="CAN"></a> **[CAN]** Context-Aware Crowd Counting (**CVPR**) [[paper](https://arxiv.org/pdf/1811.10452.pdf)] [[code](https://github.com/weizheliu/Context-Aware-Crowd-Counting)]
- <a name="PACNN"></a> **[PACNN]** Revisiting Perspective Information for Efficient Crowd Counting (**CVPR**)[[paper](https://arxiv.org/abs/1807.01989v3)]
- <a name="PSDDN"></a> **[PSDDN]** Point in, Box out: Beyond Counting Persons in Crowds (**CVPR(oral)**)[[paper](https://arxiv.org/abs/1904.01333)]
- <a name="ADCrowdNet"></a> **[ADCrowdNet]** ADCrowdNet: An Attention-injective Deformable Convolutional Network for Crowd Understanding (**CVPR**) [[paper](https://arxiv.org/abs/1811.11968)]
- <a name="CCWld"></a> **[CCWld, SFCN]** Learning from Synthetic Data for Crowd Counting in the Wild (**CVPR**) [[paper](http://gjy3035.github.io/pdf/CC_Wild_0308_cvpr2019.pdf)] [[Project](https://gjy3035.github.io/GCC-CL/)] [[arxiv](https://arxiv.org/abs/1903.03303)]
- <a name="DG-GAN"></a> **[DG-GAN]** Dense Crowd Counting Convolutional Neural Networks with Minimal Data using Semi-Supervised Dual-Goal Generative Adversarial Networks (**CVPRW**)[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Weakly%20Supervised%20Learning%20for%20Real-World%20Computer%20Vision%20Applications/Olmschenk_Dense_Crowd_Counting_Convolutional_Neural_Networks_with_Minimal_Data_using_CVPRW_2019_paper.pdf)]
- <a name="GSP"></a> **[GSP]** Global Sum Pooling: A Generalization Trick for Object Counting with Small Datasets of Large Images (**CVPRW**)[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Deep%20Vision%20Workshop/Aich_Global_Sum_Pooling_A_Generalization_Trick_for_Object_Counting_with_CVPRW_2019_paper.pdf)]
- <a name="SL2R"></a>  **[SL2R]** Exploiting Unlabeled Data in CNNs by Self-supervised Learning to Rank (**T-PAMI**) [[paper](https://arxiv.org/abs/1902.06285)](extension of [L2R](#L2R))
- <a name="IA-DNN"></a> **[IA-DNN]** Inverse Attention Guided Deep Crowd Counting Network (**AVSS Best Paper**) [[paper](https://arxiv.org/abs/1907.01193)]
- <a name="MTCNet"></a> **[MTCNet]** MTCNET: Multi-task Learning Paradigm for Crowd Count Estimation (**AVSS**) [[paper](https://arxiv.org/abs/1908.08652)]
- <a name="CODA"></a> **[CODA]** CODA: Counting Objects via Scale-aware Adversarial Density Adaption (**ICME**) [[paper](https://arxiv.org/abs/1903.10442)][[code](https://github.com/Willy0919/CODA)]
- <a name="LSTN"></a> **[LSTN]** Locality-Constrained Spatial Transformer Network for Video Crowd Counting (**ICME(oral)**)  [[paper](https://arxiv.org/abs/1907.07911)]
- <a name="DRD"></a> **[DRD]** Dynamic Region Division for Adaptive Learning Pedestrian Counting (**ICME**) [[paper](https://arxiv.org/abs/1908.03978)]
- <a name="MVSAN"></a> **[MVSAN]** Crowd Counting via Multi-View Scale Aggregation Networks (**ICME**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8784912)]
- <a name="ASD"></a> **[ASD]** Adaptive Scenario Discovery for Crowd Counting (**ICASSP**) [[paper](https://arxiv.org/abs/1812.02393)]
- <a name="SAAN"></a> **[SAAN]** Crowd Counting Using Scale-Aware Attention Networks (**WACV**) [[paper](http://www.cs.umanitoba.ca/~ywang/papers/wacv19.pdf)]
- <a name="SPN"></a> **[SPN]** Scale Pyramid Network for Crowd Counting (**WACV**) [[paper](http://ieeexplore.ieee.org/xpl/mostRecentIssue.jsp?punumber=8642793)]
- <a name="GWTA-CCNN"></a> **[GWTA-CCNN]** Almost Unsupervised Learning for Dense Crowd Counting (**AAAI**) [[paper](http://val.serc.iisc.ernet.in/valweb/papers/AAAI_2019_WTACNN.pdf)]
- <a name="GPC"></a> **[GPC]** Geometric and Physical Constraints for Drone-Based Head Plane Crowd Density Estimation (**IROS**) [[paper](https://arxiv.org/abs/1803.08805)]
- <a name="PCC-Net"></a> **[PCC-Net]** PCC Net: Perspective Crowd Counting via Spatial Convolutional Network (**T-CSVT**) [[paper](https://arxiv.org/abs/1905.10085)] [[code](https://github.com/gjy3035/PCC-Net)]
- <a name="CLPC"></a> **[CLPC]** Cross-Line Pedestrian Counting Based on Spatially-Consistent Two-Stage Local Crowd Density Estimation and Accumulation (**T-CSVT**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8295124)]
- <a name="MAN"></a> **[MAN]** Mask-aware networks for crowd counting (**T-CSVT**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8796427)]
- <a name="CCLL"></a> **[CCLL]** Crowd Counting With Limited Labeling Through Submodular Frame Selection (**T-ITS**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8360780)]
- <a name="ACSPNet"></a> **[ACSPNet]** Atrous convolutions spatial pyramid network for crowd counting and density estimation (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219304059)]
- <a name="DDCN"></a> **[DDCN]** Removing background interference for crowd counting via de-background detail convolutional network (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231218315042)]
- <a name="MRA-CNN"></a> **[MRA-CNN]** Multi-resolution attention convolutional neural network for crowd counting (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231218312542)]
- <a name="ACM-CNN"></a> **[ACM-CNN]** Attend To Count: Crowd Counting with Adaptive Capacity Multi-scale CNNs (**Neurocomputing**) [[paper](https://arxiv.org/abs/1908.02797)]
- <a name="SDA-MCNN"></a> **[SDA-MCNN]** Counting crowds using a scale-distribution-aware network and adaptive human-shaped kernel (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219314651)]
- <a name="DENet"></a> **[DENet]** DENet: A Universal Network for Counting Crowd with Varying Densities and Scales (**Neurocomputing**) [[paper](https://arxiv.org/abs/1904.08056)][[code](https://github.com/liuleiBUAA/DENet)]
- <a name="SCAR"></a> **[SCAR]** SCAR: Spatial-/Channel-wise Attention Regression Networks for Crowd Counting (**Neurocomputing**) [[paper](https://arxiv.org/abs/1908.03716)][[code](https://github.com/gjy3035/SCAR)]
- <a name="MLCNN"></a> **[GMLCNN]** Learning Multi-Level Density Maps for Crowd Counting (**TNNLS**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8848475)]
- <a name="LDL"></a> **[LDL]** Indoor Crowd Counting by Mixture of Gaussians Label Distribution Learning (**TIP**) [[paper](http://palm.seu.edu.cn/xgeng/files/tip19.pdf)]

### 2018

- <a name="SANet"></a> **[SANet]** Scale Aggregation Network for Accurate and Efficient Crowd Counting (**ECCV**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.pdf)]
- <a name="ic-CNN"></a> **[ic-CNN]** Iterative Crowd Counting (**ECCV**) [[paper](https://arxiv.org/abs/1807.09959)]
- <a name="CL"></a> **[CL]** Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds (**ECCV**) [[paper](https://arxiv.org/abs/1808.01050)]
- <a name="LCFCN"></a> **[LCFCN]**  Where are the Blobs: Counting by Localization with Point Supervision (**ECCV**) [[paper](https://arxiv.org/abs/1807.09856)] [[code](https://github.com/ElementAI/LCFCN)]
- <a name="CSR"></a> **[CSR]**  CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (**CVPR**) [[paper](https://arxiv.org/abs/1802.10062)] [[code](https://github.com/leeyeehoo/CSRNet-pytorch)]
- <a name="L2R"></a>  **[L2R]** Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (**CVPR**) [[paper](https://arxiv.org/abs/1803.03095)] [[code](https://github.com/xialeiliu/CrowdCountingCVPR18)] 
- <a name="ACSCP"></a> **[ACSCP]**  Crowd Counting via Adversarial Cross-Scale Consistency Pursuit  (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf)]   [unofficial code: [PyTorch](https://github.com/RQuispeC/pytorch-ACSCP)]
- <a name="DecideNet"></a> **[DecideNet]** DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density (**CVPR**) [[paper](https://arxiv.org/abs/1712.06679)]
- <a name="AMDCN"></a>  **[AMDCN]** An Aggregated Multicolumn Dilated Convolution Network for Perspective-Free Counting (**CVPRW**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Deb_An_Aggregated_Multicolumn_CVPR_2018_paper.pdf)] [[code](https://github.com/diptodip/counting)] 
- <a name="D-ConvNet"></a> **[D-ConvNet]** Crowd Counting with Deep Negative Correlation Learning (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf)] [[code](https://github.com/shizenglin/Deep-NCL)]
- <a name="IG-CNN"></a> **[IG-CNN]** Divide and Grow: Capturing Huge Diversity in Crowd Images with
Incrementally Growing CNN (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sam_Divide_and_Grow_CVPR_2018_paper.pdf)]
- <a name="SCNet"></a>**[SCNet]** In Defense of Single-column Networks for Crowd Counting (**BMVC**) [[paper](https://arxiv.org/abs/1808.06133)]
- <a name="AFP"></a>**[AFP]** Crowd Counting by Adaptively Fusing Predictions from an Image Pyramid (**BMVC**) [[paper](https://arxiv.org/abs/1805.06115)]
- <a name="DRSAN"></a>**[DRSAN]** Crowd Counting using Deep Recurrent Spatial-Aware Network (**IJCAI**) [[paper](https://arxiv.org/abs/1807.00601)]
- <a name="TDF-CNN"></a>**[TDF-CNN]** Top-Down Feedback for Crowd Counting Convolutional Neural Network (**AAAI**) [[paper](https://arxiv.org/abs/1807.08881)]
- <a name="CAC"></a>**[CAC]** Class-Agnostic Counting (**ACCV**) [[paper](https://arxiv.org/abs/1811.00472)] [[code](https://github.com/erikalu/class-agnostic-counting)]
- <a name="A-CCNN"></a> **[A-CCNN]** A-CCNN: Adaptive CCNN for Density Estimation and Crowd Counting (**ICIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451399)]
- <a name=""></a> Crowd Counting with Fully Convolutional Neural Network (**ICIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451787)]
- <a name="MS-GAN"></a> **[MS-GAN]** Multi-scale Generative Adversarial Networks for Crowd Counting (**ICPR**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545683)]
- <a name="DR-ResNet"></a> **[DR-ResNet]** A Deeply-Recursive Convolutional Network for Crowd Counting (**ICASSP**) [[paper](https://arxiv.org/abs/1805.05633)] 
- <a name="GAN-MTR"></a> **[GAN-MTR]** Crowd Counting With Minimal Data Using Generative Adversarial Networks For Multiple Target Regression (**WACV**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354235)]
- <a name="SaCNN"></a> **[SaCNN]** Crowd counting via scale-adaptive convolutional neural network (**WACV**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354231)] [[code](https://github.com/miao0913/SaCNN-CrowdCounting-Tencent_Youtu)]
- <a name="Improved SaCNN"></a> **[Improved SaCNN]** Improved Crowd Counting Method Based on Scale-Adaptive Convolutional Neural Network (**IEEE Access**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8643345)]
- <a name="DA-Net"></a> **[DA-Net]** DA-Net: Learning the Fine-Grained Density Distribution With Deformation Aggregation Network (**IEEE Access**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8497050)][[code](https://github.com/BigTeacher-777/DA-Net)]
- <a name="BSAD"></a> **[BSAD]** Body Structure Aware Deep Crowd Counting (**TIP**) [[paper](http://mac.xmu.edu.cn/rrji/papers/IP%202018-Body.pdf)] 
- <a name="NetVLAD"></a> **[NetVLAD]** Multiscale Multitask Deep NetVLAD for Crowd Counting (**TII**) [[paper](https://staff.fnwi.uva.nl/z.shi/files/counting-netvlad.pdf)] [[code](https://github.com/shizenglin/Multitask-Multiscale-Deep-NetVLAD)]
- <a name="W-VLAD"></a> **[W-VLAD]** Crowd Counting via Weighted VLAD on Dense Attribute Feature Maps (**T-CSVT**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7778134)]

### 2017

- <a name="ACNN"></a> **[ACNN]** Incorporating Side Information by Adaptive Convolution (**NIPS**) [[paper](http://papers.nips.cc/paper/6976-incorporating-side-information-by-adaptive-convolution.pdf)][[Project](http://visal.cs.cityu.edu.hk/research/acnn/)]
- <a name="CP-CNN"></a> **[CP-CNN]** Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (**ICCV**) [[paper](https://arxiv.org/abs/1708.00953)]
- <a name="ConvLSTM"></a> **[ConvLSTM]** Spatiotemporal Modeling for Crowd Counting in Videos (**ICCV**) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xiong_Spatiotemporal_Modeling_for_ICCV_2017_paper.pdf)]
- <a name="CMTL"></a> **[CMTL]** CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting (**AVSS**) [[paper](https://arxiv.org/abs/1707.09605)] [[code](https://github.com/svishwa/crowdcount-cascaded-mtl)]
- <a name="ResnetCrowd"></a> **[ResnetCrowd]** ResnetCrowd: A Residual Deep Learning Architecture for Crowd Counting, Violent Behaviour Detection and Crowd Density Level Classification (**AVSS**) [[paper](https://arxiv.org/abs/1705.10698)]
- <a name="SCNN"></a> **[Switching CNN]** Switching Convolutional Neural Network for Crowd Counting (**CVPR**) [[paper](https://arxiv.org/abs/1708.00199)] [[code](https://github.com/val-iisc/crowd-counting-scnn)]
- <a name="DAL-SVR"></a> **[DAL-SVR]** Boosting deep attribute learning via support vector regression for fast moving crowd counting (**PR Letters**) [[paper](https://www.sciencedirect.com/science/article/pii/S0167865517304415)]
- <a name="MSCNN"></a> **[MSCNN]** Multi-scale Convolution Neural Networks for Crowd Counting (**ICIP**) [[paper](https://arxiv.org/abs/1702.02359)] [[code](https://github.com/Ling-Bao/mscnn)]
- <a name="FCNCC"></a> **[FCNCC]** Fully Convolutional Crowd Counting On Highly Congested Scenes (**VISAPP**) [[paper](https://arxiv.org/abs/1612.00220)] 

### 2016 

- <a name="Hydra-CNN"></a> **[Hydra-CNN]** Towards perspective-free object counting with deep learning  (**ECCV**) [[paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf)] [[code](https://github.com/gramuah/ccnn)]
- <a name="CNN-Boosting"></a> **[CNN-Boosting]** Learning to Count with CNN Boosting (**ECCV**) [[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-46475-6_41)] 
- <a name="Crossing-line"></a> **[Crossing-line]** Crossing-line Crowd Counting with Two-phase Deep Neural Networks (**ECCV**) [[paper](https://www.ee.cuhk.edu.hk/~xgwang/papers/ZhaoLZWeccv16.pdf)] 
- <a name="GP"></a> **[GP]** Gaussian Process Density Counting from Weak Supervision (**ECCV**) [[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-46448-0_22)]
- <a name="CrowdNet"></a> **[CrowdNet]** CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (**ACMMM**) [[paper](https://arxiv.org/abs/1608.06197)] [[code](https://github.com/davideverona/deep-crowd-counting_crowdnet)]
- <a name="MCNN"></a> **[MCNN]** Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (**CVPR**) [[paper](https://pdfs.semanticscholar.org/7ca4/bcfb186958bafb1bb9512c40a9c54721c9fc.pdf)] [unofficial code: [TensorFlow](https://github.com/aditya-vora/crowd_counting_tensorflow) [PyTorch](https://github.com/svishwa/crowdcount-mcnn)]
- <a name="Shang2016"></a> **[Shang 2016]** End-to-end crowd counting via joint learning local and global count (**ICIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7532551)]
- <a name="DE-VOC"></a> **[DE-VOC]** Fast visual object counting via example-based density estimation (**ICIP**) [[paper](http://web.pkusz.edu.cn/adsp/files/2015/10/Fast-Visual-Object-Counting-via-Example-based-Density-Estimation.pdf)] 
- <a name="RPF"></a> **[RPF]** Crowd Density Estimation based on Rich Features and Random Projection Forest (**WACV**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7477682)] 
- <a name="CS-SLR"></a> **[CS-SLR]** Cost-sensitive sparse linear regression for crowd counting with imbalanced training data (**ICME**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552905)] 
- <a name="Faster-OHEM-KCF"></a> **[Faster-OHEM-KCF]** Deep People Counting with Faster R-CNN and Correlation Tracking (**ICME**) [[paper](https://dl.acm.org/citation.cfm?id=3007745)] 

### 2015

- <a name="COUNTForest"></a> **[COUNT Forest]** COUNT Forest: CO-voting Uncertain Number of Targets using Random Forest
for Crowd Density Estimation (**ICCV**) [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Pham_COUNT_Forest_CO-Voting_ICCV_2015_paper.pdf)]
- <a name="Bayesian"></a> **[Bayesian]** Bayesian Model Adaptation for Crowd Counts (**ICCV**) [[paper](https://ieeexplore.ieee.org/document/7410832?arnumber=7410832&tag=1)]
- <a name="Zhang2015"></a> **[Zhang 2015]** Cross-scene Crowd Counting via Deep Convolutional Neural Networks (**CVPR**) [[paper](https://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf)] [[code](https://github.com/wk910930/crowd_density_segmentation)]
- <a name="Wang2015"></a> **[Wang 2015]** Deep People Counting in Extremely Dense Crowds (**ACMMM**) [[paper](https://dl.acm.org/citation.cfm?id=2806337)]
- <a name="Fu2015"></a> **[FU 2015]** Fast crowd density estimation with convolutional neural networks (**Artificial Intelligence**) [[paper](https://www.sciencedirect.com/science/article/pii/S0952197615000871)]

### 2014

- <a name="Arteta2014"></a> **[Arteta 2014]** Interactive Object Counting (**ECCV**) [[paper](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/papers/8691/86910504.pdf)] 

### 2013

- <a name="Idrees2013"></a> **[Idrees 2013]** Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Idrees_Multi-source_Multi-scale_Counting_2013_CVPR_paper.pdf)]
- <a name="Ma2013"></a> **[Ma 2013]** Crossing the Line: Crowd Counting by Integer Programming with Local Features (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Ma_Crossing_the_Line_2013_CVPR_paper.pdf)]
- <a name="Chen2013"></a> **[Chen 2013]** Cumulative Attribute Space for Age and Crowd Density Estimation (**CVPR**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Chen_Cumulative_Attribute_Space_2013_CVPR_paper.pdf)]
- <a name="SSR"></a> **[SSR]** From Semi-Supervised to Transfer Counting of Crowds (**ICCV**) [[paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Loy_From_Semi-supervised_to_2013_ICCV_paper.pdf)]

### 2012

- <a name="Chen2012"></a> **[Chen 2012]** Feature mining for localised crowd counting (**BMVC**) [[paper](https://pdfs.semanticscholar.org/c5ec/65e36bccf8a64050d38598511f0352653d6f.pdf)]

### 2011

- <a name="Rodriguez2011"></a> **[Rodriguez 2011]** Density-aware person detection and tracking in crowds (**ICCV**) [[paper](https://hal-enpc.archives-ouvertes.fr/hal-00654266/file/ICCV11a.pdf)]

### 2010

- <a name="Lempitsky2010"></a> **[Lempitsky 2010]** Learning To Count Objects in Images (**NIPS**) [[paper](http://papers.nips.cc/paper/4043-learning-to-count-objects-in-images)]

### 2008

- <a name="Chan2008"></a> **[Chan 2008]** Privacy preserving crowd monitoring: Counting people without people models or tracking (**CVPR**) [[paper](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr08-peoplecnt.pdf)]



## Leaderboard
The section is being continually updated. Note that some values have superscript, which indicates their source. 


### ShanghaiTech Part A

| Year-Conference/Journal | Methods           | MAE   | MSE   | PSNR  | SSIM | Params | Pre-trained   Model |
| ---- | ------------------------------------ | ----- | ----- | ----- | ---- | ------ | ------------------- |
| 2016--CVPR | [MCNN](#MCNN)     | 110.2 | 173.2 | 21.4<sup>[CSR](#CSR)</sup> | 0.52<sup>[CSR](#CSR)</sup>  | 0.13M<sup>[SANet](#SANet)</sup>  | None  |
| 2017--AVSS | [CMTL](#CMTL)                                | 101.3 | 152.4 | -  | -  | -  | None        |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 90.4  | 135.0 | -  | -  | 15.11M<sup>[SANet](#SANet)</sup>  | VGG-16      |
| 2017--ICIP | [MSCNN](#MSCNN)                              | 83.8  | 127.4 | -  | -  | -  | -           |
| 2017--ICCV | [CP-CNN](#CP-CNN) | 73.6  | 106.4 | 21.72<sup>[CP-CNN](#CP-CNN)</sup> | 0.72<sup>[CP-CNN](#CP-CNN)</sup>  | 68.4M<sup>[SANet](#SANet)</sup>  | - |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                          | 97.5  | 145.1 | -  | -  | -  | -           |
| 2018--WACV | [SaCNN](#SaCNN)                              | 86.8  | 139.2 | -  | -  | -  | -           |
| 2018--CVPR | [ACSCP](#ACSCP)                              | 75.7  | 102.7 | -  | -  | 5.1M | None      |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                   | 73.5  | 112.3 | -  | -  | -  | VGG-16      |
| 2018--CVPR | [IG-CNN](#IG-CNN)                            | 72.5  | 118.2 | -  | -  | -  | VGG-16      |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example) | 72.0  | 106.6 | -  | -  | -  | VGG-16      |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)          | 73.6  | 112.0 | -  | -  | -  | VGG-16      |
| 2019--CVPRW| [GSP](#GSP) (one stage, efficient)           | 70.7  | 103.6 | -  | -  | -  | VGG-16      |
| 2018--IJCAI| [DRSAN](#DRSAN)                              | 69.3  | 96.4  | -  | -  | -  | -           |
| 2018--ECCV | [ic-CNN](#ic-CNN) (one stage)                | 69.8  | 117.3 | -  | -  | -  | -           |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)               | 68.5  | 116.2 | -  | -  | -  | -           |
| 2018--CVPR | [CSRNet](#CSR)   | 68.2  | 115.0 | 23.79 | 0.76 | 16.26M<sup>[SANet](#SANet)</sup> |VGG-16|
| 2018--ECCV | [SANet](#SANet)                              | 67.0  | 104.5 | -  | -  | 0.91M | None     |
| 2019--AAAI | [GWTA-CCNN](#GWTA-CCNN)                      | 154.7 | 229.4 | -  | -  | -  | -           |
| 2019--ICASSP | [ASD](#ASD)                                | 65.6  | 98.0  | -  | -  | -  | -           |
| 2019--ICCV | [CFF](#CFF)                                  | 65.2  | 109.4 | 25.4  | 0.78 | -     | -   |
| 2019--CVPR | [SFCN](#CCWld)                               | 64.8  | 107.5 | -  | -  | -  | -           |
| 2019--ICCV | [SPN+L2SM](#L2SM)                            | 64.2  | 98.4  | -  | -  | -  | -           |
| 2019--CVPR | [TEDnet](#TEDnet)                            | 64.2  | 109.1 | 25.88 | 0.83 | 1.63M | -   |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-bAttn-DME)     | 63.2  | 98.9  | 24.48 | 0.88 | -     | -   |
| 2019--CVPR | [PACNN](#PACNN)                              | 66.3  | 106.4 | -  | -  | -  | -           |
| 2019--CVPR | [PACNN+CSRNet](#PACNN)                       | 62.4  | 102.0 | -  | -  | -  | -           |
| 2019--CVPR | [CAN](#CAN)                                  | 62.3  | 100.0 | -  | -  | -  | VGG-16      |
| 2019--TIP  | [HA-CCN](#HA-CCN)                            | 62.9  | 94.9  | -  | -  | -  | -           |
| 2019--ICCV | [BL](#BL)                                    | 62.8  | 101.8 | -  | -  | -  | -           |
| 2019--WACV | [SPN](#SPN)                                  | 61.7  | 99.5  | -  | -  | -  | -           |
| 2019--ICCV | [DSSINet](#DSSINet)                          | 60.63 | 96.04 | -  | -  | -  | -           |
| 2019--ICCV | [MBTTBF-SCFB](#MBTTBF)                       | 60.2  | 94.1  | -  | -  | -  | -           |
| 2019--ICCV | [RANet](#RANet)                              | 59.4  | 102.0 | -  | -  | -  | -           |
| 2019--ICCV | [SPANet+SANet](#SPANet)                      | 59.4  | 92.5  | -  | -  | -  | -           |
| 2019--TIP  | [PaDNet](#PaDNet)                            | 59.2  | 98.1  | -  | -  | -  | -           |
| 2019--ICCV | [S-DCNet](#S-DCNet)                          | 58.3  | 95.0  | -  | -  | -  | -           |
| 2019--ICCV |**[PGCNet](#PGCNet)**                         | **57.0** | **86.0** | -  | -  | -  | -         |


### ShanghaiTech Part B

| Year-Conference/Journal | Methods                          | MAE   | MSE   |
| ---- | ---------------- | ----- | ---- |
| 2016--CVPR | [MCNN](#MCNN)                                 | 26.4  | 41.3  |
| 2017--ICIP | [MSCNN](#MSCNN)                               | 17.7  | 30.2  |
| 2017--AVSS | [CMTL](#CMTL)                                 | 20.0  | 31.1  |
| 2017--CVPR | [Switching CNN](#SCNN)                        | 21.6  | 33.4  |
| 2017--ICCV | [CP-CNN](#CP-CNN)                             | 20.1  | 30.1  |
| 2018--TIP  | [BSAD](#BSAD)                                 | 20.2  | 35.6  |
| 2018--WACV | [SaCNN](#SaCNN)                               | 16.2  | 25.8  |
| 2018--CVPR | [ACSCP](#ACSCP)                               | 17.2  | 27.4  |
| 2018--CVPR | [CSRNet](#CSR)                                | 10.6  | 16.0  |
| 2018--CVPR | [IG-CNN](#IG-CNN)                             | 13.6  | 21.1  |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                    | 18.7  | 26.0  |
| 2018--CVPR | [DecideNet](#DecideNet)                       | 21.53 | 31.98 |
| 2018--CVPR | [DecideNet + R3](#DecideNet)                  | 20.75 | 29.42 |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example)  | 14.4  | 23.8  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)           | 13.7  | 21.4  |
| 2018--IJCAI| [DRSAN](#DRSAN)                               | 11.1  | 18.2  |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                           | 20.7  | 32.8  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (one stage)                 | 10.4  | 16.7  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)                | 10.7  | 16.0  |
| 2019--CVPRW| [GSP](#GSP) (one stage, efficient)            | 9.1   | 15.9  |
| 2018--ECCV | [SANet](#SANet)                               | 8.4   | 13.6  |
| 2019--WACV | [SPN](#SPN)                                   | 9.4   | 14.4  |
| 2019--ICCV | [PGCNet](#PGCNet)                             | 8.8   | 13.7  |
| 2019--ICASSP | [ASD](#ASD)                                 | 8.5   | 13.7  |
| 2019--CVPR | [TEDnet](#TEDnet)                             | 8.2   | 12.8  |
| 2019--TIP  | [HA-CCN](#HA-CCN)                             | 8.1   | 13.4  |
| 2019--TIP  | [PaDNet](#PaDNet)                             | 8.1   | 12.2  |
| 2019--ICCV | [RANet](#RANet)                               | 7.9   | 12.9  |
| 2019--CVPR | [CAN](#CAN)                                   | 7.8   | 12.2  |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-attn-DME)       | 7.7   | 12.9  |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-DME)            | 7.6   | 13.9  |
| 2019--CVPR | [SFCN](#CCWld)                                | 7.6   | 13.0  |
| 2019--CVPR | [PACNN](#PACNN)                               | 8.9   | 13.5  |
| 2019--CVPR | [PACNN+CSRNet](#PACNN)                        | 7.6   | 11.8  |
| 2019--ICCV | [BL](#BL)                                     | 7.7   | 12.7  |
| 2019--ICCV | [CFF](#CFF)                                   | 7.2   | 12.2  |
| 2019--ICCV | [SPN+L2SM](#L2SM)                             | 7.2   | 11.1  |
| 2019--ICCV | [DSSINet](#DSSINet)                           | 6.85  | 10.34 |
| 2019--ICCV | [S-DCNet](#S-DCNet)                           | 6.7   | 10.7  |
| 2019--ICCV | **[SPANet+SANet](#SPANet)**                   | **6.5** | **9.9** |

### UCF-QNRF

| Year-Conference/Journal | Method | C-MAE | C-NAE | C-MSE | DM-MAE | DM-MSE | DM-HI | L- Av. Precision	| L-Av. Recall | L-AUC |
| --- | --- | --- | --- |--- | --- | --- |--- | --- | --- | ---|
| 2013--CVPR | [Idrees 2013](#Idrees2013)<sup>[CL](#CL)</sup>| 315 | 0.63 | 508 | - | - | - | - | - | - |
| 2016--CVPR | [MCNN](#MCNN)<sup>[CL](#CL)</sup> | 277 | 0.55 | 426 |0.006670| 0.0223 | 0.5354 |59.93% | 63.50% | 0.591|
| 2017--AVSS | [CMTL](#CMTL)<sup>[CL](#CL)</sup>            | 252 | 0.54 | 514 | 0.005932 | 0.0244 | 0.5024 | - | - | - |
| 2017--CVPR | [Switching CNN](#SCNN)<sup>[CL](#CL)</sup>   | 228 | 0.44 | 445 | 0.005673 | 0.0263 | 0.5301 | - | - | - |
| 2018--ECCV | [CL](#CL)     | 132 | 0.26 | 191 | 0.00044| 0.0017 | 0.9131 | 75.8% | 59.75%	| 0.714|
| 2019--TIP  | [HA-CCN](#HA-CCN)   | 118.1 | - | 180.4 | - | - | - | - | - | - |
| 2019--CVPR | [TEDnet](#TEDnet)   | 113 | - | 188 | - | - | - | - | - | - |
| 2019--ICCV | [RANet](#RANet)     | 111 | - | 190 | - | - | - | - | - | - |
| 2019--CVPR | [CAN](#CAN)         | 107 | - | 183 | - | - | - | - | - | - |
| 2019--ICCV | [SPN+L2SM](#L2SM)   | 104.7 | - | 173.6 | - | - | - | - | - | - |
| 2019--ICCV | [S-DCNet](#S-DCNet) | 104.4 | - | 176.1 | - | - | - | - | - | - |
| 2019--CVPR | [SFCN](#CCWld)  | 102.0 | - | 171.4 | - | - | - | - | - | - |
| 2019--ICCV | [DSSINet](#DSSINet)  | 99.1 | - | 159.2 | - | - | - | - | - | - |
| 2019--ICCV | [MBTTBF-SCFB](#MBTTBF)      | 97.5 | - | 165.2 | - | - | - | - | - | - |
| 2019--TIP  | [PaDNet](#PaDNet)           | 96.5 | - | 170.2 | - | - | - | - | - | - |
| 2019--ICCV | **[BL](#BL)**  | **88.7** | - | **154.8** | - | - | - | - | - | - |


### UCF_CC_50

| Year-Conference/Journal | Methods                         | MAE   | MSE   |
| ---- | ---------------- | ----- | ---- |
| 2013--CVPR | [Idrees 2013](#Idrees2013)                   | 468.0 | 590.3  |
| 2015--CVPR | [Zhang 2015](#Zhang2015)                     | 467.0 | 498.5  |
| 2016--ACM MM | [CrowdNet](#CrowdNet)                      | 452.5 |   -    |
| 2016--CVPR | [MCNN](#MCNN)                                | 377.6 | 509.1  |
| 2016--ECCV | [CNN-Boosting](#CNN-Boosting)                | 364.4 |   -    |
| 2016--ECCV | [Hydra-CNN](#Hydra-CNN)                      | 333.73| 425.26 |
| 2016--ICIP | [Shang 2016](#Shang2016)                     | 270.3 |   -    |
| 2017--ICIP | [MSCNN](#MSCNN)                              | 363.7 | 468.4  |
| 2017--AVSS | [CMTL](#CMTL)                                | 322.8 | 397.9  |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 318.1 | 439.2  |
| 2017--ICCV | [CP-CNN](#CP-CNN)                            | 298.8 | 320.9  |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)                     | 284.5 | 297.1  |
| 2018--TIP  | [BSAD](#BSAD)                                | 409.5 | 563.7  |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                          | 354.7 | 491.4  |
| 2018--WACV | [SaCNN](#SaCNN)                              | 314.9 | 424.8  |
| 2018--CVPR | [IG-CNN](#IG-CNN)                            | 291.4 | 349.4  |
| 2018--CVPR | [ACSCP](#ACSCP)                              | 291.0 | 404.6  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example) | 291.5 | 397.6  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)          | 279.6 | 388.9  |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                   | 288.4 | 404.7  |
| 2018--CVPR | [CSRNet](#CSR)                               | 266.1 | 397.5  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)               | 260.9 | 365.5  |
| 2018--ECCV | [SANet](#SANet)                              | 258.4 | 334.9  |
| 2018--IJCAI| [DRSAN](#DRSAN)                              | 219.2 | 250.2  |
| 2019--AAAI | [GWTA-CCNN](#GWTA-CCNN)                      | 433.7 | 583.3  |
| 2019--WACV | [SPN](#SPN)                                  | 259.2 | 335.9  |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(DME)               | 257.1 | 363.5  |
| 2019--TIP  | [HA-CCN](#HA-CCN)                            | 256.2 | 348.4  |
| 2019--CVPR | [TEDnet](#TEDnet)                            | 249.4 | 354.5  |
| 2019--CVPR | [PACNN](#PACNN)                              | 267.9 | 357.8  |
| 2019--CVPR | [PACNN+CSRNet](#PACNN)                       | 241.7 | 320.7  |
| 2019--ICCV | [RANet](#RANet)                              | 239.8 | 319.4  |
| 2019--ICCV | [MBTTBF-SCFB](#MBTTBF)                       | 233.1 | 300.9  |
| 2019--ICCV | [BL](#BL)                                    | 229.3 | 308.2  |
| 2019--ICCV | [DSSINet](#DSSINet)                          | 216.9 | 302.4  |
| 2019--CVPR | [SFCN](#CCWld)                               | 214.2 | 318.2  |
| 2019--CVPR | **[CAN](#CAN)**                              | 212.2 | **243.7** |
| 2019--ICCV | [S-DCNet](#S-DCNet)                          | 204.2 | 301.3  |
| 2019--ICASSP| [ASD](#ASD)                                 | 196.2 | 270.9  |
| 2019--ICCV | [SPN+L2SM](#L2SM)                            | 188.4 | 315.3 |
| 2019--TIP  | **[PaDNet](#PaDNet)**                        | **185.8** | 278.3 |

### WorldExpo'10
| Year-Conference/Journal | Method | S1 | S2 | S3 | S4 | S5 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015--CVPR | [Zhang 2015](#Zhang2015)              | 9.8  | 14.1  | 14.3  | 22.2 | 3.7  | 12.9 |
| 2016--CVPR | [MCNN](#MCNN)                         | 3.4  | 20.6  | 12.9  | 13.0 | 8.1  | 11.6 |
| 2017--ICIP | [MSCNN](#MSCNN)                       | 7.8  | 15.4  | 14.9  | 11.8 | 5.8  | 11.7 |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)              | 8.6  | 16.9  | 14.6  | 15.4 | 4.0  | 11.9 |
| 2017--ICCV | [ConvLSTM](#ConvLSTM)                 | 7.1  | 15.2  | 15.2  | 13.9 | 3.5  | 10.9 |
| 2017--ICCV | [Bidirectional   ConvLSTM](#ConvLSTM) | 6.8  | 14.5  | 14.9  | 13.5 | 3.1  | 10.6 |
| 2017--CVPR | [Switching CNN](#SCNN)                | 4.4  | 15.7  | 10.0  | 11.0 | 5.9  | 9.4  |
| 2017--ICCV | [CP-CNN](#CP-CNN)                     | 2.9  | 14.7  | 10.5  | 10.4 | 5.8  | 8.86 |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                   | 2.7  | 23.4  | 10.7  | 17.6 | 3.3  | 11.5 |
| 2018--CVPR | [IG-CNN](#IG-CNN)                     | 2.6  | 16.1  | 10.15 | 20.2 | 7.6  | 11.3 |
| 2018--TIP  | [BSAD](#BSAD)                         | 4.1  | 21.7  | 11.9  | 11.0 | 3.5  | 10.5 |
| 2018--ECCV | [ic-CNN](#ic-CNN)                     | 17.0 | 12.3  | 9.2   | 8.1  | 4.7  | 10.3 |
| 2018--CVPR | [DecideNet](#DecideNet)               | 2.0  | 13.14 | 8.9   | 17.4 | 4.75 | 9.23 |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)            | 1.9  | 12.1  | 20.7  | 8.3  | 2.6  | 9.1  |
| 2018--CVPR | [CSRNet](#CSR)                        | 2.9  | 11.5  | 8.6   | 16.6 | 3.4  | 8.6  |
| 2018--WACV | [SaCNN](#SaCNN)                       | 2.6  | 13.5  | 10.6  | 12.5 | 3.3  | 8.5  |
| 2018--ECCV | [SANet](#SANet)                       | 2.6  | 13.2  | 9.0   | 13.3 | 3.0  | 8.2  |
| 2018--IJCAI| [DRSAN](#DRSAN)                       | 2.6  | 11.8  | 10.3  | 10.4 | 3.7  | 7.76 |
| 2018--CVPR | [ACSCP](#ACSCP)                       | 2.8  | 14.05 | 9.6   | 8.1  | 2.9  | 7.5  |
| 2019--ICCV | [PGCNet](#PGCNet)                     | 2.5  | 12.7  | **8.4** | 13.7 | 3.2 | 8.1 |
| 2019--CVPR | [TEDnet](#TEDnet)                     | 2.3  | 10.1  | 11.3  | 13.8 | 2.6  | 8.0  |
| 2019--CVPR | [PACNN](#PACNN)                       | 2.3  | 12.5  | 9.1   | 11.2 | 3.8  | 7.8  |
| 2019--CVPR | **[ADCrowdNet](#ADCrowdNet)(AMG-bAttn-DME)** | 1.7   | 14.4  | 11.5 | **7.9** | 3.0 | 7.7 |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-attn-DME)      | 1.6   | 13.2  | 8.7  | 10.6    | 2.6 | 7.3 |
| 2019--CVPR | **[CAN](#CAN)**                       | 2.9  | 12.0  | 10.0  | **7.9** | 4.3 | 7.4  |
| 2019--CVPR | **[CAN](#CAN)(ECAN)**                 | 2.4  | **9.4** | 8.8 | 11.2 | 4.0 | 7.2 |
| 2019--ICCV | **[DSSINet](#DSSINet)**               | **1.57**  | 9.51 | 9.46 | 10.35 | **2.49** | **6.67** |



### UCSD

| Year-Conference/Journal | Method | MAE | MSE |
| --- | --- | --- | --- |
| 2015--CVPR | [Zhang 2015](#Zhang2015)                | 1.60 | 3.31 |
| 2016--ECCV | [Hydra-CNN](#Hydra-CNN)                 | 1.65 |  -   |
| 2016--ECCV | [CNN-Boosting](#CNN-Boosting)           | 1.10 |  -   |
| 2016--CVPR | [MCNN](#MCNN)                           | 1.07 | 1.35 |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)                | 1.73 | 3.52 |
| 2017--CVPR | [Switching CNN](#SCNN)                  | 1.62 | 2.10 |
| 2017--ICCV | [ConvLSTM](#ConvLSTM)                   | 1.30 | 1.79 |
| 2017--ICCV | [Bidirectional   ConvLSTM](#ConvLSTM)   | 1.13 | 1.43 |
| 2018--CVPR | [CSRNet](#CSR)                          | 1.16 | 1.47 |
| 2018--CVPR | [ACSCP](#ACSCP)                         | 1.04 | 1.35 |
| 2018--ECCV | [SANet](#SANet)                         | 1.02 | 1.29 |
| 2018--TIP  | [BSAD](#BSAD)                           | 1.00 | 1.40 |
| 2019--WACV | [SPN](#SPN)                             | 1.03 | 1.32 |
| 2019--ICCV | [SPANet+SANet](#SPANet)                 | 1.00 | 1.28 |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(DME)          | 0.98 | 1.25 |
| 2019--BMVC | [E3D](#E3D)                             | 0.93 | 1.17 |
| 2019--CVPR | [PACNN](#PACNN)                         | 0.89 | 1.18 |
| 2019--TIP  | **[PaDNet](#PaDNet)**                   | **0.85** | **1.06** |

### Mall

| Year-Conference/Journal | Method | MAE | MSE |
| --- | --- | --- | --- |
| 2012--BMVC | [Chen 2012](#Chen2012)                  | 3.15 | 15.7 |
| 2016--ECCV | [CNN-Boosting](#CNN-Boosting)           | 2.01 |  -   |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)                | 2.53 | 11.2 |
| 2017--ICCV | [ConvLSTM](#ConvLSTM)                   | 2.24 | 8.5  |
| 2017--ICCV | [Bidirectional   ConvLSTM](#ConvLSTM)   | 2.10 | 7.6  |
| 2018--CVPR | [DecideNet](#DecideNet)                 | 1.52 | 1.90 |
| 2018--IJCAI| [DRSAN](#DRSAN)                         | 1.72 | 2.1  |
| 2019--BMVC | [E3D](#E3D)                             | 1.64 | 2.13 |
| 2019--WACV | **[SAAN](#SAAN)**                       | **1.28** | **1.68** |
