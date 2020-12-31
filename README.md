# Awesome Crowd Counting

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Misc](#misc)
* [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard)

## Misc

### News
- [2020.04] The JHU-CROWD++ Dataset is released.

### Call for Papers
- [IET Image Processing] Special Issue on: Crowd Understanding and Analysis [[Link](https://digital-library.theiet.org/content/journals/iet-ipr/info/spl-issues;jsessionid=rfnb4mhi25p6.x-iet-live-01)] [[PDF](https://digital-library.theiet.org/files/IET_IPR_CFP_CUA.pdf)]

### Challenge
- [[VisDrone 2020](http://aiskyeye.com/challenge/crowd-counting/)] ~~Crowd counting. ECCV Workshop. Deadline: **2020.07.15**.~~
- [[NWPU-Crowd Counting](https://www.crowdbenchmark.com/nwpucrowd.html)] Crowd counting. Deadline: none.

### Code
- [[C^3 Framework](https://github.com/gjy3035/C-3-Framework)] An open-source PyTorch code for crowd counting, which is released.
- [[CCLabeler](https://github.com/Elin24/cclabeler)] A web tool for labeling pedestrians in an image, which is released.

### Technical blog
- [Chinese Blog] 人群计数论文解读 [[Link](https://zhuanlan.zhihu.com/c_1111215695622352896)]
- [2019.05] [Chinese Blog] C^3 Framework系列之一：一个基于PyTorch的开源人群计数框架 [[Link](https://zhuanlan.zhihu.com/p/65650998)]
- [2019.04] Crowd counting from scratch [[Link](https://github.com/CommissarMa/Crowd_counting_from_scratch)]
- [2017.11] Counting Crowds and Lines with AI [[Link1](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/)] [[Link2](https://count.dimroc.com/)] [[Code](https://github.com/dimroc/count)]

###  GT generation
- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)] [[Fast Python Code](https://github.com/vlad3996/computing-density-maps)] [[Pytorch CUDA Code](https://github.com/gjy3035/NWPU-Crowd-Sample-Code/blob/master/misc/dot_ops.py)]

## Datasets

Please refer to [this page](src/Datasets.md).

## Papers

Considering the increasing number of papers in this field, we roughly summarize some articles and put them into the following categories (they are still listed in this document):

| [[**Top Conference/Journal**](src/Top_Conference-Journal.md)] | [[**Survey**](src/Survey.md)] | [[**Un-/semi-/weakly-/self- Supervised Learning**](src/Un-_Semi-_Weakly-_Self-_supervised_Learning.md)] |
| :---- | :---- | :---- |
| [[**Auxiliary Tasks**](src/Auxiliary_Tasks.md)] | [[**Localization**](src/Localization.md)] | [[**Transfer Learning and Domain Adaptation**](src/Transfer_Learning_and_Domain_Adaptation.md)] |
| [[**Light-weight Models**](src/Light-weight_Model.md)] | [[**Video**](src/Video.md)] | [[**Network Design, Search**](src/Network_Design_and_Search.md)] |
| [[**Perspective Map**](src/Perspective_Map.md)] | [[**Attention**](src/Attention.md)] | Todo |

### arXiv papers
Note that all unpublished arXiv papers are not included in [the leaderboard of performance](#performance).
- Dilated-Scale-Aware Attention ConvNet For Multi-Class Object Counting [[paper](https://arxiv.org/abs/2012.08149)]
- Counting People by Estimating People Flows [[paper](https://arxiv.org/abs/2012.00452)][[code](https://github.com/weizheliu/People-Flows)]
- A Strong Baseline for Crowd Counting and Unsupervised People Localization [[paper](https://arxiv.org/abs/2011.03725)]
- AdaCrowd: Unlabeled Scene Adaptation for Crowd Counting [[paper](https://arxiv.org/abs/2010.12141)]
- Multi-Resolution Fusion and Multi-scale Input Priors Based Crowd Counting [[paper](https://arxiv.org/abs/2010.01664)]
- Completely Self-Supervised Crowd Counting via Distribution Matching [[paper](https://arxiv.org/abs/2009.06420)][[code](https://github.com/val-iisc/css-ccnn)]
- A Study of Human Gaze Behavior During Visual Crowd Counting [[paper](https://arxiv.org/abs/2009.06502)]
- Bayesian Multi Scale Neural Network for Crowd Counting [[paper](https://arxiv.org/abs/2007.14245)][[code](https://github.com/abhinavsagar/bmsnn)]
- A Self-Training Approach for Point-Supervised Object Detection and Counting in Crowds [[paper](https://arxiv.org/abs/2007.12831)]
- DeepCorn: A Semi-Supervised Deep Learning Method for High-Throughput Image-Based Corn Kernel Counting and Yield Estimation  [[paper](https://arxiv.org/abs/2007.10521)]
- Dense Crowds Detection and Counting with a Lightweight Architecture [[paper](https://arxiv.org/abs/2007.06630)]
- Fine-Grained Crowd Counting [[paper](https://arxiv.org/abs/2007.06146)]
- Exploit the potential of Multi-column architecture for Crowd Counting [[paper](https://arxiv.org/abs/2007.05779)][[code](https://github.com/JunhaoCheng/Pyramid_Scale_Network)]
- Recurrent Distillation based Crowd Counting [[paper](https://arxiv.org/abs/2006.07755)]
- Interlayer and Intralayer Scale Aggregation for Scale-invariant Crowd Counting [[paper](https://arxiv.org/abs/2005.11943)]
- Ambient Sound Helps: Audiovisual Crowd Counting in Extreme Conditions [[paper](https://arxiv.org/abs/2005.07097)][[code](https://github.com/qingzwang/AudioVisualCrowdCounting)]
- Neuron Linear Transformation: Modeling the Domain Shift for Crowd Counting [[paper](https://arxiv.org/abs/2004.02133)]

<details>
<summary>Earlier ArXiv Papers</summary>

- Understanding the impact of mistakes on background regions in crowd counting [[paper](https://arxiv.org/abs/2003.13759)]
- CNN-based Density Estimation and Crowd Counting: A Survey [[paper](https://arxiv.org/abs/2003.12783)]
- Drone Based RGBT Vehicle Detection and Counting: A Challenge [[paper](https://arxiv.org/abs/2003.02437)]
- Towards Using Count-level Weak Supervision for Crowd Counting [[paper](https://arxiv.org/abs/2003.00164)]
- PDANet: Pyramid Density-aware Attention Net for Accurate Crowd Counting [[paper](https://arxiv.org/abs/2001.05643)]
- From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting [[paper](https://arxiv.org/abs/2001.01886)](extension of [S-DCNet](#S-DCNet))
- AutoScale: Learning to Scale for Crowd Counting [[paper](https://arxiv.org/abs/1912.09632)](extension of [L2SM](#L2SM))
- Domain-adaptive Crowd Counting via Inter-domain Features Segregation and Gaussian-prior Reconstruction [[paper](https://arxiv.org/abs/1912.03677)]
- Drone-based Joint Density Map Estimation, Localization and Tracking with Space-Time Multi-Scale Attention Network [[paper](https://arxiv.org/abs/1912.01811)][[code](https://github.com/VisDrone)]
- Using Depth for Pixel-Wise Detection of Adversarial Attacks in Crowd Counting [[paper](https://arxiv.org/abs/1911.11484)]
- Estimating People Flows to Better Count them in Crowded Scenes [[paper](https://arxiv.org/abs/1911.10782)]
- Segmentation Guided Attention Network for Crowd Counting via Curriculum Learning [[paper](https://arxiv.org/abs/1911.07990)]
- Deep Density-aware Count Regressor [[paper](https://arxiv.org/abs/1908.03314)][[code](https://github.com/GeorgeChenZJ/deepcount)]
- Fast Video Crowd Counting with a Temporal Aware Network [[paper](https://arxiv.org/abs/1907.02198)]
- Dense Scale Network for Crowd Counting [[paper](https://arxiv.org/abs/1906.09707)][unofficial code: [PyTorch](https://github.com/rongliangzi/Dense-Scale-Network-for-Crowd-Counting)]
- Content-aware Density Map for Crowd Counting and Density Estimation [[paper](https://arxiv.org/abs/1906.07258)]
- Crowd Transformer Network [[paper](https://arxiv.org/abs/1904.02774)]
- W-Net: Reinforced U-Net for Density Map Estimation [[paper](https://arxiv.org/abs/1903.11249)][[code](https://github.com/ZhengPeng7/W-Net-Keras)]
- Improving Dense Crowd Counting Convolutional Neural Networks using Inverse k-Nearest Neighbor Maps and Multiscale Upsampling [[paper](https://arxiv.org/abs/1902.05379)]
- Dual Path Multi-Scale Fusion Networks with Attention for Crowd Counting [[paper](https://arxiv.org/pdf/1902.01115.pdf)]
- Scale-Aware Attention Network for Crowd Counting [[paper](https://arxiv.org/pdf/1901.06026.pdf)]
- Attention to Head Locations for Crowd Counting [[paper](https://arxiv.org/abs/1806.10287)]
- Crowd Counting with Density Adaption Networks [[paper](https://arxiv.org/abs/1806.10040)]
- Improving Object Counting with Heatmap Regulation [[paper](https://arxiv.org/abs/1803.05494)][[code](https://github.com/littleaich/heatmap-regulation)]
- Structured Inhomogeneous Density Map Learning for Crowd Counting [[paper](https://arxiv.org/pdf/1801.06642.pdf)]
- Image Crowd Counting Using Convolutional Neural Network and Markov Random Field [[paper](https://arxiv.org/abs/1706.03686)] [[code](https://github.com/hankong/crowd-counting)]
</details>


### 2020

- <a name="CFANet"></a> **[CFANet]** Coarse- and Fine-grained Attention Network with Background-aware Loss for Crowd Density Map Estimation (**WACV**) [[paper](https://arxiv.org/abs/2011.03721)][[code](https://github.com/rongliangzi/MARUNet)]
- <a name="M-SFANet"></a> **[M-SFANet]** Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting (**ICPR**) [[paper](https://arxiv.org/abs/2003.05586)][[code](https://github.com/Pongpisit-Thanasutives/Variations-of-SFANet-for-Crowd-Counting)]
- <a name="JHU-CROWD"></a> **[JHU-CROWD]** JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method (**T-PAMI**) [[paper](https://arxiv.org/abs/2004.03597)](extension of [CG-DRCN](#CG-DRCN))
- <a name="DM-Count"></a> **[DM-Count]** Distribution Matching for Crowd Counting (**NeurIPS**) [[paper](https://arxiv.org/abs/2009.13077)][[code](https://github.com/cvlab-stonybrook/DM-Count)]
- <a name=""></a> Modeling Noisy Annotations for Crowd Counting (**NeurIPS**)
- <a name="KDMG"></a> **[KDMG]** Kernel-based Density Map Generation for Dense Object Counting (**T-PAMI**) [[paper](https://ieeexplore.ieee.org/document/9189836)][[code](https://github.com/jia-wan/KDMG_Counting)]
- <a name="NWPU"></a> **[NWPU]** NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization (**T-PAMI**) [[paper](https://arxiv.org/abs/2001.03360)][[code](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)]
- <a name="PWCU"></a> **[PWCU]** Pixel-wise Crowd Understanding via Synthetic Data (**IJCV**) [[paper](https://arxiv.org/abs/2007.16032)]
- <a name="SKT"></a> **[SKT]** Efficient Crowd Counting via Structured Knowledge Transfer (**ACM MM(oral)**) [[paper](https://arxiv.org/abs/2003.10120)][[code](https://github.com/HCPLab-SYSU/SKT)]
- <a name="RDBT"></a> **[RDBT]** Towards Unsupervised Crowd Counting via Regression-Detection Bi-knowledge Transfer (**ACM MM**) [[paper](https://arxiv.org/abs/2008.05383)]
- <a name="PeopleFlow"></a> **[PeopleFlow]** Estimating People Flows to Better Count Them in Crowded Scenes (**ECCV**) [[paper](https://arxiv.org/abs/1911.10782)][[code](https://github.com/weizheliu/People-Flows)]
- <a name="AMSNet"></a> **[AMSNet]** NAS-Count: Counting-by-Density with Neural Architecture Search (**ECCV**) [[paper](https://arxiv.org/abs/2003.00217)]
- <a name="AMRNet"></a> **[AMRNet]** Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting (**ECCV**) [[paper](https://arxiv.org/abs/2005.05776)][[code](https://github.com/xiyang1012/Local-Crowd-Counting)]
- <a name="LibraNet"></a> **[LibraNet]** Weighting Counts: Sequential Crowd Counting by Reinforcement Learning (**ECCV**) [[paper](https://arxiv.org/abs/2007.08260)][[code](https://github.com/poppinace/libranet)]
- <a name="GP"></a> **[GP]** Learning to Count in the Crowd from Limited Labeled Data (**ECCV**) [[paper](https://arxiv.org/abs/2007.03195)]
- <a name="IRAST"></a> **[IRAST]** Semi-supervised Crowd Counting via Self-training on Surrogate Tasks (**ECCV**) [[paper](https://arxiv.org/abs/2007.03207)]
- <a name="PSSW"></a> **[PSSW]** Active Crowd Counting with Limited Supervision (**ECCV**) [[paper](https://arxiv.org/abs/2007.06334)]
- <a name="CCLS"></a> **[CCLS]** Weakly-Supervised Crowd Counting Learns from Sorting rather than Locations (**ECCV**) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530001.pdf)]
- A Flow Base Bi-path Network for Cross-scene Video Crowd Understanding in Aerial View (**ECCVW**) [[paper](https://arxiv.org/abs/2009.13723)]
- <a name="ADSCNet"></a> **[ADSCNet]** Adaptive Dilated Network with Self-Correction Supervision for Counting (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Bai_Adaptive_Dilated_Network_With_Self-Correction_Supervision_for_Counting_CVPR_2020_paper.pdf)] 
- <a name="RPNet"></a> **[RPNet]** Reverse Perspective Network for Perspective-Aware Object Counting (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Reverse_Perspective_Network_for_Perspective-Aware_Object_Counting_CVPR_2020_paper.pdf)] [[code](https://github.com/CrowdCounting)]
- <a name="ASNet"></a> **[ASNet]** Attention Scaling for Crowd Counting (**CVPR**) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Attention_Scaling_for_Crowd_Counting_CVPR_2020_paper.pdf)] [[code](https://github.com/laridzhang/ASNet)]
- <a name="LSC-CNN"></a> **[LSC-CNN]** Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection (**T-PAMI**) [[paper](https://arxiv.org/abs/1906.07538)][[code](https://github.com/val-iisc/lsc-cnn)]
- <a name="SRF-Net"></a> **[SRF-Net]** Scale-Aware Rolling Fusion Network for Crowd Counting (**ICME**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102854)]
- <a name="HSRNet"></a> **[HSRNet]** Crowd Counting via Hierarchical Scale Recalibration Network (**ECAI**) [[paper](https://arxiv.org/abs/2003.03545)]
- <a name="SOFA-Net"></a> **[SOFA-Net]** SOFA-Net: Second-Order and First-order Attention Network for Crowd Counting (**BMVC**) [[paper](https://arxiv.org/abs/2008.03723)]
- <a name="CWAN"></a> **[CWAN]** Weakly Supervised Crowd-Wise Attention For Robust Crowd Counting (**ICASSP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054258)]
- <a name="AGRD"></a> **[AGRD]** Attention Guided Region Division for Crowd Counting (**ICASSP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053761)]
- <a name="BBA-NET"></a> **[BBA-NET]** BBA-NET: A Bi-Branch Attention Network For Crowd Counting (**ICASSP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053955)]
- <a name="SMANet"></a> **[SMANet]** Stochastic Multi-Scale Aggregation Network for Crowd Counting (**ICASSP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054238)]
- <a name="Stacked-Pool"></a> **[Stacked-Pool]** Stacked Pooling For Boosting Scale Invariance Of Crowd Counting (**ICASSP**) [[paper](https://siyuhuang.github.io/papers/ICASSP-2020-STACKED%20POOLING%20FOR%20BOOSTING%20SCALE%20INVARIANCE%20OF%20CROWD%20COUNTING.pdf)] [[arxiv](https://arxiv.org/abs/1808.07456)] [[code](http://github.com/siyuhuang/crowdcount-stackpool)]
- <a name="MSPNET"></a> **[MSPNET]** Multi-supervised Parallel Network for Crowd Counting (**ICASSP**) [[paper](https://crabwq.github.io/pdf/2020%20MSPNET%20Multi-supervised%20Parallel%20Network%20for%20Crowd%20Counting.pdf)]
- <a name="ASPDNet"></a> **[ASPDNet]** Counting dense objects in remote sensing images (**ICASSP**) [[paper](https://arxiv.org/abs/2002.05928)]
- <a name="FSC"></a> **[FSC]** Focus on Semantic Consistency for Cross-domain Crowd Understanding (**ICASSP**) [[paper](https://arxiv.org/abs/2002.08623)]
- <a name="C-CNN"></a> **[C-CNN]** A Real-Time Deep Network for Crowd Counting (**ICASSP**) [[arxiv](https://arxiv.org/abs/2002.06515)][[ieee](https://ieeexplore.ieee.org/abstract/document/9053780/)]
- <a name="HyGnn"></a> **[HyGnn]** Hybrid  Graph  Neural  Networks  for  Crowd  Counting (**AAAI**) [[paper](https://arxiv.org/abs/2002.00092)]
- <a name="DUBNet"></a> **[DUBNet]** Crowd Counting with Decomposed Uncertainty (**AAAI**) [[paper](https://arxiv.org/abs/1903.07427)]
- <a name="SDANet"></a> **[SDANet]** Shallow  Feature  based  Dense  Attention  Network  for  Crowd  Counting (**AAAI**) [[paper](http://wrap.warwick.ac.uk/130173/1/WRAP-shallow-feature-dense-attention-crowd-counting-Han-2019.pdf)]
- <a name="3DCC"></a> **[3DCC]** 3D Crowd Counting via Multi-View Fusion with 3D Gaussian Kernels (**AAAI**) [[paper](https://arxiv.org/abs/2003.08162)][[Project](http://visal.cs.cityu.edu.hk/research/aaai20-3d-counting/)]
- <a name="FFSA"></a> **[FSSA]** Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning (**WACV**) [[paper](https://arxiv.org/abs/2002.00264)][[code](https://github.com/maheshkkumar/fscc)] 
- <a name="CC-Mod"></a> **[CC-Mod]** Plug-and-Play Rescaling Based Crowd Counting in Static Images (**WACV**) [[paper](https://arxiv.org/abs/2001.01786)]
- <a name="CTN"></a> **[CTN]** Uncertainty Estimation and Sample Selection for Crowd Counting (**ACCV**) [[paper](https://arxiv.org/abs/2009.14411)]
- <a name="CLPNet"></a> **[CLPNet]** Cross-Level Parallel Network for Crowd Counting (**TII**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798674)]
- <a name="HA-CCN"></a> **[HA-CCN]** HA-CCN: Hierarchical Attention-based Crowd Counting Network (**TIP**) [[paper](https://arxiv.org/abs/1907.10255)]
- <a name="PaDNet"></a> **[PaDNet]** PaDNet: Pan-Density Crowd Counting (**TIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8897143)]
- <a name="CRNet"></a> **[CRNet]** Crowd Counting via Cross-stage Refinement Networks (**TIP**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9096602)][[code](https://github.com/lytgftyf/Crowd-Counting-via-Cross-stage-Refinement-Networks)] 
- Feature-aware Adaptation and Density Alignment for Crowd Counting in Video Surveillance (**TCYB**) [[paper](https://arxiv.org/abs/1912.03672)]
- <a name="MS-GAN"></a> **[MS-GAN]** Adversarial Learning for Multiscale Crowd Counting Under Complex Scenes (**TCYB**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8949751)]
- Density-aware Curriculum Learning for Crowd Counting (**TCYB**) [[code](https://github.com/Elin24/DCL-CrowdCounting)]
- <a name="ZoomCount"></a> **[ZoomCount]** ZoomCount: A Zooming Mechanism for Crowd Counting in Static Images (**T-CSVT**) [[paper](https://arxiv.org/abs/2002.12256)]
- <a name="DensityCNN"></a> **[DensityCNN]** Density-Aware Multi-Task Learning for Crowd Counting (**TMM**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9037113)]
- <a name="DENet"></a> **[DENet]** DENet: A Universal Network for Counting Crowd with Varying Densities and Scales (**TMM**) [[paper](https://arxiv.org/abs/1904.08056)][[code](https://github.com/liuleiBUAA/DENet)]
- <a name="FMLF"></a> **[FMLF]** Crowd Density Estimation Using Fusion of Multi-Layer Features (**TITS**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9063540)]
- <a name="MLSTN"></a> **[MLSTN]** Multi-level feature fusion based Locality-Constrained Spatial Transformer network for video crowd counting (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220301454)](extension of [LSTN](#LSTN))
- <a name="SRN+PS"></a> **[SRN+PS]** Scale-Recursive Network with point supervision for crowd scene analysis (**Neurocomputing**) [[paper](https://sciencedirect.xilesou.top/science/article/abs/pii/S0925231219317795)]
- <a name="ASDF"></a> **[ASDF]** Counting crowds with varying densities via adaptive scenario discovery framework (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220302356)](extension of [ASD](#ASD))
- <a name="CAT-CNN"></a> **[CAT-CNN]** Crowd counting with crowd attention convolutional neural network (**Neurocomputing**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219316662)]
- <a name="RRP"></a> **[RRP]** Relevant Region Prediction for Crowd Counting (**Neurocomputing**) [[paper](https://arxiv.org/abs/2005.09816)]
- <a name="SCAN"></a> **[SCAN]** Crowd Counting via Scale-Communicative Aggregation Networks (**Neurocomputing**) [[paper]()](extension of [MVSAN](#MVSAN))
- <a name="MobileCount"></a> **[MobileCount]** MobileCount: An Efficient Encoder-Decoder Framework for Real-Time Crowd Counting (**Neurocomputing**) [[conference paper](https://link.springer.com/chapter/10.1007/978-3-030-31723-2_50)] [[journal paper](https://www.sciencedirect.com/science/article/pii/S0925231220308912)] [[code](https://github.com/SelinaFelton/MobileCount)]

### 2019

- <a name="D-ConvNet"></a> **[D-ConvNet]** Nonlinear Regression via Deep Negative Correlation Learning (**T-PAMI**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8850209)](extension of [D-ConvNet](#D-ConvNet))[[Project](https://mmcheng.net/dncl/)]
- <a name=""></a>Generalizing semi-supervised generative adversarial networks to regression using feature contrasting (**CVIU**)[[paper](https://arxiv.org/abs/1811.11269)]
- <a name="CG-DRCN"></a> **[CG-DRCN]** Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and Benchmark Method (**ICCV**)[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sindagi_Pushing_the_Frontiers_of_Unconstrained_Crowd_Counting_New_Dataset_and_ICCV_2019_paper.pdf)]
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

- <a name="Lempitsky2010"></a> **[Lempitsky 2010]** Learning To Count Objects in Images (**NeurIPS**) [[paper](http://papers.nips.cc/paper/4043-learning-to-count-objects-in-images)]

### 2008

- <a name="Chan2008"></a> **[Chan 2008]** Privacy preserving crowd monitoring: Counting people without people models or tracking (**CVPR**) [[paper](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr08-peoplecnt.pdf)]



## Leaderboard
The section is being continually updated. Note that some values have superscript, which indicates their source. 

### NWPU

| Year-Conference/Journal | Methods                 | Val-MAE | Val-MSE | Test-MAE | Test-MSE | Test-NAE | Backbone |
| ---- | ------------------------------------------ | ------- | ------- | -------- | -------- | -------- | -------- |
| 2016--CVPR | [MCNN](#MCNN)                        | 218.5  | 700.6 | 232.5 | 714.6 | 1.063 | FS |
| 2018--CVPR | [CSRNet](#CSR)                       | 104.8  | 433.4 | 121.3 | 387.8 | 0.604 | VGG-16 |
| 2019--CSVT | [PCC-Net](#PCC-Net)                  | 100.7  | 573.1 | 112.3 | 457.0 | 0.251 | VGG-16 |
| 2019--CVPR | **[CAN](#CAN)**                      | 93.5   | 489.9 | 106.3 | **386.5** | 0.295 | VGG-16 |
| 2019--NC   | [SCAR](#SCAR)                        | 81.5   | 397.9 | 110.0 | 495.3 | 0.288 | VGG-16 |
| 2019--ICCV | [BL](#BL)                            | 93.6   | 470.3 | 105.4 | 454.2 | 0.203 | VGG-19 |
| 2019--CVPR | [SFCN](#CCWld)                       | 95.4   | 608.3 | 105.4 | 424.1 | 0.254 | ResNet-101 |
| 2020--NeurIPS |**[DM-Count](#DM-Count)**          | **70.5** | **357.6** | **88.4** | 388.6 | 0.169 | VGG-19 |

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
| 2020--AAAI | [DUBNet](#DUBNet)                            | 64.6  | 106.8 | -  | -  | -  | -           |
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
| 2020--ICPR | [M-SFANet+M-SegNet](#M-SFANet)               | 57.55 | 94.48 | -  | -  | -  | -           |
| 2019--ICCV |**[PGCNet](#PGCNet)**                         | 57.0 | **86.0** | -  | -  | -  | -         |
| 2020--CVPR |**[ADSCNet](#ADSCNet)**                       | **55.4** | 97.7 | -  | -  | -  | -         |

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
| 2020--AAAI | [DUBNet](#DUBNet)                             | 7.7   | 12.5  |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-DME)            | 7.6   | 13.9  |
| 2019--CVPR | [SFCN](#CCWld)                                | 7.6   | 13.0  |
| 2019--CVPR | [PACNN](#PACNN)                               | 8.9   | 13.5  |
| 2019--CVPR | [PACNN+CSRNet](#PACNN)                        | 7.6   | 11.8  |
| 2019--ICCV | [BL](#BL)                                     | 7.7   | 12.7  |
| 2019--ICCV | [CFF](#CFF)                                   | 7.2   | 12.2  |
| 2019--ICCV | [SPN+L2SM](#L2SM)                             | 7.2   | 11.1  |
| 2019--ICCV | [DSSINet](#DSSINet)                           | 6.85  | 10.34 |
| 2019--ICCV | [S-DCNet](#S-DCNet)                           | 6.7   | 10.7  |
| 2019--ICCV | **[SPANet+SANet](#SPANet)**                   | 6.5   | **9.9** |
| 2020--CVPR | [ADSCNet](#ADSCNet)                           | 6.4   | 11.3 |
| 2020--ICPR | **[M-SFANet+M-SegNet](#M-SFANet)**            | **6.32** | 10.06 |

### UCF-QNRF

| Year-Conference/Journal | Method | C-MAE | C-NAE | C-MSE | DM-MAE | DM-MSE | DM-HI | L- Av. Precision	| L-Av. Recall | L-AUC |
| --- | --- | --- | --- |--- | --- | --- |--- | --- | --- | ---|
| 2013--CVPR | [Idrees 2013](#Idrees2013)<sup>[CL](#CL)</sup>| 315 | 0.63 | 508 | - | - | - | - | - | - |
| 2016--CVPR | [MCNN](#MCNN)<sup>[CL](#CL)</sup> | 277 | 0.55 | 426 |0.006670| 0.0223 | 0.5354 |59.93% | 63.50% | 0.591|
| 2017--AVSS | [CMTL](#CMTL)<sup>[CL](#CL)</sup>            | 252 | 0.54 | 514 | 0.005932 | 0.0244 | 0.5024 | - | - | - |
| 2017--CVPR | [Switching CNN](#SCNN)<sup>[CL](#CL)</sup>   | 228 | 0.44 | 445 | 0.005673 | 0.0263 | 0.5301 | - | - | - |
| 2018--ECCV | [CL](#CL)     | 132 | 0.26 | 191 | 0.00044| 0.0017 | 0.9131 | 75.8% | 59.75%	| 0.714|
| 2019--TIP  | [HA-CCN](#HA-CCN)   | 118.1 | - | 180.4 | - | - | - | - | - | - |
| 2019--CVPR | [TEDnet](#TEDnet)   | 113   | - | 188   | - | - | - | - | - | - |
| 2019--ICCV | [RANet](#RANet)     | 111   | - | 190   | - | - | - | - | - | - |
| 2019--CVPR | [CAN](#CAN)         | 107   | - | 183   | - | - | - | - | - | - |
| 2020--AAAI | [DUBNet](#DUBNet)   | 105.6 | - | 180.5 | - | - | - | - | - | - |
| 2019--ICCV | [SPN+L2SM](#L2SM)   | 104.7 | - | 173.6 | - | - | - | - | - | - |
| 2019--ICCV | [S-DCNet](#S-DCNet) | 104.4 | - | 176.1 | - | - | - | - | - | - |
| 2019--CVPR | [SFCN](#CCWld)      | 102.0 | - | 171.4 | - | - | - | - | - | - |
| 2019--ICCV | [DSSINet](#DSSINet) | 99.1  | - | 159.2 | - | - | - | - | - | - |
| 2019--ICCV | [MBTTBF-SCFB](#MBTTBF)      | 97.5 | - | 165.2 | - | - | - | - | - | - |
| 2019--TIP  | [PaDNet](#PaDNet)           | 96.5 | - | 170.2 | - | - | - | - | - | - |
| 2019--ICCV | [BL](#BL)                   | 88.7 | - | 154.8 | - | - | - | - | - | - |
| 2020--ICPR | [M-SFANet](#M-SFANet)       | 85.6 | - | 151.23 | - | - | - | - | - | - |
| 2020--CVPR |**[ADSCNet](#ADSCNet)**      | **71.3** | - | **132.5** | - | - | - | - | - | - |


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
| 2020--AAAI | [DUBNet](#DUBNet)                            | 243.8 | 329.3  |
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
| 2019--TIP  | [PaDNet](#PaDNet)                            | 185.8 | 278.3 |
| 2020--ICPR | **[M-SFANet](#M-SFANet)**                    | **162.33** | 276.76 |

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
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-bAttn-DME)     | 1.7   | 14.4  | 11.5 | 7.9  | 3.0  | 7.7 |
| 2019--CVPR | [ADCrowdNet](#ADCrowdNet)(AMG-attn-DME)      | 1.6   | 13.2  | 8.7  | 10.6    | 2.6 | 7.3 |
| 2019--CVPR | [CAN](#CAN)                           | 2.9  | 12.0  | 10.0  | 7.9  | 4.3  | 7.4  |
| 2019--CVPR | **[CAN](#CAN)(ECAN)**                 | 2.4  | **9.4** | 8.8 | 11.2 | 4.0 | 7.2 |
| 2019--ICCV | **[DSSINet](#DSSINet)**               | **1.57**  | 9.51 | 9.46 | 10.35 | **2.49** | **6.67** |
| 2020--ICPR | **[M-SFANet](#M-SFANet)**             | 1.88 | 13.24 | 10.07 | **7.5** | 3.87 | 7.32 |



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
