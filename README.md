# Awesome Crowd Counting

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Tools](#tools)
* [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard)

##  Tools

- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)] [[Fast Python Code](https://github.com/vlad3996/computing-density-maps)]


## Datasets

- UCF-QNRF Dataset [[Link](http://crcv.ucf.edu/data/ucf-qnrf/)]
- ShanghaiTech Dataset [Link: [Dropbox ](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)/ [BaiduNetdisk](https://pan.baidu.com/s/1nuAYslz)]
- WorldExpo'10 Dataset [[Link](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)]
- UCF CC 50 Dataset [[Link](http://crcv.ucf.edu/data/ucf-cc-50/)]
- Mall Dataset  [[Link](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)]
- UCSD Dataset [[Link](http://www.svcl.ucsd.edu/projects/peoplecnt/)]
- SmartCity Dataset [Link: [GoogleDrive ](https://drive.google.com/file/d/1xqflSQv9dZ0A93_lP34pSIfcpheT2Fi8/view?usp=sharing)/ [BaiduNetdisk](https://pan.baidu.com/s/1pMuGyNp)]
- AHU-Crowd Dataset [[Link](http://cs-chan.com/downloads_crowd_dataset.html)] 

## Papers

### arXiv papers
This section only includes the last ten papers since 2018 in [arXiv.org](arXiv.org). Previous papers will be hidden using  ```<!--...-->```. If you want to view them, please open the [raw file](https://raw.githubusercontent.com/gjy3035/Awesome-Crowd-Counting/master/README.md) to read the source code. Note that all unpublished arXiv papers are not included into [the leaderboard of performance](#performance).

- <a name="PaDNet"></a> PaDNet: Pan-Density Crowd Counting [[paper]( https://arxiv.org/abs/1811.02805 )]
- Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance [[paper](https://arxiv.org/abs/1808.07456)][[code](http://github.com/siyuhuang/crowdcount-stackpool)]
- In Defense of Single-column Networks for Crowd Counting [[paper](https://arxiv.org/abs/1808.06133)]
- Perspective-Aware CNN For Crowd Counting [[paper](https://arxiv.org/abs/1807.01989)]
- Attention to Head Locations for Crowd Counting [[paper](https://arxiv.org/abs/1806.10287)]
- Crowd Counting with Density Adaption Networks [[paper](https://arxiv.org/abs/1806.10040)]
- Geometric and Physical Constraints for Head Plane Crowd Density Estimation in Videos [[paper](https://arxiv.org/abs/1803.08805)]
- Improving Object Counting with Heatmap Regulation [[paper](https://arxiv.org/abs/1803.05494)][[code](https://github.com/littleaich/heatmap-regulation)]
- Depth Information Guided Crowd Counting for Complex Crowd Scenes [[paper](https://arxiv.org/abs/1803.02256)]
- Structured Inhomogeneous Density Map Learning for Crowd Counting [[paper](https://arxiv.org/pdf/1801.06642.pdf)]
  <!-- - Image Crowd Counting Using Convolutional Neural Network and Markov Random Field  [[paper](https://arxiv.org/abs/1706.03686)] [[code](https://github.com/hankong/crowd-counting)] -->



### 2018
- <a name="AFP"></a>**[AFP]** Crowd Counting by Adaptively Fusing Predictions from an Image Pyramid (**BMVC2018**) [[paper](https://arxiv.org/abs/1805.06115)]
- <a name="DRSAN"></a>**[DRSAN]** Crowd Counting using Deep Recurrent Spatial-Aware Network (**IJCAI2018**) [[paper](https://arxiv.org/abs/1807.00601)]
- <a name="TDF-CNN"></a>**[TDF-CNN]** Top-Down Feedback for Crowd Counting Convolutional Neural Network (**AAAI2018**) [[paper](https://arxiv.org/abs/1807.08881)]
- <a name="SANet"></a> **[SANet]** Scale Aggregation Network for Accurate and Efficient Crowd Counting (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.pdf)]
- <a name="ic-CNN"></a> **[ic-CNN]** Iterative Crowd Counting (**ECCV2018**) [[paper](https://arxiv.org/abs/1807.09959)]
- <a name="CL"></a> **[CL]** Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds (**ECCV2018**) [[paper](https://arxiv.org/abs/1808.01050)]
- <a name="D-ConvNet"></a> **[D-ConvNet]** Crowd Counting with Deep Negative Correlation Learning (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf)] [[code](https://github.com/shizenglin/Deep-NCL)]
- <a name="IG-CNN"></a> **[IG-CNN]** Divide and Grow: Capturing Huge Diversity in Crowd Images with
Incrementally Growing CNN (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sam_Divide_and_Grow_CVPR_2018_paper.pdf)]
- <a name="BSAD"></a> **[BSAD]** Body Structure Aware Deep Crowd Counting (**TIP2018**) [[paper](http://mac.xmu.edu.cn/rrji/papers/IP%202018-Body.pdf)] 
- <a name="CSR"></a> **[CSR]**  CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (**CVPR2018**) [[paper](https://arxiv.org/abs/1802.10062)] [[code](https://github.com/leeyeehoo/CSRNet)]
- <a name="L2R"></a>  **[L2R]** Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (**CVPR2018**) [[paper](https://arxiv.org/abs/1803.03095)] [[code](https://github.com/xialeiliu/CrowdCountingCVPR18)] 
- <a name="ACSCP"></a> **[ACSCP]**  Crowd Counting via Adversarial Cross-Scale Consistency Pursuit  (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf)]
- <a name="DecideNet"></a> **[DecideNet]** DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density (**CVPR2018**) [[paper](https://arxiv.org/abs/1712.06679)]
- <a name="DR-ResNet"></a> **[DR-ResNet]** A Deeply-Recursive Convolutional Network for Crowd Counting (**ICASSP2018**) [[paper](https://arxiv.org/abs/1805.05633)] 
- <a name="SaCNN"></a> **[SaCNN]** Crowd counting via scale-adaptive convolutional neural network (**WACV2018**) [[paper](https://arxiv.org/abs/1711.04433)] [[code](https://github.com/miao0913/SaCNN-CrowdCounting-Tencent_Youtu)]
- <a name="GAN-MTR"></a> **[GAN-MTR]** Crowd Counting With Minimal Data Using Generative Adversarial Networks For Multiple Target Regression (**WACV2018**) [[paper(http://visionlab.engr.ccny.cuny.edu/ccvcl/assets/publications/155/paper/crowd_gans_wacv_paper_final.pdf)] 

### 2017
- <a name="CP-CNN"></a> **[CP-CNN]** Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (**ICCV2017**) [[paper](https://arxiv.org/abs/1708.00953)]
- <a name="ConvLSTM"></a> **[ConvLSTM]** Spatiotemporal Modeling for Crowd Counting in Videos (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xiong_Spatiotemporal_Modeling_for_ICCV_2017_paper.pdf)]
- <a name="CMTL"></a> **[CMTL]** CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting (**AVSS2017**) [[paper](https://arxiv.org/abs/1707.09605)] [[code](https://github.com/svishwa/crowdcount-cascaded-mtl)]
- <a name="ResnetCrowd"></a> **[ResnetCrowd]** ResnetCrowd: A Residual Deep Learning Architecture for Crowd Counting, Violent Behaviour Detection and Crowd Density Level Classification (**AVSS2017**) [[paper](https://arxiv.org/abs/1705.10698)]
- <a name="SCNN"></a> **[Switching CNN]** Switching Convolutional Neural Network for Crowd Counting (**CVPR2017**) [[paper](https://arxiv.org/abs/1708.00199)] [[code](https://github.com/val-iisc/crowd-counting-scnn)]
- A **Survey** of Recent Advances in CNN-based Single Image Crowd Counting and Density
Estimation (**PR Letters**) [[paper](https://arxiv.org/abs/1707.01202)]
- <a name="MSCNN"></a> **[MSCNN]** Multi-scale Convolution Neural Networks for Crowd Counting (**ICIP2017**) [[paper](https://arxiv.org/abs/1702.02359)] [[code](https://github.com/Ling-Bao/mscnn)]
- <a name="FCNCC"></a> **[FCNCC]** Fully Convolutional Crowd Counting On Highly Congested Scenes (**VISAPP2017**) [[paper](https://arxiv.org/abs/1612.00220)] 

### 2016 

- <a name="Hydra-CNN"></a> **[Hydra-CNN]** Towards perspective-free object counting with deep learning  (**ECCV2016**) [[paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf)] [[code](https://github.com/gramuah/ccnn)]
- <a name="CrowdNet"></a> **[CrowdNet]** CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (**ACMMM2016**) [[paper](https://arxiv.org/abs/1608.06197)] [[code](https://github.com/davideverona/deep-crowd-counting_crowdnet)]
- <a name="MCNN"></a> **[MCNN]** Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (**CVPR2016**) [[paper](https://pdfs.semanticscholar.org/7ca4/bcfb186958bafb1bb9512c40a9c54721c9fc.pdf)] [unofficial code: [TensorFlow](https://github.com/aditya-vora/crowd_counting_tensorflow) [PyTorch](https://github.com/svishwa/crowdcount-mcnn)]

- <a name="Shang2016"></a> **[Shang 2016]** End-to-end crowd counting via joint learning local and global count (**ICIP2016**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7532551)] 


### 2015

- <a name="COUNTForest"></a> **[COUNT Forest]** COUNT Forest: CO-voting Uncertain Number of Targets using Random Forest
for Crowd Density Estimation (**ICCV2015**) [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Pham_COUNT_Forest_CO-Voting_ICCV_2015_paper.pdf)]
- <a name="Zhang2015"></a> **[Zhang 2015]** Cross-scene Crowd Counting via Deep Convolutional Neural Networks (**CVPR2015**) [[paper](https://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf)] [[code](https://github.com/wk910930/crowd_density_segmentation)]

### 2013

- <a name="Idrees2013"></a> **[Idrees 2013]** Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images (**CVPR2013**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Idrees_Multi-source_Multi-scale_Counting_2013_CVPR_paper.pdf)]
- <a name="Ma2013"></a> **[Ma 2013]** Crossing the Line: Crowd Counting by Integer Programming with Local Features (**CVPR2013**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Ma_Crossing_the_Line_2013_CVPR_paper.pdf)]

### 2012

- <a name="Chen2013"></a> **[Chen 2013]** Feature mining for localised crowd counting (**BMVC2012**) [[paper](https://pdfs.semanticscholar.org/c5ec/65e36bccf8a64050d38598511f0352653d6f.pdf)]

### 2008
- <a name="Chan2008"></a> **[Chan 2008]** Privacy preserving crowd monitoring: Counting people without people models or tracking (**CVPR 2008**) [[paper](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr08-peoplecnt.pdf)]



## Leaderboard
The section is being continually updated. Note that some values have superscript, which indicates their source. 


### ShanghaiTech Part A

| Year-Conference/Journal | Methods                              | MAE   | MSE   | PSNR  | SSIM | Params | Pre-trained   Model |
| ---- | ------------------------------------ | ----- | ----- | ----- | ---- | ------ | ------------------- |
| 2016--CVPR | [MCNN](#MCNN)                                 | 110.2 | 173.2 | 21.4<sup>[CSR](#CSR)</sup> | 0.52<sup>[CSR](#CSR)</sup> | 0.13M<sup>[SANet](#SANet)</sup>  | None                 |
| 2017--ICIP | [MSCNN](#MSCNN)                           | 83.8  | 127.4 | -     | -    | -      | -                   |
| 2017--AVSS | [CMTL](#CMTL)                                 | 101.3 | 152.4 | -     | -    | -      | None                |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 90.4  | 135   | -     | -    | -      |VGG-16                 |
| 2017--ICCV | [CP-CNN](#CP-CNN)                              | 73.6  | 106.4 | -     | -    | -      | -                   |
| 2018-WACV | [SaCNN](#SaCNN)                                | 86.8  | 139.2 | -     | -    | -      | -                   |
|  2018--CVPR | [ACSCP](#ACSCP)              | 75.7  | 102.7 | -     | -    | 5.1M     | None                 |
|  2018--CVPR | [CSRNet](#CSR)                 | 68.2  | 115   | 23.79 | 0.76 | 16.26M<sup>[SANet](#SANet)</sup>   |VGG-16               |
|  2018--CVPR | [IG-CNN](#IG-CNN)                               | 72.5  | 118.2 | -     | -    | -      | -                   |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                        | 73.5  | 112.3 | -     | -    | -      | -                   |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example) | 72    | 106.6 | -     | -    | -      | VGG-16                 |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)          | 73.6  | 112   | -     | -    | -      |VGG-16               |
| 2018--IJCAI | **[DRSAN](#DRSAN)**                              | 69.3  | **96.4**  | -     | -    | -      | -                   |
| 2018--ECCV | [ic-CNN](#ic-CNN) (one stage)                   | 69.8  | 117.3 | -     | -    | -      | -                   |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)                  | 68.5  | 116.2 | -     | -    | -      | -                   |
| 2018--ECCV | **[SANet](#SANet)**                                | **67.0**    | 104.5 | -     | -    | 0.91M     | None               |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                              | 97.5  | 145.1 | -     | -    | -      | -                   |


### ShanghaiTech Part B


| Year-Conference/Journal | Methods                              | MAE   | MSE   | 
| ---- | ---------------- | ----- | ---- |
| 2016--CVPR | [MCNN](#MCNN)                                 |  26.4 | 41.3 |
| 2017--ICIP | [MSCNN](#MSCNN)                           | 17.7  | 30.2  |
| 2017--AVSS | [CMTL](#CMTL)                                 | 20    | 31.1  |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 21.6  | 33.4  |
| 2017--ICCV | [CP-CNN](#CP-CNN)                              |  20.1  | 30.1  |
| 2018-TIP | [BSAD](#BSAD)                                  | 20.2  | 35.6  |
| 2018-WACV | [SaCNN](#SaCNN)                                | 16.2  | 25.8  |
|  2018--CVPR | [ACSCP](#ACSCP)              | 17.2  | 27.4  |
|  2018--CVPR | [CSRNet](#CSR)                 |10.6  | 16    |
|  2018--CVPR | [IG-CNN](#IG-CNN)                               | 13.6  | 21.1  |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                        | 18.7  | 26    |
| 2018--CVPR | [DecideNet](#DecideNet)                            | 21.53 | 31.98 |
| 2018--CVPR | [DecideNet + R3](#DecideNet)                       | 20.75 | 29.42 |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example) | 14.4  | 23.8  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)          | 13.7  | 21.4  |
| 2018--IJCAI | [DRSAN](#DRSAN)                            | 11.1  | 18.2  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (one stage)                   | 10.4  | 16.7  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)                  | 10.7  | 16    |
| 2018--ECCV | **[SANet](#SANet)**                                |  **8.4**   | **13.6**  |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                              | 20.7  | 32.8  |

### UCF-QNRF

| Year-Conference/Journal | Method | C-MAE | C-NAE | C-MSE | DM-MAE | DM-MSE | DM-HI |L- Av. Precision	|L-Av. Recall |	L-AUC
| --- | --- | --- | --- |--- | --- | --- |--- | --- | --- | ---|
| 2013--CVPR | [Idrees 2013](#Idrees2013)<sup>[CL](#CL)</sup>| 315 | 0.63 | 508|  - | - | - | - | - | - |
| 2016--CVPR | [MCNN](#MCNN)<sup>[CL](#CL)</sup> | 277 | 0.55 |  |0.006670| 0.0223 | 0.5354 |59.93% | 63.50% | 0.591|
| 2017--AVSS | [CMTL](#CMTL)<sup>[CL](#CL)</sup>                                 | 252 | 0.54 | 514	| 0.005932 | 0.0244 | 0.5024 | - | - | - |
| 2017--CVPR | [Switching CNN](#SCNN)<sup>[CL](#CL)</sup>                       | 228 | 0.44 | 445 | 0.005673 | 0.0263 | 0.5301 | - | - | - |
| 2018--ECCV | **[CL](#CL)** | **132** | 0.26 | **191** | 0.00044| 0.0017 | 0.9131 | 75.8% | 59.75%	| 0.714|


### UCF_CC_50
| Year-Conference/Journal | Methods                              | MAE   | MSE   | 
| ---- | ---------------- | ----- | ---- |
| 2013--CVPR | [Idrees 2013](#Idrees2013)| 468   | 590.3  |
| 2015--CVPR | [Zhang 2015](#Zhang2015) | 467   | 498.5  |
| 2016--CVPR | [MCNN](#MCNN)                                 |  377.6 | 509.1  |
| 2016--ACM MM | [CrowdNet](#CrowdNet)                    | 452.5 | -      |
| 2016--ECCV | [Hydra-CNN](#Hydra-CNN)                            | 333.7 | 425.2  |
| 2016--ICIP | [Shang 2016](#Shang2016)         | 270.3 | -      |
| 2017--ICIP | [MSCNN](#MSCNN)                           | 363.7 | 468.4  |
| 2017--AVSS | [CMTL](#CMTL)                                 | 322.8 | 397.9  |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 318.1 | 439.2  |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)                          | 284.5 | 297.1  |
| 2017--ICCV | [CP-CNN](#CP-CNN)                              |  295.8 | 320.9  |
| 2018-TIP | [BSAD](#BSAD)                                  | 409.5 | 563.7  |
| 2018-WACV | [SaCNN](#SaCNN)                                | 314.9 | 424.8  |
|  2018--CVPR | [ACSCP](#ACSCP)              | 291   | 404.6  |
|  2018--CVPR | [CSRNet](#CSR)                 |266.1 | 397.5  |
|  2018--CVPR | [IG-CNN](#IG-CNN)                               | 291.4 | 349.4  |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                        |288.4 | 404.7  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Query-by-example) | 291.5 | 397.6  |
| 2018--CVPR | [L2R](#L2R) (Multi-task,   Keyword)          | 279.6 | 388.9  |
| 2018--IJCAI | **[DRSAN](#DRSAN)**                            | **219.2** | **250.2**  |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)                  | 260.9 | 365.5  |
| 2018--ECCV | [SANet](#SANet)                              | 258.4 | 334.9  |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                              | 354.7 | 491.4  |

### WorldExpo'10
| Year-Conference/Journal | Method | S1 | S2 | S3 | S4 | S5 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015--CVPR | [Zhang 2015](#Zhang2015) |  9.8  | 14.1  | 14.3  | 22.2 | 3.7  | 12.9 |
| 2016--CVPR | [MCNN](#MCNN)                                 |  3.4  | 20.6  | 12.9  | 13   | 8.1  | 11.6 |
| 2017--ICIP | [MSCNN](#MSCNN)                           | 7.8  | 15.4  | 14.9  | 11.8 | 5.8  | 11.7 |
| 2017--CVPR | [Switching CNN](#SCNN)                       | 4.4  | 15.7  | 10    | 11   | 5.9  | 9.4  |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)                       | 8.6  | 16.9  | 14.6  | 15.4 | 4    | 11.9 |
| 2017--ICCV | [ConvLSTM](#ConvLSTM)                         | 7.1  | 15.2  | 15.2  | 13.9 | 3.5  | 10.9 |
| 2017--ICCV | [Bidirectional   ConvLSTM](#ConvLSTM)     | 6.8  | 14.5  | 14.9  | 13.5 | 3.1  | 10.6 |
| 2017--ICCV | [CP-CNN](#CP-CNN)                              |  2.9  | 14.7  | 10.5  | 10.4 | 5.8  | 8.86 |
| 2018-TIP | [BSAD](#BSAD)                                  | 4.1  | 21.7  | 11.9  | 11   | 3.5  | 10.5 |
| 2018-WACV | [SaCNN](#SaCNN)                                | 2.6  | 13.5  | 10.6  | 12.5 | 3.3  | 8.5  |
|  2018--CVPR | **[ACSCP](#ACSCP)**              | 2.8  | 14.05 | 9.6   | **8.1**  | 2.9  | **7.5**  |
|  2018--CVPR | [CSRNet](#CSR)                 |2.9  | **11.5**  | **8.6**   | 16.6 | 3.4  | 8.6  |
|  2018--CVPR | [IG-CNN](#IG-CNN)                               | 2.6  | 16.1  | 10.15 | 20.2 | 7.6  | 11.3 |
| 2018--CVPR | [D-ConvNet-v1](#D-ConvNet)                        | **1.9**  | 12.1  | 20.7  | 8.3  | **2.6**  | 9.1  |
| 2018--CVPR | [DecideNet](#DecideNet)                            | 2    | 13.14 | 8.9   | 17.4 | 4.75 | 9.23 |
| 2018--IJCAI | [DRSAN](#DRSAN)                            | 2.6  | 11.8  | 10.3  | 10.4 | 3.7  | 7.76 |
| 2018--ECCV | [ic-CNN](#ic-CNN) (two stages)                  | 17   | 12.3  | 9.2   | 8.1  | 4.7  | 10.3 |
| 2018--ECCV | [SANet](#SANet)                                | 2.6  | 13.2  | 9     | 13.3 | 3    | 8.2  |
| 2018--AAAI | [TDF-CNN](#TDF-CNN)                              | 2.7  | 23.4  | 10.7  | 17.6 | 3.3  | 11.5 |



### UCSD
| Year-Conference/Journal | Method | MAE | MSE |
| --- | --- | --- | --- |
| 2015--CVPR | [Zhang 2015](#Zhang2015)  | 1.6  | 3.31 |
| 2016--CVPR | [MCNN](#MCNN)                    | 1.07 | 1.35 |
| 2017--CVPR | [Switching CNN](#SCNN)              | 1.62 | 2.1  |
| 2017--ICCV | [ConvLSTM-nt](#ConvLSTM)               | 1.73 | 3.52 |
| 2017--ICCV | [ConvLSTM](#ConvLSTM)                 | 1.3  | 1.79 |
| 2017--ICCV | [Bidirectional   ConvLSTM](#ConvLSTM)  | 1.13 | 1.43 |
| 2018-TIP | **[BSAD](#BSAD)**                       | **1.0**   | 1.4  |
|  2018--CVPR | [ACSCP](#ACSCP)                      | 1.04 | 1.35 |
|  2018--CVPR | [CSRNet](#CSR)                    | 1.07 | 1.35 |
| 2018--ECCV | **[SANet](#SANet)**                      | 1.02 | **1.29** |
