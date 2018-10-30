# Awesome Crowd Counting

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Tools](#Tools)
* [Datasets](#Datasets)
* [Papers](#Papers)
* [Leaderboard](#Leaderboard)

##  <a name="Tools"></a> Tools

- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)]


## <a name="Datasets"></a> Datasets

- UCF-QNRF Dataset [[Link](http://crcv.ucf.edu/data/ucf-qnrf/)]
- ShanghaiTech Dataset [Link: [Dropbox ](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)/ [BaiduNetdisk](https://pan.baidu.com/s/1nuAYslz)]
- WorldExpo'10 Dataset [[Link](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)]
- UCF CC 50 Dataset [[Link](http://crcv.ucf.edu/data/ucf-cc-50/)]
- Mall Dataset  [[Link](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)]
- UCSD Dataset [[Link](http://www.svcl.ucsd.edu/projects/peoplecnt/)]
- SmartCity Dataset [Link: [GoogleDrive ](https://drive.google.com/file/d/1xqflSQv9dZ0A93_lP34pSIfcpheT2Fi8/view?usp=sharing)/ [BaiduNetdisk](https://pan.baidu.com/s/1pMuGyNp)]
- AHU-Crowd Dataset [[Link](http://cs-chan.com/downloads_crowd_dataset.html)] 

## <a name="Papers"></a> Papers

### arXiv papers
This section only includes the last ten papers since 2018 in [arXiv.org](arXiv.org). Previous papers will be hidden using  ```<!--...-->```. If you want to view them, please open the [raw file](https://raw.githubusercontent.com/gjy3035/Awesome-Crowd-Counting/master/README.md) to read the source code. Note that all unpublished arXiv papers are not included into [the leaderboard of performance](#performance).

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
- <a name="todo"></a> Crowd Counting by Adaptively Fusing Predictions from an Image Pyramid (**BMVC2018**) [[paper](https://arxiv.org/abs/1805.06115)]
- <a name="todo"></a> Crowd Counting using Deep Recurrent Spatial-Aware Network (**IJCAI2018**) [[paper](https://arxiv.org/abs/1807.00601)]
- <a name="todo"></a> Top-Down Feedback for Crowd Counting Convolutional Neural Network (**AAAI2018**) [[paper](https://arxiv.org/abs/1807.08881)]
- <a name="SANet"></a> **[SANet]** Scale Aggregation Network for Accurate and Efficient Crowd Counting (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.pdf)]
- <a name="ic-CNN"></a> **[ic-CNN]** Iterative Crowd Counting (**ECCV2018**) [[paper](https://arxiv.org/abs/1807.09959)]
- <a name="todo"></a> **[CL]** Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds (**ECCV2018**) [[paper](https://arxiv.org/abs/1808.01050)]
- <a name="NCL"></a> Crowd Counting with Deep Negative Correlation Learning (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf)] [[code](https://github.com/shizenglin/Deep-NCL)]
- <a name="IG-CNN"></a> **[IG-CNN]** Divide and Grow: Capturing Huge Diversity in Crowd Images with
Incrementally Growing CNN (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sam_Divide_and_Grow_CVPR_2018_paper.pdf)]
- <a name="BSAD"></a> **[BSAD]** Body Structure Aware Deep Crowd Counting (**TIP2018**) [[paper](http://mac.xmu.edu.cn/rrji/papers/IP%202018-Body.pdf)] 
- <a name="CSR"></a> **[CSR]**  CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (**CVPR2018**) [[paper](https://arxiv.org/abs/1802.10062)] [[code](https://github.com/leeyeehoo/CSRNet)]
- <a name="L2R"></a>  **[L2R]** Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (**CVPR2018**) [[paper](https://arxiv.org/abs/1803.03095)] [[code](https://github.com/xialeiliu/CrowdCountingCVPR18)] 
- <a name="ACSCP"></a> **[ACSCP]**  Crowd Counting via Adversarial Cross-Scale Consistency Pursuit  (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf)]
- <a name="DecideNet"></a> **[DecideNet]** DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density (**CVPR2018**) [[paper](https://arxiv.org/abs/1712.06679)]
- <a name="DR-ResNet"></a> **[DR-ResNet]** A Deeply-Recursive Convolutional Network for Crowd Counting (**ICASSP2018**) [[paper](https://arxiv.org/abs/1805.05633)] 
- <a name="SaCNN"></a> **[SaCNN]** Crowd counting via scale-adaptive convolutional neural network (**WACV2018**) [[paper](https://arxiv.org/abs/1711.04433)] [[code](https://github.com/miao0913/SaCNN-CrowdCounting-Tencent_Youtu)]

### 2017
- Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (**ICCV2017**) [[paper](https://arxiv.org/abs/1708.00953)]
- Spatiotemporal Modeling for Crowd Counting in Videos (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xiong_Spatiotemporal_Modeling_for_ICCV_2017_paper.pdf)]
- CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting (**AVSS2017**) [[paper](https://arxiv.org/abs/1707.09605)] [[code](https://github.com/svishwa/crowdcount-cascaded-mtl)]
- Switching Convolutional Neural Network for Crowd Counting (**CVPR2017**) [[paper](https://arxiv.org/abs/1708.00199)] [[code](https://github.com/val-iisc/crowd-counting-scnn)]
- A **Survey** of Recent Advances in CNN-based Single Image Crowd Counting and Density
Estimation (**PR Letters**) [[paper](https://arxiv.org/abs/1707.01202)]
- Multi-scale Convolution Neural Networks for Crowd Counting (**ICIP2017**) [[paper](https://arxiv.org/abs/1702.02359)] [[code](https://github.com/Ling-Bao/mscnn)]

### 2016 

- Towards perspective-free object counting with deep learning  (**ECCV2016**) [[paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf)] [[code](https://github.com/gramuah/ccnn)]
- CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (**ACMMM2016**) [[paper](https://arxiv.org/abs/1608.06197)] [[code](https://github.com/davideverona/deep-crowd-counting_crowdnet)]
- <a name="MCNN"></a> **[MCNN]** Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (**CVPR2016**) [[paper](https://pdfs.semanticscholar.org/7ca4/bcfb186958bafb1bb9512c40a9c54721c9fc.pdf)] [unofficial code: [TensorFlow](https://github.com/aditya-vora/crowd_counting_tensorflow) [PyTorch](https://github.com/svishwa/crowdcount-mcnn)]
### 2015

- COUNT Forest: CO-voting Uncertain Number of Targets using Random Forest
for Crowd Density Estimation (**ICCV2015**) [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Pham_COUNT_Forest_CO-Voting_ICCV_2015_paper.pdf)]
- Cross-scene Crowd Counting via Deep Convolutional Neural Networks (**CVPR2015**) [[paper](https://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf)] [[code](https://github.com/wk910930/crowd_density_segmentation)]

### 2013

- Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images (**CVPR2013**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Idrees_Multi-source_Multi-scale_Counting_2013_CVPR_paper.pdf)]
- Crossing the Line: Crowd Counting by Integer Programming with Local Features (**CVPR2013**) [[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Ma_Crossing_the_Line_2013_CVPR_paper.pdf)]

### 2012

- Feature mining for localised crowd counting (**BMVC2012**) [[paper](https://pdfs.semanticscholar.org/c5ec/65e36bccf8a64050d38598511f0352653d6f.pdf)]

### 2008
- Privacy preserving crowd monitoring: Counting people without people models or tracking (**CVPR 2008**) [[paper](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr08-peoplecnt.pdf)]



## <a name="Leaderboard"></a> Leaderboard
The section is being continually updated. Note that some values have superscript, which indicates their source. 


### ShanghaiTech Part A

| Year-Conference/Journal | Method | MAE | MSE | PSNR | SSIM | Model Size | Params | Pre-trained Model |
| ---  | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018--ECCV | [SANet](#SANet) | 67.0 | 104.5 | - | - | -  | 0.91M | None |
| 2018--ECCV | [ic-CNN](#ic-CNN) | 69.8 | 117.3 | - | - | - | - | None |
| 2018--CVPR | [CSR](#CSR) | 68.2 | 115.0 | 23.79 | 0.76 | - | 16.26M<sup>[SANet](#SANet)</sup> | VGG-16 |
| 2018--CVPR | [L2R](#L2R) | 73.6 | 112.0 | - | - | - | - | VGG-16 |
| 2018--CVPR | [ACSCP](#ACSCP) | 75.7 | 102.7 | - | - | - | 5.1M | None |
| 2016--CVPR | [MCNN](#MCNN) | 110.2 | 173.2 | 21.4<sup>[CSR](#CSR)</sup> | 0.52<sup>[CSR](#CSR)</sup> | - | 0.13M<sup>[SANet](#SANet)</sup> | None |


### ShanghaiTech Part B

| Year-Conference/Journal | Method | MAE | MSE | 
| --- | --- | --- | --- | 
| 2018--ECCV | [SANet](#SANet) | 8.4 | 13.6 |
| 2018--ECCV | [ic-CNN](#ic-CNN) | 10.7 | 16.0 |
| 2018--TIP | [BSAD](#BSAD) | 20.2 | 35.6 |
| 2018--CVPR | [CSR](#CSR) | 10.6 | 16.0 |
| 2018--CVPR | [L2R](#L2R) | 13.7 | 21.4 | 
| 2018--CVPR | [DecideNet](#DecideNet) | 21.53 | 31.98 | 
| 2018--CVPR | [ACSCP](#ACSCP) | 17.2 | 27.4 | 
| 2016--CVPR | [MCNN](#MCNN) | 26.4 | 41.3 |


### UCF-QNRF

| Year-Conference/Journal | Method | MAE | MSE | 
| --- | --- | --- | --- | 
| 2018--ECCV | [CL](#CL) | 132 | 191 |
| 2016--CVPR | [MCNN](#MCNN) | 277<sup>[CL](#CL)</sup> | 426<sup>[CL](#CL)</sup> |


### UCF_CC_50
| Year-Conference/Journal | Method | MAE | MSE | 
| --- | --- | --- | --- | 
| 2018--ECCV | [SANet](#SANet) | 258.4 | 334.9 |
| 2018--ECCV | [ic-CNN](#ic-CNN) | 260.9 | 365.5 |
| 2018--TIP | [BSAD](#BSAD) | 409.5 | 563.7 | 
| 2018--CVPR | [CSR](#CSR) | 266.1 | 397.5 |
| 2018--CVPR | [L2R](#L2R) | 279.6 | 388.9 | 
| 2018--CVPR | [ACSCP](#ACSCP) | 291.0 | 404.6 | 

### WorldExpo'10
| Year-Conference/Journal | Method | S1 | S2 | S3 | S4 | S5 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2018--ECCV | [SANet](#SANet) | 2.6 | 13.2 | 9.0 | 13.3 | 3.0 | 8.2 |
| 2018--ECCV | [ic-CNN](#ic-CNN) | 17.0 | 12.3 | 9.2 | 8.1 | 4.7 | 10.3 |
| 2018--TIP | [BSAD](#BSAD) | 4.1 | 21.7 | 11.9 | 11.0 | 3.5 | 10.5 |
| 2018--CVPR | [CSR](#CSR) | 2.9 | 11.5 | 8.6 | 16.6 | 3.4 | 8.6 |
| 2018--CVPR | [DecideNet](#DecideNet) | 2.0 | 13.14 | 8.90 | 17.40 | 4.75 | 9.23 |
| 2018--CVPR | [ACSCP](#ACSCP) | 2.8 | 14.05 | 9.6 | 8.1 | 2.9 | 7.5 |

### UCSD
| Year-Conference/Journal | Method | MAE | MSE | 
| --- | --- | --- | --- |
| 2018--ECCV | [SANet](#SANet) | 1.02 | 1.29 |
| 2018--TIP | [BSAD](#BSAD) | 1.00 | 1.40 | 
| 2018--CVPR | [CSR](#CSR) | 1.16 | 1.47 |
| 2018--CVPR | [ACSCP](#ACSCP) | 1.04 | 1.35 |















