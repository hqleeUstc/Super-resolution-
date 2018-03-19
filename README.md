# Network-based Super-resolution
Latest Research on super-resolution

## 图像空间超分辨率
图像的空间超分辨率从观测到的低空间分辨率图像重建出相应的高分辨率图像。因为SR是一个逆问题，一个低分辨率图存在多张高分辨率图与之对应，传统方法会引入先验信息进行规范性约束。神经网络的方法直接学习数据包含的低/高分辨率小块之间的映射。

**先对图像bicubic插值**
1. SRCNN 2014, Image Super-Resolution Using Deep Convolutional Networks, [Project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html). 
三层卷积网，先用bicubic插值对图像进行upsample，然后用三层卷积网恢复高频信息。因为对高分辨率图作用，运算复杂度大。
2. VDSR 2016，Accurate Image Super-Resolution Using Very Deep Convolutional Networks， [Paper](https://arxiv.org/abs/1511.04587), [Code](https://github.com/huangzehao/caffe-vdsr) . 先用bicubic插值对图像进行upsample，然后用20层卷积网学习残差。利用残差学习加快学习速率，学习率高，用gradient clipping避免梯度爆炸。SGD。
**直接在LR空间操作**
2. FSRCNN 2016，Accelerating the Super-Resolution Convolutional Neural Network， [Project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html). 
先在低分辨率图上用卷积层提取特征，然后用卷积核为1x1的卷积层进行shrink，限制后续的feature mapping过程在低维空间进行。做完feature mapping后，用卷积核为1x1的卷积层进行expand，提升feature maps数量。最后进行deconvolution，对图像进行分辨率提升。每个卷积层后的激活函数是PReLU。
3. ESPCN 2016，Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, [Paper](https://arxiv.org/abs/1609.05158), [Code](https://github.com/leftthomas/ESPCN).开创性的提出了sub-pixel convolution概念，实现了depth-to-sapce思想，对不同channel用不同的卷积核卷积，提升空间分辨率。比如进行r=2倍上采样，则sub-pixel convolution是r^2c个channel的低分辨率图，输出是c个channel的高空间分辨率图。每个channel的卷积核对应了一种pattern，是低分辨率的像素在高分辨率图中所处的位置pattern。卷积核每次滑动1/r，所以每在一个方向上滑动r次，参与运算LR的pixel变动1次（这r次每次都是用同样的LR中的点，只不过对应卷积核pattern不同），根据被active的权重位置的不同，卷积核共有r^2种pattern，其中active最多的pattern有ceil(ks/r)^2个位置（权重）被actived。
4. LaPSRN 2017, [Paper](https://arxiv.org/pdf/1710.01992.pdf), [Code](https://github.com/twtygqyy/pytorch-LapSRN) 采用级联金字塔结构，可生成多种分辨率的SR结果，用卷积层在LR空间提取特征，和进行feature mapping，然后用deconvolutional layer进行上采样，输出的特征图一方面连卷积层输出残差，和upsample的低分辨率图一起重构该分辨率下的SR结果，一方面输入下一个分辨率的超分辨率网。三层的金字塔可以实现x2,x4,x8的超分辨率。
5. EDSR, MDSR cvpr challenge 2017 冠军，[Paper](https://arxiv.org/abs/1707.02921). EDSR：借鉴ResNet提出适用于SR的ResBlock，Resblock由Conv-ReLu-Conv+0.1的residual scaling组成。另外由于卷积层占内存为O(B^2F),B为网络层数，F为feature maps数，所以网络采用增大feature maps数的方法增强网络的表达能力，同时用residual scaling让训练更加稳定。另外，网络用x2的模型预训练x3,x4的模型。也采用L1 loss加快收敛速度。EDSR很宽，有32个卷积层，每层256个 maps。MDSR：是EDSR的改进版，可同时进行x2.x3.x4模型的训练。在网络的前端设置pre-processing modules，对三种scale的模型都用两个kernel size为5x5的Resblocks提取特征，对输入的不同data激活对应的预处理module，再经过fearure mapping，map后的图进入对应scale的卷积层，输出对应scale的超分辨率结果。总而言之，网络的前端和末端是scale-variant的，中间是scale-invariant的。前端的kernel size大，其他的kernel size都是5x5. 另外，训练时还采用了geomeric self-ensemble，对输入数据进行8中几何变换，每种几何变换产生的数据去训练一个模型，最后的输出是8种模型输出结果的平均值。
6. **RDBnet CVPR 2018**, 将ResNet和Dense Net同时在SR中应用，提出 residual dense block（RDB）的概念，在每个RDB中有三个kernel size为3x3的卷积层+relu，前面的所有卷积层的输出会作为后面所有卷积层的输入，构成dense connections，然后接一个kernel size为1x1的卷积层进行shrink，降低feature maps数量，输出的feature map和RDB的输入加在一起作为总输出，即引入残差学习，卷积网学习的是残差。在RDB内部有local residual learning和local feature fusion（dense结构和1x1的卷积）。多个RDB的连接方式也是dense的，也有global residual learning。不仅可处理普通图像SR，还可以处理有噪声和blur的情况，效果都很好。

## 视频空间超分辨率
视频超分辨率的常规步骤：1.对输入视频进行运动估计和运动补偿，消除大的 motion blur；2.进行多帧融合，把多帧的信息融合在一起；3.超分辨率
1. VSRnet 2016, 先对视频各帧进行bicubic上插值，然后用Druleas算法估计光流（结合了LG思路和total variation），并采用adaptive MC的思路，如果运动很大和有遮挡，主要用当前帧信息，如果运动小，相邻帧的信息用的多。做完MC后输入超分辨率网络，在第一个卷积层之前concatenate channels或第二层之前，即采用fusion的方式融合多帧信息。考虑到前后帧相对于当前帧的对称性，前后帧的卷积层share weights。
2. VESPCN 2017. end-to-end的视频空间SR网络。[Paper](https://arxiv.org/pdf/1611.05250), 用空间变换网的思路进行运动补偿，然后用early fusion、 fusion或3D conv进行多帧融合，然后用sub-pixel convolution实现空间超分辨率，速度非常快。
3. SPMC 2017, Detail-revealing Deep Video Super-resolution, [Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tao_Detail-Revealing_Deep_Video_ICCV_2017_paper.pdf), [Code](https://github.com/hqleeUstc/SPMC_VideoSR). 腾讯公司的ICCV 2017的成果，通过构建sub-pixel motion compensation(SPMC)层，在运动补偿的同时进行分辨率的提升，超分辨率采用encoder-LSTM-decoder的思路，有skip connections。
4. FRVSR **CVPR 2018**，[Paper](https://arxiv.org/pdf/1801.04590), 对前后两低分辨率帧，采用encoder-decoder估计光流，用双线性插值upscale光流，并warp前一帧高光谱图，warp后的图用space-to-depth变成低分辨率图，和当前帧的低分辨率图一起进行超分辨率，超分辨率用的是EnhanceNet。
																											
																											
