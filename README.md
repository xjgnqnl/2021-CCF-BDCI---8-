
2021 CCF BDCI 基于飞桨实现花样滑冰选手骨骼点动作识别-第8名方案

本方案基于官方Baseline提供的AGCN和STGCN进行了改进，并且引入了新的模型posec3d，新的posec3d模型
在fsd-30花样滑冰比赛数据上面有良好的性能

### 一.基于官方Baseline的改进
#### 针对可视化出来的数据分析，发现数据中大量存在“瞬移”的帧，即前后连续两帧之间差异巨大，而且是几乎所有keypoint都进行了跳变
针对上述的现象，我们猜测可能是镜头的切换导致的，对于动作分类来说，动作连续性尤为重要，这种“瞬移”的帧对于动作识别模型来说，会产生
很大的干扰，所以我们进行的第一个改进是通过简单的筛选算法把这些“瞬移”的突变帧给全部置为0，我们也尝试了删除这些“瞬移”帧的实验，但是发现
效果一般，而把“瞬移”帧置为0，则效果要比Baseline高不到2个点这也说明了应该维持时间维度的信息。

#### 另一个发现的问题是，在很多帧中，大量存在置信度为0的keypoint，即数据是干扰较大的
针对这个问题，我们分析了，在官方Baseline中的数据预处理中，没有很好的考虑到这一点，而是让每帧的所有点都减去人体的中心点，但是对于这些
置信度为0的点，它们减去人体的中心点，就会得到很脏的坐标，所以我们在数据预处理中，考虑到了这一点，并且通过置信度来控制相对坐标的生成。

#### 官方给出的Baseline只尝试了keypoint本身信息，没有考虑到bone，motion这些信息
针对这个问题，我们充分考虑了bone，motion以及joint-bone-motion这种更强的表示能力的数据

### 二.训练命令
#### STGCN模型
python main.py -c configs/recognition/stgcn/stgcn_fsd_joint.yaml

python main.py -c configs/recognition/stgcn/stgcn_fsd_bone.yaml

python main.py -c configs/recognition/stgcn/stgcn_fsd_concate.yaml

python main.py -c configs/recognition/stgcn/stgcn_fsd_motion.yaml

#### AGCN模型
python main.py -c configs/recognition/agcn/agcn_fsd_joint.yaml

python main.py -c configs/recognition/agcn/agcn_fsd_bone.yaml

python main.py -c configs/recognition/agcn/agcn_fsd_concate.yaml

python main.py -c configs/recognition/agcn/agcn_fsd_motion.yaml

以上命令分别可以得到STGCN，AGCN模型分别在joint，bone，concate-joint-bone-motion，motion四种数据模态上面的结果，模型
文件分别存储在output文件夹中

### 二.测试命令
#### STGCN模型
python main.py --test -c configs/recognition/stgcn/stgcn_fsd_joint.yaml -w output/STGCN/STGCN_epoch_00090.pdparams

······

#### AGCN模型
······

每个模型测试完成，都会在当前目录生成一个score.pkl的分数矩阵
