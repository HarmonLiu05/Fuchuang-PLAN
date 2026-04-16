### 已经经过裁剪对齐之后的数据集224*224 符合imgnet预训练权重的图片输入大小 

#### 训练集：

[**AgeDB**](https://huggingface.co/datasets/lhx05/agedb-224-by-identity)

图片总数：16488

[**Webface**](https://huggingface.co/datasets/lhx05/WebFace224)

图片总数：490623

#### 验证集：

[**除了FGnet的汇总**](https://huggingface.co/datasets/lhx05/lab1-resized-face-datasets)

[**FGnet**](https://huggingface.co/datasets/lhx05/fgnet-age30-protocol) 

图片数：1002

## kaggle原版112大小Dataset

You can download aligned and cropped (112x112) training and validation datasets from Kaggle.

### Training Data

- [CASIA-WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112) from `opensphere`
  - Identities: 10.6k
  - #Images: 491k
- [VGGFace2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112) from `opensphere`
  - Identities: 8.6k
  - #Images: 3.1M
- [MS1MV2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset) from `insightface`
  - Identities: 85.7k
  - #Images: 5.8M

### Validation Data

Validation data contains AgeDB_30, CALFW, CPLFW, and LFW datasets.

- [AgeDB_30, CALFW, CPLFW, LFW 112x112](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112)

数据集下载地址。

暂时可以只是用照片数最少的CASIA作为训练集 拿最后训练集跑出来的权重进行测试集：验证集使用AgeDB和LFW

