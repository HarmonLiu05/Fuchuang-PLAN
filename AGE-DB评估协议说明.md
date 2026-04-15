# 人脸识别模型评估协议说明

## 一、LFW 验证评估协议

### 1.1 数据集
LFW标准验证集使用其中 **6000 对**图像对（3000 对正样本 + 3000 对负样本）进行二分类验证。

### 1.2 特征提取
对每张输入图像分别进行两次前向传播：
- **原图分支**：直接输入模型提取特征
- **翻转图分支**：水平翻转后输入模型提取特征

两路 512 维特征沿特征维度拼接，得到 **1024 维**最终表征。

### 1.3 相似度计算
采用**余弦相似度**度量两张人脸特征的匹配程度：

```
sim(f1, f2) = (f1 · f2) / (||f1|| × ||f2|| + ε)
```

### 1.4 10 折交叉验证
将 6000 对样本均分为 10 折（每折 600 对）。**所有相似度在交叉验证前已预先计算完成**，后续流程不涉及模型参数更新。每折执行：

1. **阈值搜索**：在 9 折训练集（5400 对）上遍历候选阈值（-1.0 到 1.0，步长 0.005），选择使该 9 折分类准确率最高的阈值
2. **测试评估**：用该阈值在剩余 1 折测试集（600 对）上计算分类准确率

轮换 10 次，使每折都作为一次测试集，得到 **10 个独立的测试准确率**。

### 1.5 最终指标
报告 10 个测试准确率的**均值**（Mean Accuracy）和**标准差**（Std）。

---

## 二、AgeDB-30 验证评估协议

### 2.1 数据集
AgeDB-30 标准验证集使用官方提供的 **12000 对**图像对（6000 对正样本 + 6000 对负样本）。其核心挑战在于**跨年龄验证**：每对图像来自同一人的不同年龄段（年龄差通常 ≥ 30 岁），用于评估模型对年龄变化的鲁棒性。

### 2.2 评估流程
与 LFW 相同，通常也采用 10 折交叉验证，但 AgeDB-30 也支持使用官方固定划分直接报告准确率。

### 2.3 最终指标
报告验证集上的**最高分类准确率**（Accuracy）。

---

## 三、本地评估参考代码

### 3.1 完整评估流程（Python）

```python
"""
人脸识别模型本地评估参考代码
适用于 LFW / AgeDB-30 等验证集
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def extract_features(model, image, device):
    """
    提取图像特征（原图 + 翻转图拼接）
    
    Args:
        model: 预训练的人脸特征提取模型
        image: PIL.Image 输入图像
        device: 计算设备 (CPU/GPU)
    
    Returns:
        torch.Tensor: 拼接后的特征向量
    """
    # 原图预处理
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # 翻转图预处理
    flipped_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # 前向传播
    original_tensor = original_transform(image).unsqueeze(0).to(device)
    flipped_tensor = flipped_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        original_feat = model(original_tensor)
        flipped_feat = model(flipped_tensor)
    
    # 拼接特征并去除 batch 维度
    combined_feat = torch.cat([original_feat, flipped_feat], dim=1).squeeze()
    return combined_feat


def compute_cosine_similarity(feat1, feat2):
    """
    计算两个特征向量的余弦相似度
    
    Args:
        feat1, feat2: 一维特征向量 (torch.Tensor 或 numpy.array)
    
    Returns:
        float: 余弦相似度值 [-1, 1]
    """
    dot_product = np.dot(feat1, feat2)
    norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
    similarity = dot_product / (norm_product + 1e-5)
    return similarity


def load_pairs_annotation(ann_file, data_root):
    """
    加载图像对标注文件
    
    Args:
        ann_file: 标注文件路径 (如 lfw_ann.txt)
        data_root: 图像根目录
    
    Returns:
        list: 每条记录为 [img1_path, img2_path, is_same]
    """
    pairs = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过表头
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            is_same, path1, path2 = parts[0], parts[1], parts[2]
            img1_path = os.path.join(data_root, path1)
            img2_path = os.path.join(data_root, path2)
            pairs.append([img1_path, img2_path, int(is_same)])
    
    return pairs


def extract_all_features(model, pairs, device):
    """
    批量提取所有图像对的特征并计算相似度
    
    Args:
        model: 预训练模型
        pairs: 图像对列表
        device: 计算设备
    
    Returns:
        numpy.ndarray: 所有图像对的预测结果 [img1_path, img2_path, similarity, is_same]
    """
    predicts = []
    
    print(f"开始提取 {len(pairs)} 对图像的特征...")
    for idx, (img1_path, img2_path, is_same) in enumerate(pairs):
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx + 1}/{len(pairs)}")
        
        # 加载图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 提取特征
        feat1 = extract_features(model, img1, device)
        feat2 = extract_features(model, img2, device)
        
        # 计算余弦相似度
        similarity = compute_cosine_similarity(feat1.numpy(), feat2.numpy())
        
        predicts.append([img1_path, img2_path, similarity, is_same])
    
    return np.array(predicts)


def find_best_threshold(predictions, thresholds):
    """
    在候选阈值中搜索最佳阈值
    
    Args:
        predictions: 预测结果数组，每条记录包含 [_, _, similarity, ground_truth]
        thresholds: 候选阈值列表
    
    Returns:
        float: 最佳阈值
    """
    best_accuracy = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        # 根据阈值进行预测
        y_pred = [1 if sim > threshold else 0 for _, _, sim, _ in predictions]
        y_true = [gt for _, _, _, gt in predictions]
        
        # 计算准确率
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold


def k_fold_cross_validation(predictions, n_folds=10):
    """
    10 折交叉验证
    
    Args:
        predictions: 所有图像对的预测结果
        n_folds: 折数
    
    Returns:
        tuple: (平均准确率, 标准差, 平均阈值, 每折准确率列表)
    """
    n_samples = len(predictions)
    fold_size = n_samples // n_folds
    
    # 候选阈值
    thresholds = np.arange(-1.0, 1.0, 0.005)
    
    accuracies = []
    best_thresholds = []
    
    for fold_idx in range(n_folds):
        # 划分训练集和测试集
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size
        
        test_set = predictions[test_start:test_end]
        train_set = np.concatenate([predictions[:test_start], predictions[test_end:]])
        
        # 在训练集上搜索最佳阈值
        best_threshold = find_best_threshold(train_set, thresholds)
        best_thresholds.append(best_threshold)
        
        # 在测试集上计算准确率
        y_pred = [1 if sim > best_threshold else 0 for _, _, sim, _ in test_set]
        y_true = [gt for _, _, _, gt in test_set]
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        accuracies.append(accuracy)
        print(f"  Fold {fold_idx + 1}/{n_folds}: Threshold={best_threshold:.4f}, Accuracy={accuracy:.4f}")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_threshold = np.mean(best_thresholds)
    
    return mean_accuracy, std_accuracy, mean_threshold, accuracies


def evaluate_model(model, model_path, ann_file, data_root, device=None):
    """
    完整评估流程
    
    Args:
        model: 模型定义
        model_path: 权重路径
        ann_file: 标注文件路径
        data_root: 图像根目录
        device: 计算设备
    
    Returns:
        dict: 评估结果
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型权重
    print(f"加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    # 2. 加载图像对
    print(f"加载图像对标注: {ann_file}")
    pairs = load_pairs_annotation(ann_file, data_root)
    print(f"  共 {len(pairs)} 对图像")
    
    # 3. 提取特征并计算相似度
    predicts = extract_all_features(model, pairs, device)
    
    # 4. 10 折交叉验证
    print("\n开始 10 折交叉验证...")
    mean_acc, std_acc, mean_thresh, fold_accs = k_fold_cross_validation(predicts, n_folds=10)
    
    # 5. 输出结果
    print("\n" + "=" * 50)
    print(f"评估结果:")
    print(f"  平均准确率: {mean_acc:.4f}")
    print(f"  标准差:     {std_acc:.4f}")
    print(f"  平均阈值:   {mean_thresh:.4f}")
    print("=" * 50)
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_threshold': mean_thresh,
        'fold_accuracies': fold_accs
    }


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 假设你已经定义好了模型
    # from models import sphere20
    
    # model = sphere20(embedding_dim=512)
    
    # 评估 LFW
    # result = evaluate_model(
    #     model=model,
    #     model_path='weights/sphere20_mcp.pth',
    #     ann_file='data/val/lfw_ann.txt',
    #     data_root='data/val',
    #     device=torch.device('cuda')
    # )
    
    print("参考代码加载完成，请根据实际情况调用 evaluate_model() 函数")
```

---

## 四、评估流程伪代码

```
========================================
  人脸识别模型评估伪代码
========================================

输入:
    - 预训练模型权重文件 (model_weights.pth)
    - 验证集图像对标注文件 (pairs.txt)
    - 验证集图像目录 (data/val)

输出:
    - 平均准确率 (Mean Accuracy)
    - 准确率标准差 (Std)
    - 平均最佳阈值 (Mean Threshold)

========================================
步骤 1: 初始化
========================================
    设备 ← 检测 GPU 可用性 (CUDA / CPU)
    模型 ← 加载网络结构
    模型.加载权重(模型权重文件)
    模型.移动到(设备)
    模型.设置为评估模式()  // 关闭 Dropout 和 BatchNorm 更新

========================================
步骤 2: 加载图像对标注
========================================
    打开标注文件
    跳过表头行
    对于每一行标注:
        解析: is_same, img1_path, img2_path
        拼接完整路径
        添加到图像对列表

========================================
步骤 3: 批量提取特征（核心推理阶段）
========================================
    初始化预测结果列表 predicts
    
    对于 图像对列表 中的每一对 (img1, img2, is_same):
        // 提取 img1 特征
        img1_原图 ← 加载图像(img1)
        img1_翻转 ← 水平翻转(img1_原图)
        
        预处理(img1_原图) → Tensor → 归一化 → 添加batch维度 → 设备
        预处理(img1_翻转) → Tensor → 归一化 → 添加batch维度 → 设备
        
        在 torch.no_grad() 下:
            feat1_原图 ← 模型.forward(img1_原图_Tensor)
            feat1_翻转 ← 模型.forward(img1_翻转_Tensor)
        
        feat1 ← 拼接(feat1_原图, feat1_翻转)  // 沿特征维度 concat
        去除 batch 维度 → 一维向量
        
        // 提取 img2 特征（同上）
        feat2 ← 提取特征(img2)
        
        // 计算余弦相似度
        相似度 ← (feat1 · feat2) / (||feat1|| × ||feat2|| + 1e-5)
        
        保存结果: [img1路径, img2路径, 相似度, is_same] → predicts

========================================
步骤 4: 10 折交叉验证
========================================
    候选阈值列表 ← 从 -1.0 到 1.0，步长 0.005 (共 400 个)
    初始化 accuracies 列表
    初始化 best_thresholds 列表
    
    对于 fold_idx 从 0 到 9:
        // 划分训练集和测试集
        测试集索引 ← [fold_idx × 600 : (fold_idx+1) × 600]
        训练集索引 ← 剩余所有样本
        
        训练集 ← predicts[训练集索引]
        测试集 ← predicts[测试集索引]
        
        // 在训练集上搜索最佳阈值
        最佳准确率 ← 0
        最佳阈值 ← 0
        
        对于 每个候选阈值 in 候选阈值列表:
            当前准确率 ← 计算分类准确率(训练集, 候选阈值)
            如果 当前准确率 > 最佳准确率:
                最佳准确率 ← 当前准确率
                最佳阈值 ← 候选阈值
        
        保存 最佳阈值 → best_thresholds
        
        // 在测试集上计算准确率
        测试准确率 ← 计算分类准确率(测试集, 最佳阈值)
        保存 测试准确率 → accuracies
        
        打印: "Fold {fold_idx+1}: 阈值={最佳阈值}, 准确率={测试准确率}"

========================================
步骤 5: 汇总结果
========================================
    平均准确率 ← 平均值(accuracies)
    标准差 ← 标准差(accuracies)
    平均阈值 ← 平均值(best_thresholds)
    
    打印: "LFW ACC: {平均准确率} ± {标准差}, Threshold: {平均阈值}"
    
    返回 平均准确率, 标准差, 平均阈值

========================================
辅助函数: 计算分类准确率
========================================
输入: 预测结果列表, 阈值
输出: 准确率

    初始化 y_true, y_pred 列表
    
    对于 每条记录 in 预测结果:
        真实标签 ← 记录的 is_same 字段
        预测标签 ← 1 if 记录的相似度 > 阈值 else 0
        
        添加 真实标签 → y_true
        添加 预测标签 → y_pred
    
    准确率 ← 均值(y_true == y_pred)
    返回 准确率
```

---

## 五、关键注意事项

### 5.1 特征拼接的意义
- 原图和翻转图分别提取特征可以增强模型对人脸姿态变化的鲁棒性
- 拼接后特征维度翻倍（512 → 1024），包含更丰富的信息

### 5.2 为什么先算相似度再做交叉验证
- 模型推理是最耗时的步骤（GPU 前向传播）
- 所有相似度只需计算一次，后续交叉验证只操作数值，避免重复推理

### 5.3 阈值搜索的原理
- 不是通过公式计算，而是暴力枚举所有候选阈值
- 在训练集上逐个测试每个阈值的分类效果
- 选择使训练集准确率最高的那个阈值

### 5.4 10 折交叉验证的优势
- 充分利用数据，每对样本都参与过一次测试
- 10 个独立准确率可以计算标准差，反映模型稳定性
- 避免单次划分的偶然性

---

## 六、常见数据集对比

| 指标 | LFW | AgeDB-30 | CFP-FP | CALFW |
|------|-----|----------|--------|-------|
| 图像对数量 | 6000 | 12000 | 10000 | 6000 |
| 正/负样本比 | 1:1 | 1:1 | 1:1 | 1:1 |
| 核心挑战 | 自然场景 | 跨年龄 | 跨姿态 (Frontal-Profile) | 跨年龄 + 跨性别 |
| 典型准确率 | 99.0%+ | 95.0%+ | 95.0%+ | 95.0%+ |
| 评估协议 | 10 折 CV | 10 折 CV / 固定划分 | 10 折 CV | 10 折 CV |
