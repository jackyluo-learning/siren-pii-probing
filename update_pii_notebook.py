import json

nb_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔒 PII Layer-Wise Probing & SIREN Pipeline Notebook\n",
    "\n",
    "本 Notebook 严格按照 ACL 2026 论文 **《LLM Safety From Within: Detecting Harmful Content with Internal Representations》**（arXiv:2604.18519）的 **1:1 论文算法规格（L1 探针网格搜索 C in {100, 200, 500, 1000}、eta=0.8 神经元筛选、alpha_l 自适应跨层融合与 LOESS 曲线平滑）** 改造实现，用于评估大模型内部隐层识别 **PII（个人身份标识：SSN/ID/年龄/信用卡/地址等）** 的能力与层级演进特征。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 1：克隆 GitHub 仓库与安装依赖"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi\n",
    "!git clone https://github.com/jackyluo-learning/siren-pii-probing.git\n",
    "%cd siren-pii-probing\n",
    "!pip install --quiet torch transformers scikit-learn matplotlib datasets tqdm seaborn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 2：1:1 论文规格 PII 表征识别与层级演进实测跑图"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 运行 1:1 论文规格改造的 PII 层级探针与 SIREN 融合实验\n",
    "# 默认采用与论文 Figure 7 一致的 Qwen 3B/4B 基座 (Qwen/Qwen2.5-3B-Instruct)\n",
    "!PYTHONPATH=. python examples/pii_layerwise_probe.py --model \"Qwen/Qwen2.5-3B-Instruct\" --train_samples 400 --test_samples 200 --pooling max\n",
    "\n",
    "# 渲染生成的高清 PII 性能曲线图\n",
    "from IPython.display import Image, display\n",
    "display(Image('pii_layerwise_performance.png'))"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('Colab_SIREN_PII_Layerwise_Probing.ipynb', 'w') as f:
    json.dump(nb_data, f, indent=2, ensure_ascii=False)

print('Updated Colab_SIREN_PII_Layerwise_Probing.ipynb with 1-to-1 paper spec!')
