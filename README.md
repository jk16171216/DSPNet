# DSPNet: A Transformer Network Based on Multi-Scale Feature Modeling and Patch Representation for Long-Term Multivariate Time Series Forecasting

## 🔍 简介
本仓库提供了 **DSPNet** 的实现代码。  
DSPNet 通过 **多尺度建模** 与 **patch 表示的 Transformer 网络**，有效提升了多变量时间序列预测的精度。实验结果表明，DSPNet 在多个公开数据集上均取得了最优性能。  

---

## 📦 环境依赖  

建议使用 **Python 3.8+**，并安装以下依赖：  
numpy
matplotlib
pandas
scikit-learn
torch==1.11.0
你也可以通过下述命令一键安装 pip install -r requirements.txt

## 🚀 运行方式
bash /scripts/DSPNet/xx.py

## 📊 数据集

实验主要使用以下时间序列预测数据集：
Electricity、ETTh1 / ETTh2、ETTm1 / ETTm2、Exchange、Traffic、Weather
请在使用前将数据下载并放置在 dataset/ 文件夹下。

