# DSPNet: A Transformer Network Based on Multi-Scale Feature Modeling and Patch Representation for Long-Term Multivariate Time Series Forecasting

## 🔍 Introduction
This repository provides the implementation of **DSPNet**.  
DSPNet employs **multi-scale modeling** and **patch-based Transformer networks** to effectively improve the accuracy of multivariate time series forecasting. Experimental results demonstrate that DSPNet achieves state-of-the-art performance on multiple public datasets.  


---

## 📦 Dependencies  

It is recommended to use **Python 3.8+** and install the following dependencies:  
- numpy  
- matplotlib  
- pandas  
- scikit-learn  
- torch==1.11.0  

You can also install all dependencies with the following command:  pip install -r requirements.txt



## 🚀 Usage  

To run the model, use the following command:  bash /scripts/DSPNet/xx.sh


## 📊 Datasets  

The experiments are conducted on the following benchmark time series datasets:  

ETTh/ETTm: https://github.com/zhouhaoyi/ETDataset

Electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Traffic: https://github.com/liyaguang/DCRNN

Weather: https://www.ncei.noaa.gov

Please download the datasets before use and place them in the `dataset/` directory.  


