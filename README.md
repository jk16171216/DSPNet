# DSPNet: A Transformer Network Based on Multi-Scale Feature Modeling and Patch Representation for Long-Term Multivariate Time Series Forecasting

## Description 
This repository provides the official implementation of **DSPNet**, a deep learning framework for long-term multivariate time series forecasting. The project includes source code, datasets, and experimental scripts for reproducing the results reported in our research. DSPNet introduces a multi-scale modeling approach and uses patch representations, combined with a Transformer-based architecture, to effectively capture more comprehensive multi-scale temporal dependencies. This repository allows researchers and practitioners to: - Reproduce experimental results on benchmark datasets. - Apply DSPNet to their own multivariate time series forecasting tasks. - Extend or adjust the architecture for applications related to AI and data-driven modeling.

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


