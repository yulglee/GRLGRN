# GRLGRN:a novel graph representation learning approach of inferring gene regulatory networks from single-cell RNA-seq data

## Table of Contents
1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---

## 1. Introduction
 We propsed a novel graph representation learning approach of inferring gene regulatory networks from single-cell RNA-seq data, namely GRLGRN. The architecture of inference model GRLGRN consists of gene embedding module, feature enhancement module, and output module. The main task of GRLGRN 
is to infer the potential regulatory dependencies between genes. To evaluate the performance of GRLGRN, we compare it with other models on seven different cell line datasets with three different ground-truth networks. The results illustrate that GRLGRN achieves considerable or better performance. Here, the codes for implementing, training, and testing GRLGRN are provided.
## 2. Python Environment
Python 3.8 and packages version:
- cuda==12.1
- matplotlib==3.2.2
- seaborn==0.13.2
- torch==2.1.0
- numpy==1.24.3 
- pandas==1.3.4
- scipy==1.4.1  
- scikit-learn==0.23.1
## 3. Project Structure
### 3.1 **Dataset**
For this study, we choose the database from BEELINE (Pratapa et al., 2020) comprises single-cell RNA-seq data from seven different types of cell lines: (i) human embryonic stem cells (hESC); (ii) human mature hepatocytes (hHEP); (iii) mouse dendritic cells (mDC); (iv) mouse embryonic stem cells (mESC); (v) mouse hematopoietic stem cells with erythroid-lineage (mHSC-E); (vi) mouse hematopoietic stem cells with granulocyte-monocytelineage (mHSC-GM); (vii) mouse hematopoietic stem cells with lymphoid-lineage (mHSC-L). Each cell line corresponds to three different ground-truth networks with varying densities documented in STRING (Szklarczyk et al., 2019), cell-type-specific ChIP-seq data (Xu et al., 2013), and non-specific ChIP-seq (Garcia-Alonso et al., 2019). According to the work of Yang et al, we regard any gene pair whose regulatory dependency appears in the ground-truth GRN of a dataset as a positive sample, which is labeled as 1; conversely, we regard any gene pair whose regulatory dependency is not in the ground-truth GRN of a dataset as a negative sample, which is labeled as 0. Subsequently, the positive and negative samples are randomly selected into training, validation and testing subsets according to a certain ratio.
### 3.2 **Model**
The architecture of inference model GRLGRN consists of gene embedding module, feature enhancement module, and output module.
 ![Model Architecture](https://github.com/yulglee/GRLGRN/blob/main/GRLGRN_model.jpg)
 - The file .pth is the GRLGRN model trained on the training subset of the corresponding scRNA-seq dataset.
 - To train the model, we can run `train.py` script using the training dataset. We can also run `test.py` to test the model
### 3.3 **script**
- `utils.py` randomly splits scRNA-seq datasets and feed them into the GRLGRN in batchsizes for training, validation, and testing.
- `utils2.py` converts a prior GRN into subgraphs and preprocesses gene expression profile data.
- `parser.py` defines the relevant hyperparameters in GRLGRN.
- `contrastive_learning.py` is used to compute the graph contrastive learning regularization term between embeddings of different genes.
- `compute_metrics.py` is used to calculate the performance metrics of the model
- `CBAM.py` implements the CBAM which integrates channel attention and spatial attention mechanisms, enhancing the representation of local features by emphasizing important channels and spatial information.
-  `GRLGRN_layer.py` uses a graph transformer network and CBAM to obtain gene representations, which are used for downstream predictions of gene interactions.





