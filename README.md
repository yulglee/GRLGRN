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
We introduce a novel method for inferring gene regulatory networks from single-cell RNA-seq data, named GRLGRN. The model comprises three key components: a gene embedding module, a feature enhancement module, and an output module. The main goal of GRLGRN is to uncover potential regulatory interactions between genes. To evaluate its effectiveness, we compared GRLGRN with several existing models across seven distinct cell line datasets and three different ground-truth networks. The results show that GRLGRN either outperforms or demonstrates comparable performance to current approaches. The implementation, training, and testing code for GRLGRN is also provided.

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
In this code, we utilized the BEELINE database (Pratapa et al., 2020), which includes single-cell RNA-seq data from seven distinct cell line types: (i) human embryonic stem cells (hESC), (ii) human mature hepatocytes (hHEP), (iii) mouse dendritic cells (mDC), (iv) mouse embryonic stem cells (mESC), (v) mouse hematopoietic stem cells with erythroid lineage (mHSC-E), (vi) mouse hematopoietic stem cells with granulocyte-monocyte lineage (mHSC-GM), and (vii) mouse hematopoietic stem cells with lymphoid lineage (mHSC-L). For each cell line, we used three different ground-truth networks with varying densities: STRING (Szklarczyk et al., 2019), cell-type-specific ChIP-seq data (Xu et al., 2013), and non-specific ChIP-seq data (Garcia-Alonso et al., 2019). Following the methodology from Yang et al., gene pairs exhibiting regulatory dependencies in the ground-truth GRN of each dataset were labeled as positive samples (1), while gene pairs not exhibiting these dependencies were labeled as negative samples (0). We then randomly divided the positive and negative samples into training, validation, and test subsets based on a specified ratio. For illustrative purposes, we provide the datasets and model parameters for mESC (TFs + 500) across three different ground-truth networks.


### 3.2 **Model**
The architecture of inference model GRLGRN consists of gene embedding module, feature enhancement module, and output module.
 ![Model Architecture](https://github.com/yulglee/GRLGRN/blob/main/GRLGRN_model.jpg)
 - The file .pth is the GRLGRN model trained on the training subset of the corresponding scRNA-seq dataset.
 - To train the model, we can run `train.py` script using the training dataset. We can also run `test.py` to test the model
### 3.3 **Script**
- `utils.py` randomly splits scRNA-seq datasets and feed them into the GRLGRN in batchsizes for training, validation, and testing.
- `utils2.py` converts a prior GRN into subgraphs and preprocesses gene expression profile data.
- `parser.py` defines the relevant hyperparameters in GRLGRN.
- `contrastive_learning.py` is used to compute the graph contrastive learning regularization term between embeddings of different genes.
- `compute_metrics.py` is used to calculate the performance metrics of the model
- `CBAM.py` implements the CBAM which integrates channel attention and spatial attention mechanisms, enhancing the representation of local features by emphasizing important channels and spatial information.
-  `GRLGRN_layer.py` uses a graph transformer network and CBAM to obtain gene representations, which are used for downstream predictions of gene interactions.





