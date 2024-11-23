# Hybrid Consistency Data Augmentation for Graph Anomaly Detection

PyTorch implementation of the paper "[Hybrid Consistency Data Augmentation for Graph Anomaly Detection]()".

#  Requirments
+ torch==2.4.0+cu121
+ torchvision==0.19.0+cu121
+ torch-geometric==2.6.1
+ torch-scatter==2.1.2
+ torch-sparse==0.6.18
+ scikit-learn==1.3.2
+ dgl==1.1.3+cu121
# Preprocessing

## Dataset
Download datasets into file './data/'  
  
[Amazon] and [Yelp]:  [https://docs.dgl.ai/en/0.8.x/api/python/dgl.data.html](https://docs.dgl.ai/en/0.8.x/api/python/dgl.data.html) 

[T-Finance] and [T-Social]:  [https://github.com/squareRoot3/Rethinking-Anomaly-Detection](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) 

## Structural Consistency Augmentation
We use the diffusion wavelets to generate structural embeddings for nodes with high structral consistency, which are saved into file './data/xx_embedding.csv'.
We put the generated structural embeddings of Amazon in './data/amazon_embedding.csv' for quick access.

The Structural Consistency Augmentation of other datasets below as an example for running instructions:
```python
# Amazon
python diffusion.py --dataset amazon --output /data/amazon_embedding.csv
# YelpChi
python diffusion.py --dataset yelp --output /data/yelp_embedding.csv
# T-Finance
python diffusion.py --dataset tfinance --output /data/tfinance_embedding.csv
# T-Social
python diffusion.py --dataset tsocial --output /data/tsocial_embedding.csv
```
## Running

Running HyGAD on four benchmark datasets, which the hyper parameter settings are in the 'config/xx.yml':
```python
# Run Amazon
python main.py --config config/amazon.yml --runs 5
# Run YelpChi
python main.py --config config/yelp.yml --runs 5
# Run T-Finance
python main.py --config config/tfinance.yml --runs 5
# Run T-Social
python main.py --config config/tsocial.yml --runs 5

```


