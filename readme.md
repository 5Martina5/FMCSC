## Bridging Gaps: Federated Multi-View Clustering in Heterogeneous Hybrid Views

Code for the paper "Bridging Gaps: Federated Multi-View Clustering in Heterogeneous Hybrid Views". (NeurIPS 2024)
## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.9 or higher

## Example execution 
To train a new model, run:

```execution
python main.py 
```

Further settings for the dataset, number of clients, multi-view clients / single-view clients, and other parameters can be configured in main.py. 


You can also transform it into an IMVC method for comparison by changing the number of clients and the ratio of multi-view clients / single-view clients. 
For example, in the MNIST-USPS dataset with a missing rate of 0.5, run the code as:
```execution
python main.py --dataset='MNIST-USPS' --num_users=2 --M_S=1
```

## Citation 
If you find our code useful, please cite:
```latex
@InProceedings{chen2024,
    author    = {Xinyue Chen,Yazhou Ren,Jie Xu,Fangfei Lin,Xiaorong Pu,Yang Yang},
    title     = {Bridging Gaps: Federated Multi-View Clustering in Heterogeneous Hybrid Views},
    booktitle = {NeurIPS},
    year      = {2024},
    pages     = {1-23}
}
```
If you have any problems, please contact me by martinachen2580@gmail.com.