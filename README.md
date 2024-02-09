This repository contains implementations of tensor methods for general (nonuniform) hypergraphs, including Z- and H-eigenvector centrality and CP decomposition from this [paper](https://arxiv.org/pdf/2306.17825.pdf).

If you use this code, please cite

@article{aksoy2023scalable,
  title={Scalable tensor methods for nonuniform hypergraphs},
  author={Aksoy, Sinan G and Amburg, Ilya and Young, Stephen J},
  journal={arXiv preprint arXiv:2306.17825},
  year={2023}
}

The hgx, hnx, simplehypergraphs, and xgi folders contain code that will be embeded in the respective hypergraph analytics libraries ([HypergraphX](https://hypergraphx.readthedocs.io/en/latest/#) (hgx), [HyperNetX](https://pnnl.github.io/HyperNetX/) (hnx), [SimpleHypergraphs.jl](https://pszufe.github.io/SimpleHypergraphs.jl/stable/) (simplehypergraphs), and [CompleXGroupInteractions](https://xgi.readthedocs.io/en/stable/) (xgi)).

The standalone-code folder contains code that you could run independent of the above libraries. The tensor-methods-for-hypergraphs-tutorial.ipynb notebook walks you through how to use the tensor times same vector functions (ttsv1, ttsv2), perform Z- and H-eigenvector centrality computations, and compute a hypergraph's associated tensor CP-decomposition.
