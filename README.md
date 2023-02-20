# pytorch-transformer-reranker-contrastive
pytorch implementation of reranker using triplet loss.

### Information about the code:

1. This is a PyTorch - PyTorch Lightning based reranker using the Triplet Loss.
2. Your `BATCH_SIZE` should be equal to the number of candidates, so a single batch should contains all candidates.
3. The dataloader should yield as follows: 
  - `batch[0]` -> Tensor of size (`B x (len(I))`), where I is the Input. 
  - `batch[1]` -> Tensor of size (`B x (len(C))`), where C is the Candidate. 
4. pytorch_lightning `predict()` function automatically pick the best prediction.
