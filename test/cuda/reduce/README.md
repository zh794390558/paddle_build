* reduce_sum_all.cu  , using CUB reduce all dims
(64, 64, 100) -> (1)
* reduce_sum_leading_dim.cu, first implement this, (reduce_num, left_num)
(64, 64, 100) -> (1, 100)
(64, 64, 100) -> (1, 64, 100)
* reduce_sum_last_dim.cu, modified from ReduceHighDim, (left_num, reduce_num)
(64, 64, 100) -> (64, 64, 1)
(64, 64, 100) -> (64, 1, 1)
