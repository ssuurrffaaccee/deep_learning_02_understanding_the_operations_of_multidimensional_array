# 理解多维数组

这个仓库是用一种具体的内存表示来深入理解多维数组的基本操作。

## 从内存存储上理解多维数组
  对于形状为[3,4,5]的多维整数数组A，假定其存储在连续内存段上：  

  0. 从下标的第一个维度分配，对于A对应的内存必然要被均分为3块，   
  
  1. 进一步再考虑下标的第二个维度的分配，上一步中分出的每一块都需要进一步被均分为4块，  
  
  2. 进一步再考虑下标的第三个维度的分配，上一步中分出的每一块都需要进一步被均分为5块，  
  
  至此下标维度被耗尽了。很明显这是一个递归过程。划分完成后每一个块都对应着A中相应的一个整数。  

  从上面数组A的内存不断划分的过程中，容易观察到：多维数组的下标的每一个维度是对内存块的一个某粒度粒度大小的划分。  
  
  对于多维数组的每一个下标维度,我们都可以问出一下问题：  
  
     0. 对于当前下标维度，在上述的递归划分过程中，其对应的均分过程中，上一步的每一块被均分为大小为多少的块？  
  
     1. 对于当前下标维度，在上述的递归划分过程中，其对应的均分过程，需要均分了多少个上一步的块？  
  
  再次考察数组A，回答上面两个问题：  
  
     0. 考虑下标的第0维度，有：需要将1个整块中的每一块划分为2块，每个块的大小应该为3x4  
  
     1. 考虑下标的第1维度，有：需要将1x2个整块中的每一块划分为3块，每个块的大小应该为4  
  
     2. 考虑下标的第2维度，有：需要将1x2x3个整块中的每一块划分为4块，每个块的大小应该为1  
  
  总结一下：  
  
     对于大小为[a0,a1,a2,...,an]的多维数组，考虑下标的第i维度，有：需要将 a0*..*a(i-1)个整块中的每一个整块划分为ai块，每个块的大小应该为a(i+1)*..*an;  
  
     我们定义：  
  
     0. block_nums(i)为a0*..*a(i-1)，表示有多少整块要划分  
  
     1. block_split_size(i)为a(i+1)*..*an，表示每个整块要被以多少的大小均分。 

     那么形状为[a0,a1,a2,...,an]的多维数组，对于第i维度，总是可以看作存储没有变化下的形状为[block_nums(i),ai,block_split_size(i)]的下标维度为3的多维数组。
     
     进一步，可以看作形状为[block_nums(i),ai*block_split_size(i)]的下标维度为2的多维数组
     
     进一步，下标维度为2的多维数组在内存上，可以使用一个起始点和一个步长进行循环访问，这是编程所需要的视角。
## concat 
  假设要要沿着axis=1对多维数组A[2,3,4]和B[2,2,4]进行concat： 
  
  使用上面的结论：  
  
     0. 对A有：沿着下标的第1维度，block_nums(1)为2，block_split_size(1)为4，可视化为(1表示一个位置或者一个数)：  
                    [[1111,1111,1111][1111,1111,1111]]  
           从头开始只要使用步长3*block_split_size(1),我们就可以在数组内存上按照未被分割的块进行顺序访问  
           当到达一个块，使用步长block_split_size(1),我们就可以在数组内存上按照被分割后的块进行顺序访问   
  
     1. 对B有：沿着下标的第1维度，block_nums(1)为2，block_split_size(i)为4，可视化为：  
                    [[1111,1111][1111,1111]]   
           从头开始只要使用步长2*block_split_size(1),我们就可以在数组内存上按照未被分割的块进行顺序访问   
           当到达一个块，使用步长block_split_size(1),我们就可以在数组内存上按照被分割后的块进行顺序访问 

  A和Bconcat的结果为：
           [[1111,1111,1111,1111,1111][1111,1111,1111,1111,1111]]   
   
   c++伪代码为：
```cpp
  auto axis=1
  auto A_block_size = A.dims[axis]*block_size(A.dims,axis);
  auto B_block_size = B.dims[axis]*block_size(A.dims,axis);
  for(Index i{0}; i < block_nums(A.dims,axis); i++){
      memcpy(res_data_ptr + i*(A_block_size + B_block_size),                 A_data_ptr + i * A_block_size, A_block_size * data_type_size);
      memcpy(res_data_ptr + i*(A_block_size + B_block_size) + A_block_size , B_data_ptr + i * B_block_size, B_block_size * data_type_size);
  }
```

# split,index和 broadcast_add
  split和index操作在`funcs.cpp`中有实现.