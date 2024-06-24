#pragma once
#include <cstring>
#include <iostream>
#include <sstream>

#include "array.hpp"
#include "helper.hpp"
//zeros
MultiDimArray zeros(const std::vector<Index>& dims, DataType data_type);
//index
MultiDimArray index(const MultiDimArray& md_array,
                    const std::vector<Index>& indices);
//reshape
void reshape(MultiDimArray& md_array, const std::vector<Index>& dims);
//arange
MultiDimArray arange(float l, float r, float step);
//concat
MultiDimArray concat(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b, Index axis);
//vstack
MultiDimArray vstack(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b);
//hstack
MultiDimArray hstack(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b);
//split
std::vector<MultiDimArray> split(const MultiDimArray& md_array, Index axis);

//add
MultiDimArray broadcast_add(const MultiDimArray& md_array_a,const MultiDimArray& md_array_b);