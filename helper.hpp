#pragma once
#include <sstream>
#include <string>
#include <vector>

#include "array.hpp"
inline MultiDimArray unkown() { return MultiDimArray{{}, DataType::UNKNOWN}; }
inline std::string to_flat_string(const MultiDimArray& md_array) {
  auto data_type = md_array.dtype();
  if (data_type == DataType::FLOAT32) {
    std::stringstream ss;
    ss << "md_f[";
    auto data_f = md_array.as<float>();
    for (Index i{0}; i < md_array.size(); i++) {
      ss << data_f[i] << ",";
    }
    ss << "]";
    return ss.str();
  } else if (data_type == DataType::INT32) {
    std::stringstream ss;
    ss << "md_i[";
    auto data_i = md_array.as<float>();
    for (Index i{0}; i < md_array.size(); i++) {
      ss << data_i[i] << ",";
    }
    ss << "]";
    return ss.str();
  } else {
    CHECK_WITH_INFO(false, "reach unknown data type");
    return std::string{};
  }
}

inline Index cal_block_num(const std::vector<Index>& dims, Index axis) {
  Index block_num{1};
  for (Index i{0}; i < axis; i++) {
    block_num *= dims[i];
  }
  return block_num;
}
inline Index cal_block_size(const std::vector<Index>& dims, Index axis) {
  Index block_size{1};
  for (Index i{axis + 1}; i < dims.size(); i++) {
    block_size *= dims[i];
  }
  return block_size;
}

inline std::vector<Index> cal_block_sizes(const std::vector<Index>& dims) {
  std::vector<Index> bases;
  bases.resize(dims.size(), 1);
  Index base{1};
  for (Index i = dims.size() - 1; i >= 0; i--) {
    bases[i] = base;
    base *= dims[i];
  }
  return bases;
}

inline Index cal_offset(const std::vector<Index>& bases,
                        const std::vector<Index>& indices) {
  Index offset{0};
  for (Index i{0}; i < bases.size(); i++) {
    offset += bases[i] * indices[i];
  }
  return offset;
}
inline Index cal_offset(const std::vector<Index>& bases,
                        const std::vector<Index>& indices,
                        const std::vector<Index>& dims) {
  Index offset{0};
  for (Index i{0}; i < bases.size(); i++) {
    offset += bases[i] * (indices[i] % dims[i]);
  }
  return offset;
}

inline std::string to_string(const std::vector<Index>& vs){
    std::stringstream ss;
    ss << "index[";
    for (Index i{0}; i < vs.size(); i++) {
      ss << vs[i] << ",";
    }
    ss << "]";
    return ss.str();
}