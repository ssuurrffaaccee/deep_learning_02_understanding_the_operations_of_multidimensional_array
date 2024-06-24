#include "funcs.hpp"
MultiDimArray zeros(const std::vector<Index>& dims,
                    DataType data_type = DataType::FLOAT32) {
  MultiDimArray md_array{dims, data_type};
  if (data_type == DataType::FLOAT32) {
    auto data_f = md_array.as<float>();
    for (Index i{0}; i < md_array.size(); i++) {
      data_f[i] = 0.0f;
    }
    return md_array;
  } else if (data_type == DataType::INT32) {
    auto data_i = md_array.as<int32_t>();
    for (Index i{0}; i < md_array.size(); i++) {
      data_i[i] = 0;
    }
    return md_array;
  } else {
    CHECK_WITH_INFO(false, "reach unknown data type");
    return unkown();
  }
}
MultiDimArray index(const MultiDimArray& md_array,
                    const std::vector<Index>& indices) {
  auto& dims = md_array.shape();
  CHECK_WITH_INFO(dims.size() == indices.size(), "Index size exceed!!!");
  for (Index i{0}; i < dims.size(); i++) {
    CHECK_WITH_INFO(indices[i] < dims[i], "Index exceed!!!")
  }
  Index offset{0};
  Index base{1};
  for (Index i = dims.size() - 1; i >= 0; i--) {
    offset += indices[i] * base;
    base *= dims[i];
  }
  auto data_type = md_array.dtype();
  MultiDimArray res_md_array{{}, data_type};
  if (data_type == DataType::FLOAT32) {
    auto data_f_s = md_array.as<float>();
    auto data_f_t = res_md_array.as<float>();
    data_f_t[0] = data_f_s[offset];
    return res_md_array;
  } else if (data_type == DataType::INT32) {
    auto data_i_s = md_array.as<float>();
    auto data_i_t = res_md_array.as<float>();
    data_i_t[0] = data_i_s[offset];
    return res_md_array;
  } else {
    CHECK_WITH_INFO(false, "reach unknown data type")
    return unkown();
  }
}
void reshape(MultiDimArray& md_array, const std::vector<Index>& dims) {
  md_array.set_new_dims(dims);
}

MultiDimArray arange(float l, float r, float step) {
  CHECK(l < r);
  Index size{0};
  for (float v{l}; v < r; v += step) {
    size++;
  }
  MultiDimArray md_array{
      size <= 1 ? std::vector<Index>{} : std::vector<Index>{size},
      DataType::FLOAT32};
  auto data_f = md_array.as<float>();
  for (Index i{0}; i < size; i++) {
    data_f[i] = l + i * step;
  }
  return md_array;
}
MultiDimArray concat(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b, Index axis) {
  CHECK(md_array_a.dtype() == md_array_b.dtype());
  auto& dims_a = md_array_a.shape();
  auto& dims_b = md_array_b.shape();
  CHECK(dims_a.size() == dims_b.size());
  CHECK(axis < dims_a.size());
  std::vector<Index> res_dims;
  res_dims.resize(dims_a.size());
  for (Index i{0}; i < dims_a.size(); i++) {
    if (i != axis) {
      CHECK(dims_a[i] == dims_b[i]);
      res_dims[i] = dims_a[i];
    } else {
      res_dims[i] = dims_a[i] + dims_b[i];
    }
  }
  Index block_size = cal_block_size(dims_a, axis);
  Index block_nums = cal_block_num(dims_a, axis);
  Index res_block_size = res_dims[axis] * block_size;
  Index a_block_size = md_array_a.shape()[axis] * block_size;
  Index b_block_size = md_array_b.shape()[axis] * block_size;
  if (md_array_a.dtype() == DataType::FLOAT32 &&
      md_array_b.dtype() == DataType::FLOAT32) {
    MultiDimArray res{res_dims, DataType::FLOAT32};
    auto data_a_f = md_array_a.as<float>();
    auto data_b_f = md_array_b.as<float>();
    auto data_res_f = res.as<float>();
    for (Index i{0}; i < block_nums; i++) {
      std::memcpy(data_res_f + (i * res_block_size),
                  data_a_f + (i * a_block_size), a_block_size * sizeof(float));
      std::memcpy(data_res_f + (i * res_block_size + a_block_size),
                  data_b_f + (i * b_block_size), b_block_size * sizeof(float));
    }
    return res;
  }
  if (md_array_a.dtype() == DataType::INT32 &&
      md_array_b.dtype() == DataType::INT32) {
    MultiDimArray res{res_dims, DataType::INT32};
    auto data_a_i = md_array_a.as<int32_t>();
    auto data_b_i = md_array_b.as<int32_t>();
    auto data_res_i = res.as<int32_t>();
    for (Index i{0}; i < block_nums; i++) {
      std::memcpy(data_res_i + (i * res_block_size),
                  data_a_i + (i * a_block_size),
                  a_block_size * sizeof(int32_t));
      std::memcpy(data_res_i + (i * res_block_size + a_block_size),
                  data_b_i + (i * b_block_size),
                  b_block_size * sizeof(int32_t));
    }
    return res;
  }
  CHECK_WITH_INFO(false, "reach unknown data type")
  return unkown();
}

MultiDimArray vstack(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b) {
  return concat(md_array_a, md_array_b, 0);
}
MultiDimArray hstack(const MultiDimArray& md_array_a,
                     const MultiDimArray& md_array_b) {
  return concat(md_array_a, md_array_b, md_array_a.shape().size() - 1);
}

std::vector<MultiDimArray> split(const MultiDimArray& md_array, Index axis) {
  auto& dims = md_array.shape();
  CHECK(dims.size() > 0);
  CHECK(axis < dims.size() && axis >= 0);
  Index block_nums = cal_block_num(dims, axis);
  Index block_size = cal_block_size(dims, axis);
  std::vector<Index> new_dims;
  new_dims.reserve(dims.size() - 1);
  for (Index i{0}; i < dims.size(); i++) {
    if (i != axis) {
      new_dims.push_back(dims[i]);
    }
  }
  std::vector<MultiDimArray> res;
  Index res_nums{dims[axis]};
  res.resize(res_nums);
  for (Index i{0}; i < res_nums; i++) {
    res[i] = MultiDimArray{new_dims, md_array.dtype()};
  }
  Index split_block_size = res_nums * block_size;
  if (md_array.dtype() == DataType::FLOAT32) {
    auto data_f = md_array.as<float>();
    for (Index i{0}; i < block_nums; i++) {
      for (Index j{0}; j < res_nums; j++) {
        auto data_f_t = res[j].as<float>();
        std::memcpy(data_f_t + i * block_size,
                    data_f + i * split_block_size + j * block_size,
                    block_size * sizeof(float));
      }
    }
    return res;
  }
  if (md_array.dtype() == DataType::INT32) {
    auto data_i = md_array.as<int32_t>();
    for (Index i{0}; i < block_nums; i++) {
      for (Index j{0}; j < res_nums; j++) {
        auto data_i_t = res[j].as<int32_t>();
        std::memcpy(data_i_t + i * block_size,
                    data_i + i * split_block_size + j * block_size,
                    block_size * sizeof(int32_t));
      }
    }
    return res;
  }
  CHECK_WITH_INFO(false, "reach unknown data type")
  return std::vector<MultiDimArray>{};
}
//slow
MultiDimArray broadcast_add(const MultiDimArray& md_array_a,
                            const MultiDimArray& md_array_b) {
  auto& dims_a = md_array_a.shape();
  auto& dims_b = md_array_b.shape();
  CHECK(dims_a.size() == dims_b.size());
  std::vector<Index> dims_max;
  dims_max.resize(dims_a.size());
  for (Index i{0}; i < dims_a.size(); i++) {
    if (dims_a[i] != dims_b[i]) {
      if (dims_a[i] == 1 || dims_b[i] == 1) {
        dims_max[i] = std::max(dims_a[i], dims_b[i]);
      } else {
        CHECK_WITH_INFO(false, "broadcast failed!!!");
      }
    } else {
      dims_max[i] = dims_a[i];
    }
  }
  std::vector<Index> a_bases = cal_block_sizes(dims_a);
  std::vector<Index> b_bases = cal_block_sizes(dims_b);
  std::vector<Index> res_bases = cal_block_sizes(dims_max);
  if (md_array_a.dtype() == DataType::FLOAT32 &&
      md_array_b.dtype() == DataType::FLOAT32) {
    MultiDimArray res{dims_max, DataType::FLOAT32};
    auto res_f = res.as<float>();
    auto data_a_f = md_array_a.as<float>();
    auto data_b_f = md_array_b.as<float>();
    std::vector<Index> indices;
    indices.resize(dims_a.size());
    for (Index i{0}; i < res.size(); i++) {
      Index index_temp{i};
      for (Index j{0}; j < dims_a.size(); j++) {
        indices[j] = index_temp / res_bases[j];
        index_temp = index_temp % res_bases[j];
      }
      // std::cout<<to_string(indices)<<"\n";
      *(res_f + cal_offset(res_bases, indices)) =
          *(data_a_f + cal_offset(a_bases, indices, dims_a)) +
          *(data_b_f + cal_offset(b_bases, indices, dims_b));
    }
    return res;
  }
  if (md_array_a.dtype() == DataType::INT32 &&
      md_array_b.dtype() == DataType::INT32) {
    MultiDimArray res{dims_max, DataType::INT32};
    auto res_i = res.as<int32_t>();
    auto data_a_i = md_array_a.as<int32_t>();
    auto data_b_i = md_array_b.as<int32_t>();
    std::vector<Index> indices;
    indices.resize(dims_a.size());

    for (Index i{0}; i < res.size(); i++) {
      for (Index j{0}; j < dims_a.size(); j++) {
        indices[j] = i % res_bases[j];
      }
      *(res_i + cal_offset(res_bases, indices)) =
          *(data_a_i + cal_offset(a_bases, indices, dims_a)) +
          *(data_b_i + cal_offset(b_bases, indices, dims_b));
    }
    return res;
  }
  CHECK_WITH_INFO(false, "reach unknown data type")
  return unkown();
}