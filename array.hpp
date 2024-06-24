#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "check.hpp"
using Index = int64_t;
enum class DataType {
  UNKNOWN,
  FLOAT32,
  INT32,
};
inline Index get_data_type_size(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::INT32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}
class MultiDimArray {
 public:
  MultiDimArray() {}
  explicit MultiDimArray(const std::vector<Index>& dims, DataType data_type) {
    data_type_ = data_type;
    dims_ = dims;
    for (auto& d : dims) {
      CHECK(d > 0);
    }
    size_ = 1;
    for (auto& dim : dims) {
      size_ *= dim;
    }
    data_ = std::make_shared<std::vector<uint8_t>>();
    data_->resize(size_ * sizeof(get_data_type_size(data_type)));
  }
  Index ndim() const { return dims_.size(); }
  const std::vector<Index>& shape() const { return dims_; }
  Index size() const { return size_; }
  DataType dtype() const { return data_type_; }
  template <typename T>
  T* as() const {
    return reinterpret_cast<T*>(data_->data());
  }

  void set_new_dims(const std::vector<Index>& dims) {
    Index new_size{1};
    for (auto& dim : dims) {
      new_size *= dim;
    }
    CHECK_WITH_INFO(new_size == size_, "dims mismatch")
    dims_ = dims;
  }

 private:
  std::shared_ptr<std::vector<uint8_t>> data_;
  Index capacity_ = 0;
  Index size_ = 0;
  std::vector<Index> dims_;
  DataType data_type_{DataType::UNKNOWN};
};