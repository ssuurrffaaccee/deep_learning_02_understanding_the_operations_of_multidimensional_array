#include <iostream>

#include "funcs.hpp"
int main() {
  try {
    {
      std::cout << "\ntest: zeros\n";
      auto x = zeros({2, 3, 4}, DataType::FLOAT32);
      auto y = index(x, {0, 1, 2});
      std::cout << to_flat_string(x) << "\n";
      std::cout << to_flat_string(y) << "\n";
    }
    {
      std::cout << "\ntest: basic_index\n";
      auto z = arange(1, 10, 1);
      reshape(z, {3, 3});
      auto y = index(z, {1, 2});
      std::cout << to_flat_string(y) << "\n";
    }
    {
      std::cout << "\ntest: reshape and concat\n";
      auto z = arange(1, 10, 1);
      reshape(z, {3, 3});
      std::cout << to_flat_string(z) << "\n";
      std::cout << "concat axis=0\n";
      auto m = concat(z, z, 0);
      std::cout << to_flat_string(m) << "\n";
      std::cout << "concat axis=1\n";
      auto n = concat(z, z, 1);
      std::cout << to_flat_string(n) << "\n";
    }
    {
      std::cout << "\ntest: split\n";
      auto z = arange(1, 10, 1);
      reshape(z, {3, 3});
      {
        std::cout << "split axis=0\n";
        auto vs = split(z, 0);
        for (auto &v : vs) {
          std::cout << to_flat_string(v) << "\n";
        }
      }
      {
        std::cout << "split axis=1\n";
        auto vs = split(z, 1);
        for (auto &v : vs) {
          std::cout << to_flat_string(v) << "\n";
        }
      }
    }
    {
      std::cout << "\ntest: broadcast_add\n";
      auto a = arange(1, 4, 1);
      reshape(a, {3, 1});
      std::cout << to_flat_string(a) << "\n";
      auto b = arange(1, 10, 1);
      reshape(b, {3, 3});
      std::cout << to_flat_string(b) << "\n";
      auto c = broadcast_add(a, b);
      std::cout << to_flat_string(c) << "\n";
      reshape(a, {1, 3});
      auto d = broadcast_add(a, b);
      std::cout << to_flat_string(d) << "\n";
    }
  } catch (MDArrayExceptoin &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}