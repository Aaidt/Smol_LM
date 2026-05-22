#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

std::vector<int> encode(std::string s, std::map<char, int> &char_to_idx) {
   std::vector<int> vec;
   for (const auto &c : s) {
      vec.push_back(char_to_idx[c]);
   }

   return vec;
}

std::string decode(std::vector<int> vec, std::map<int, char> &idx_to_char) {
   std::string decoded_str;
   for (const auto &x : vec) {
      decoded_str += idx_to_char[x];
   }

   return decoded_str;
}

class Tensor {
 private:
   std::vector<int> data;

 public:
   Tensor() {}

   Tensor(const std::vector<int> &values) : data(values) {}

   void print() const {
      std::cout << "[ ";
      for (const int &x : data) {
         std::cout << x << " ";
      }
      std::cout << "]\n";
   }

   int size() const { return data.size(); }

   int operator[](int idx) const { return data[idx]; }
};

namespace torch {
Tensor randint(int max, int size) {
   std::mt19937 rng(std::random_device{}());

   std::uniform_int_distribution<int> dist(0, max - 1);

   std::vector<int> result;

   for (int i = 0; i < size; i++) {
      result.push_back(dist(rng));
   }

   return Tensor(result);
}
} // namespace torch

std::pair<std::vector<int>, std::vector<int>>
get_batch(const std::string &split, int batch_size, int context_length,
          const std::vector<int> &train_data,
          const std::vector<int> &test_data) {

   const std::vector<int> &data = (split == "train") ? train_data : test_data;

   Tensor ix = torch::randint(data.size() - context_length, batch_size);
}

int main() {
   std::ifstream inputFile("shakespeare.txt");

   if (!inputFile.is_open()) {
      std::cerr << "Error: Cannot open file" << std::endl;
      return 1;
   }

   std::string content((std::istreambuf_iterator<char>(inputFile)),
                       std::istreambuf_iterator<char>());

   std::set<char> vocab(content.begin(), content.end());
   std::cout << "Vocab size: " << vocab.size() << "\n";

   std::map<int, char> idx_to_char;
   std::map<char, int> char_to_idx;

   int idx = 0;
   for (const char &c : vocab) {
      idx_to_char[idx] = c;
      char_to_idx[c] = idx;
      idx++;
   }

   // std::vector<int> res = encode("Hello", char_to_idx);
   // std::cout << decode(res, idx_to_char) << "\n";

   std::vector<int> encoded_data = encode(content, char_to_idx);

   std::vector<int> train_data;
   std::vector<int> test_data;

   float n = 0.9 * encoded_data.size();
   for (int i = 0; i < n; i++) {
      train_data.push_back(encoded_data[i]);
   }
   for (int i = n; i < encoded_data.size(); i++) {
      test_data.push_back(encoded_data[i]);
   }

   inputFile.close();

   return 0;
}
