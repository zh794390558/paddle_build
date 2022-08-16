#pragma once


#include <vector>
#include <fstream>

// #include "boost/json/src.hpp"
// #include "boost/json.hpp"

#include "utils/log.h"

namespace ppspeech {

class Cmvn {
 public:
  Cmvn(const std::string& cmvn_path);
   
  // Compute cmvn, return num frames
  int Compute(std::vector<std::vector<float>>& feats);

 private:
  std::vector<float> mean_;
  std::vector<float> var_inv_;
  uint64_t frame_num_;

};

} // namespace ppspeech