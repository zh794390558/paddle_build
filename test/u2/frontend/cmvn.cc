
#include <fstream>
#include <vector>

#include "boost/json/src.hpp"
// #include "boost/json.hpp"
#include "cmvn.h"

namespace ppspeech {

Cmvn::Cmvn(const std::string& cmvn_path) {
  std::ifstream in(cmvn_path);
  CHECK(in.is_open());
  VLOG(1) << "Load cmvn: " << cmvn_path;
  std::string json_str((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  boost::json::value value = boost::json::parse(json_str);

  if (!value.is_object()) {
    LOG(ERROR) << "Input json file format error.";
  }

  for (auto obj : value.as_object()) {
    if (obj.key() == "mean_stat") {
      VLOG(2) << "mean_stat: " << obj.value();
    }
    if (obj.key() == "var_stat") {
      VLOG(2) << "var_stat: " << obj.value();
    }
    if (obj.key() == "frame_num") {
      VLOG(2) << "frame_num: " << obj.value();
    }
  }

  frame_num_ = value.at("frame_num").as_int64();
  VLOG(2) << "nframe: " << frame_num_;

  boost::json::array mean_stat = value.at("mean_stat").as_array();
  for (auto it = mean_stat.begin(); it != mean_stat.end(); it++) {
    // compute mean
    mean_.push_back(it->as_double() / frame_num_);
  }

  boost::json::array var_stat = value.at("var_stat").as_array();
  for (boost::json::array::iterator it = var_stat.begin(); it != var_stat.end();
       it++) {
    int i = std::distance(var_stat.begin(), it);
    double var = it->as_double() / frame_num_ - mean_[i] * mean_[i];
    double floor = 1.0e-20;
    if (var < floor) {
      LOG(WARNING) << "Flooring cepstral variance from " << var << " to "
                   << floor;
      var = floor;
    }
    double scale = 1.0 / sqrt(var);
    if (scale != scale || 1 / scale == 0.0)
      LOG(ERROR) << "NaN or infinity in cepstral mean/variance computation";

    var_inv_.push_back(scale);
  }
}

// Compute cmvn, return num frames
int Cmvn::Compute(std::vector<std::vector<float>>& feats) {
  int nframe = feats.size();
  CHECK(nframe > 0);
  int feat_dim = feats[0].size();

  for (int i = 0; i < nframe; i++) {
    for (int j = 0; j < feat_dim; j++) {
      feats[i][j] = float(feats[i][j] - mean_[j]) * var_inv_[j];
    }
  }

  return nframe;
}

}  // namespace ppspeech