// Copyright (c) 2017 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frontend/feature_pipeline.h"
#include <algorithm>
#include <utility>

#ifdef USE_PROFILING
#include "paddle/fluid/platform/profiler.h"
using paddle::platform::TracerEventType;
using paddle::platform::RecordEvent;
#endif

namespace ppspeech {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config)
    : config_(config),
      feature_dim_(config.num_bins),
      num_frames_(0),
      input_finished_(false) {
        config_.Info();
        if (config_.pipeline_type == "graph"){
            // force feature pipeline on cpu
            auto dev = phi::CPUPlace();
            PaddleLayer model = paddle::jit::Load(config_.model_path_w_prefix, dev);
            model_ = std::make_shared<PaddleLayer>(std::move(model));
            feature_pipeline_func_ = model_->Function("forward_feature");
            CHECK(feature_pipeline_func_.IsValid());
        } else {
            fbank_ = std::move(std::make_shared<Fbank>(config_.num_bins, config_.sample_rate, config_.frame_length, config_.frame_shift));
            cmvn_ = std::move(std::make_shared<Cmvn>(config_.cmvn_path));
        }
      }

void FeaturePipeline::AcceptWaveform(const float* pcm, const int& size) {
#ifdef USE_PROFILING
    RecordEvent event("AcceptWaveform", TracerEventType::UserDefined, 1);
#endif

  std::vector<std::vector<float>> feats;

  // add wave cache
  std::vector<float> waves;
  waves.insert(waves.begin(), remained_wav_.begin(), remained_wav_.end());
  waves.insert(waves.end(), pcm, pcm + size);

  // compute feature
  int num_frames;
  if (config_.pipeline_type == "kaldi"){
      num_frames = fbank_->Compute(waves, &feats);
      num_frames = cmvn_->Compute(feats);
  } else if (config_.pipeline_type == "graph"){
      // waves to tensor
      size_t size = waves.size();
      paddle::Tensor audio = paddle::zeros({size}, paddle::DataType::FLOAT32);
      std::memcpy(audio.data<float>(), waves.data(), size * sizeof(float));
      paddle::Tensor audio_int16 = paddle::experimental::cast(audio, paddle::DataType::INT16);
      std::vector<paddle::Tensor> inputs{audio_int16};
      std::vector<paddle::Tensor> outputs = feature_pipeline_func_(inputs);
      paddle::Tensor t_feats = outputs[0];

      // (T, D)
      std::vector<int64_t> shape = t_feats.shape();
      CHECK(shape.size() == 2);
      num_frames = shape[0];
      int feat_dim = shape[1];

      CHECK(feat_dim == feature_dim_);
      const float* feats_ptr = t_feats.data<float>();

      feats.resize(num_frames);
      for (int i = 0; i < num_frames; i ++) {
        feats[i].resize(feat_dim);
       
        std::memcpy(feats[i].data(), feats_ptr, feat_dim * sizeof(float));
        
        feats_ptr += feat_dim;
      }
  } else {
    CHECK(false);
  }

  feature_queue_.Push(std::move(feats));
  num_frames_ += num_frames;

  // update wave cache 
  int left_samples = waves.size() - config_.frame_shift * num_frames;
  remained_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames,
            waves.end(),
            remained_wav_.begin());
  // we are still adding wave, notify input is not finished
  finish_condition_.notify_one();
}

void FeaturePipeline::AcceptWaveform(const int16_t* pcm, const int& size) {
  auto* float_pcm = new float[size];
  for (size_t i = 0; i < size; i++) {
    // cast int16 to float
    float_pcm[i] = static_cast<float>(pcm[i]);
  }
  this->AcceptWaveform(float_pcm, size);
  delete[] float_pcm;
}

void FeaturePipeline::SetInputFinished() {
  CHECK(!input_finished_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_finished_ = true;
  }
  finish_condition_.notify_one();
}

bool FeaturePipeline::ReadOne(std::vector<float>* feat) {
  if (!feature_queue_.Empty()) {
    *feat = std::move(feature_queue_.Pop());
    return true;
  } else {
    // queue empty
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      if (!feature_queue_.Empty()) {
        *feat = std::move(feature_queue_.Pop());
        return true;
      }
    }
    CHECK(input_finished_);
    // Double check queue.empty, see issue#893 for detailed discussions.
    if (!feature_queue_.Empty()) {
      *feat = std::move(feature_queue_.Pop());
      return true;
    } else {
      return false;
    }
  }
}

bool FeaturePipeline::Read(int num_frames,
                           std::vector<std::vector<float>>* feats) {
  feats->clear();
  if (feature_queue_.Size() >= num_frames) {
    *feats = std::move(feature_queue_.Pop(num_frames));
    return true;
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      if (feature_queue_.Size() >= num_frames) {
        *feats = std::move(feature_queue_.Pop(num_frames));
        return true;
      }
    }
    CHECK(input_finished_);
    // Double check queue.empty, see issue#893 for detailed discussions.
    if (feature_queue_.Size() >= num_frames) {
      *feats = std::move(feature_queue_.Pop(num_frames));
      return true;
    } else {
      *feats = std::move(feature_queue_.Pop(feature_queue_.Size()));
      return false;
    }
  }
}

void FeaturePipeline::Reset() {
  input_finished_ = false;
  num_frames_ = 0;
  remained_wav_.clear();
  feature_queue_.Clear();
}

}  // namespace ppspeech