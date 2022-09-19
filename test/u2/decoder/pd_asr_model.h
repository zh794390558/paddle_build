// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "decoder/asr_itf.h"

#include "paddle/extension.h"
#include "paddle/jit/all.h"
#include "paddle/phi/api/all.h"

namespace ppspeech {

class PaddleAsrModel : public AsrModelItf {
 public:
  using PaddleLayer = paddle::jit::Layer;
  PaddleAsrModel() = default;
  PaddleAsrModel(const PaddleAsrModel& other);

  void Read(const std::string& model_path_w_prefix);

  std::shared_ptr<PaddleLayer> paddle_model() const { return model_; }

  void Reset() override;

  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;

  std::shared_ptr<AsrModelItf> Copy() const override;

  // debug
  void FeedEncoderOuts(paddle::Tensor& encoder_out);

  // protected:
 public:
  void ForwardEncoderChunkImpl(
      const std::vector<std::vector<float>>& chunk_feats,
      std::vector<std::vector<float>>* ctc_probs) override;

  float ComputePathScore(const paddle::Tensor& prob,
                         const std::vector<int>& hyp,
                         int eos);

  void Warmup();

 private:
  phi::Place dev_;
  std::shared_ptr<PaddleLayer> model_ = nullptr;
  std::vector<paddle::Tensor> encoder_outs_{};
  // transformer/conformer attention cache
  paddle::Tensor att_cache_ = paddle::full({0, 0, 0, 0}, 0.0);
  // conformer-only conv_module cache
  paddle::Tensor cnn_cache_ = paddle::full({0, 0, 0, 0}, 0.0);

  paddle::jit::Function forward_encoder_chunk_;
  paddle::jit::Function forward_attention_decoder_;
  paddle::jit::Function ctc_activation_;
};

}  // namespace ppspeech