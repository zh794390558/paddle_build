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


#include "decoder/pd_asr_model.h"

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cassert>

namespace ppspeech{

// load model and attrs
void PaddleAsrModel::Read(const std::string& model_path_w_prefix){
    paddle::jit::utils::InitKernelSignatureMap();

#ifdef USE_GPU
    dev_ = phi::GPUPlace();
#else
    dev_ = phi::CPUPlace();
#endif
    PaddleLayer model = paddle::jit::Load(model_path_w_prefix, dev_);
    model_ = std::make_shared<PaddleLayer>(std::move(model));

    subsampling_rate_ = model_->Attribute<int>("subsampling_rate");
    right_context_ = model_->Attribute<int>("right_context");

    sos_ = model_->Attribute<int>("sos_symbol");
    eos_ = model_->Attribute<int>("eos_symbol");
    is_bidecoder_ = false;

    forward_encoder_chunk_ = model_->Function("jit.forward_encoder_chunk");
    forward_attention_decoder_ = model_->Function("jit.forward_attention_decoder");

    std::cout << "Paddle Model Info: " << std::endl;
    std::cout << "\tsubsampling_rate " << subsampling_rate_<< std::endl;
    std::cout << "\tright context " << right_context_<< std::endl;
    std::cout << "\tsos " << sos_<< std::endl;
    std::cout << "\teos " << eos_<< std::endl;
    std::cout << "\tis bidecoder " << is_bidecoder_<< std::endl;
}

// shallow copy
PaddleAsrModel::PaddleAsrModel(const PaddleAsrModel& other){
  // copy meta
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidecoder_ = other.is_bidecoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  // copy model ptr
  model_ = other.model_;

  //ignore inner states
}

std::shared_ptr<AsrModelItf> PaddleAsrModel::Copy() const {
    auto asr_model = std::make_shared<PaddleAsrModel>(*this);
    // reset inner state for new decoding
    asr_model->Reset();
    return asr_model;
}

void PaddleAsrModel::Reset(){
    offset_ = 0;
    cached_feats_.clear();

    att_cache_ = std::move(paddle::full({0,0,0,0}, 0.0));
    cnn_cache_ = std::move(paddle::full({0,0,0,0}, 0.0));

    encoder_outs_.clear();
}

void PaddleAsrModel::ForwardEncoderChunkImpl(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob){
    // TODO: impl
    
}

float PaddleAsrModel::ComputePathScore(const paddle::Tensor& prob, const std::vector<int>& hyp, int eos){
    // sum `hyp` path scores in `prob`
    // prob (1, Umax, V)
    // hyp (U,)
    float score = 0.0f;
    std::vector<int64_t> dims = prob.shape();
    assert(dims.size() == 3);
    assert(dims[0] == 1);
    int vocab_dim = static_cast<int>(dims[1]);

    const float* prob_ptr = prob.data<float>();
    for (size_t i = 0; i < hyp.size(); ++i) {
        const float* row = prob_ptr + i*vocab_dim;
        // score += prob_ptr[i][hyp[i]];
        score += row[hyp[i]];
    }
    //  score += prob_ptr[hyp.size()][eos];
    const float* row = prob_ptr + hyp.size()*vocab_dim;
    score += row[eos];
    return score;
}

void PaddleAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps, float reverse_weight, std::vector<float>* rescoring_score){
    assert(rescoring_score != nullptr);

    int num_hyps = hyps.size();
    rescoring_score->resize(num_hyps, 0.0f);

    if (num_hyps == 0) return ;

    if (encoder_outs_.size() == 0){
        // no encoder outs
        std::cerr << "encoder_outs_.size() is zero. Please check it." << std::endl;
        return;
    }

    // prepare input
    paddle::Tensor hyps_lens = paddle::full({num_hyps}, 0, paddle::DataType::INT64);
    int64_t* hyps_len_ptr = hyps_lens.mutable_data<int64_t>();
    int max_hyps_len = 0;
    for (size_t i =0; i < num_hyps; ++i){
        int len = hyps[i].size() + 1; // eos
        max_hyps_len = std::max(max_hyps_len, len);
        hyps_len_ptr[i] = static_cast<int64_t>(len);
    }

    paddle::Tensor hyps_tensor = paddle::full({num_hyps, max_hyps_len}, 0, paddle::DataType::INT64);
    int64_t* hyps_ptr = hyps_tensor.mutable_data<int64_t>();
    for (size_t i = 0; i < num_hyps; ++i){
        const std::vector<int>& hyp = hyps[i];
        int64_t* row = hyps_ptr + max_hyps_len*i;
        row[0] = sos_;
        for (size_t j = 0; j < hyp.size(); ++j){
            row[j+1]  = hyp[j];
        }
    }
    // forward attention decoder by hyps and correspoinding encoder_outs_
    paddle::Tensor encoder_out = paddle::concat(encoder_outs_, 1);
    // auto encoder_out = paddle::full({1, 20, 512}, 1, paddle::DataType::FLOAT32, phi::CPUPlace());

    std::vector<paddle::experimental::Tensor> inputs{hyps_tensor, hyps_lens, encoder_out};
    std::vector<paddle::Tensor> outputs = (*forward_attention_decoder_)(inputs);
    std::cout << "dddd" << outputs[0].data<float>()[0] << std::endl;

    assert(outputs.size() == 1); // not support backward decoder

    // (B, Umax, V)
    paddle::Tensor probs = outputs[0];
    std::vector<int64_t> probs_shape = probs.shape();
    assert(probs_shape.size() == 3);
    assert(probs_shape[0] == num_hyps);
    assert(probs_shape[1] == max_hyps_len);

    // fake reverse probs
    assert(std::fabs(reverse_weight - 0.0f) < std::numeric_limits<float>::epsilon());
    paddle::Tensor r_probs = outputs[0];
    std::vector<int64_t> r_probs_shape = r_probs.shape();
    assert(r_probs_shape.size() == 3);
    assert(r_probs_shape[0] == num_hyps);
    assert(r_probs_shape[1] == max_hyps_len);

    // compute rescoring score
    using IntArray = paddle::experimental::IntArray;
    std::vector<paddle::Tensor> probs_v = paddle::split(probs, IntArray({num_hyps}), 0);
    std::vector<paddle::Tensor> r_probs_v = paddle::split(r_probs, IntArray({num_hyps}), 0);

    for (size_t i = 0; i < num_hyps; ++i){
        const std::vector<int>& hyp = hyps[i];

        // left-to-right decoder score
        float score = 0.0f;
        score = ComputePathScore(probs_v[i], hyp, eos_);

        // right-to-left decoder score
        float r_score = 0.0f;
        if (is_bidecoder_ && reverse_weight > 0){
            std::vector<int> r_hyp(hyp.size());
            std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
            r_score = ComputePathScore(r_probs_v[i], r_hyp, eos_);
        }

        // combinded left-to-right and right-to-lfet score
        (*rescoring_score)[i] = score * (1 - reverse_weight) + r_score * reverse_weight; 
    }

}

}   // namespace ppspeech
