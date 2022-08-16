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
#include <cstring>
#include <fstream>
#include <sstream>

#include "utils/log.h"

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

    // forward_encoder_chunk_ = model_->Function("jit.forward_encoder_chunk");
    // forward_attention_decoder_ = model_->Function("jit.forward_attention_decoder");
    // ctc_activation_ = model_->Function("jit.ctc_activation");
    forward_encoder_chunk_ = model_->Function("forward_encoder_chunk");
    forward_attention_decoder_ = model_->Function("forward_attention_decoder");
    ctc_activation_ = model_->Function("ctc_activation");
    // CHECK(forward_encoder_chunk_ != nullptr);
    // CHECK(forward_attention_decoder_ != nullptr);
    // CHECK(ctc_activation_ != nullptr);
    
    std::cout << "Paddle Model Info: " << std::endl;
    std::cout << "\tsubsampling_rate " << subsampling_rate_<< std::endl;
    std::cout << "\tright context " << right_context_<< std::endl;
    std::cout << "\tsos " << sos_<< std::endl;
    std::cout << "\teos " << eos_<< std::endl;
    std::cout << "\tis bidecoder " << is_bidecoder_<< std::endl;
}

// shallow copy
PaddleAsrModel::PaddleAsrModel(const PaddleAsrModel& other){
  forward_encoder_chunk_ = other.forward_encoder_chunk_;
  forward_attention_decoder_ = other.forward_attention_decoder_;
  ctc_activation_ = other.ctc_activation_;

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
    
    //1. splice cached_feature, and chunk_feats
    // First dimension is B, which is 1.
    int num_frames = cached_feats_.size() + chunk_feats.size();
    const int feature_dim = chunk_feats[0].size();

    VLOG(1)<< "num_frames: " << num_frames;
    VLOG(1)<< "feature_dim: " << feature_dim ;

    // feats (B=1,T,D)
    paddle::Tensor feats = paddle::full({1, num_frames, feature_dim}, 0.0f, paddle::DataType::FLOAT32);
    float* feats_ptr = feats.mutable_data<float>();

    for(size_t i = 0; i < cached_feats_.size(); ++i){
        float* row = feats_ptr + i*feature_dim;
        // for (int j = 0; j < feature_dim; ++j){
        //     row[j] = cached_feats_[i].data()[j];
        // }
        std::memcpy(row, cached_feats_[i].data(), feature_dim * sizeof(float));
    }
    for(size_t i = 0; i < chunk_feats.size(); ++i){
        float* row = feats_ptr + (cached_feats_.size() + i) * feature_dim;
        // for (int j =0; j < feature_dim; ++j){
        //     row[j] = chunk_feats[i].data()[j];
        // }
        std::memcpy(row, chunk_feats[i].data(), feature_dim * sizeof(float));
    }


    VLOG(1) << "feats shape: " << feats.shape()[0] << ", "  << feats.shape()[1] << ", " << feats.shape()[2]; 


    std::stringstream path("feat", std::ios_base::app | std::ios_base::out);
    path << offset_;
    std::ofstream feat_fobj(path.str().c_str(), std::ios::out);
    CHECK(feat_fobj.is_open());
    feat_fobj << feats.shape()[0] << " "  << feats.shape()[1] << " " << feats.shape()[2] << "\n";
    for (int i = 0; i < feats.numel(); i++){
        feat_fobj << feats_ptr[i] << " ";
        if ( (i+1) % feature_dim == 0){
            feat_fobj<< "\n";
        }
    }
    feat_fobj << "\n";



    // Endocer chunk forward
#ifdef USE_GPU
    feats = feats.copy_to(paddle::GPUPlace(), /*blocking*/ false );
    att_cache_ = att_cache_.copy_to(paddle::GPUPlace()), /*blocking*/ false;
    cnn_cache_ = cnn_cache_.copy_to(Paddle::GPUPlace(), /*blocking*/ false);
#endif

    int required_cache_size = num_left_chunks_ * chunk_size_; // -1 * 16
    // must be scalar, but paddle do not have scalar.
    paddle::Tensor offset = paddle::full({1}, offset_, paddle::DataType::INT32);
    // VLOG(1) << "offset shape: " << offset.shape()[0] ; 
    // VLOG(1) << "att_cache_ shape: " << att_cache_.shape()[0] << ", "  << att_cache_.shape()[1] << ", " << att_cache_.shape()[2] << ", " << cnn_cache_.shape()[3]; 
    // VLOG(1) << "cnn_cache_ shape: " << cnn_cache_.shape()[0] << ", "  << cnn_cache_.shape()[1] << ", " << cnn_cache_.shape()[2] << ", " << cnn_cache_.shape()[3]; 
    // freeze `required_cache_size` in graph, so not specific it in function call.
    std::vector<paddle::Tensor> inputs = {feats, offset, /*required_cache_size, */ att_cache_, cnn_cache_};
    VLOG(1) << "inputs size: " << inputs.size(); 
    std::vector<paddle::Tensor> outputs = forward_encoder_chunk_(inputs);
    VLOG(1) << "outputs size: " << outputs.size(); 
    CHECK(outputs.size() == 3);

#ifdef USE_GPU
    paddle::Tensor chunk_out = outputs[0].copy_to(paddle::CPUPlace());
    att_cache_ = outputs[1].copy_to(paddle::CPUPlace());
    cnn_cache_ = outputs[2].copy_to(paddle::CPUPlace());
#else
    paddle::Tensor chunk_out = outputs[0];
    att_cache_ = outputs[1];
    cnn_cache_ = outputs[2];
#endif

    // current offset in decoder frame
    offset_ += chunk_out.shape()[1];

#ifdef USE_GPU
#error "Not implementation."
#else
    // ctc_activation == log_softmax
    inputs.clear();
    outputs.clear();
    inputs = std::move(std::vector<paddle::Tensor>({chunk_out}));


    path.str("logits");
    path << offset_ - chunk_out.shape()[1];
    std::ofstream logits_fobj(path.str().c_str(), std::ios::out);
    CHECK(logits_fobj.is_open());
    logits_fobj << chunk_out.shape()[0] << " " <<  chunk_out.shape()[1] << " " << chunk_out.shape()[2]  << "\n";
    const float* chunk_out_ptr = chunk_out.data<float>();
    for (int i = 0; i < chunk_out.numel(); i++){
        logits_fobj << chunk_out_ptr[i] << " ";
        if ( (i+1) % chunk_out.shape()[2] == 0){
            logits_fobj<< "\n";
        }
    }
    logits_fobj << "\n";


    outputs = ctc_activation_(inputs);
    paddle::Tensor ctc_log_probs = outputs[0];


    path.str("logprob");
    path << offset_ - chunk_out.shape()[1];
  
    std::ofstream logprob_fobj(path.str().c_str(), std::ios::out);
    CHECK(logprob_fobj.is_open());
    logprob_fobj << ctc_log_probs.shape()[0] << " " <<  ctc_log_probs.shape()[1] << " " << ctc_log_probs.shape()[2]  << "\n";
    const float* logprob_ptr = ctc_log_probs.data<float>();
    for (int i = 0; i < ctc_log_probs.numel(); i++){
        logprob_fobj << logprob_ptr[i] << " ";
        if ( (i+1) % ctc_log_probs.shape()[2] == 0){
            logprob_fobj<< "\n";
        }
    }
    logprob_fobj << "\n";


    // collects encoder outs.
    encoder_outs_.push_back(std::move(chunk_out));
#endif

    // Copy to output, (B=1,T,D)
    std::vector<int64_t> ctc_log_probs_shape = ctc_log_probs.shape();
    int B = ctc_log_probs_shape[0];
    CHECK(B==1);
    int T = ctc_log_probs_shape[1];
    int D = ctc_log_probs_shape[2];

    float* ctc_log_probs_ptr = ctc_log_probs.data<float>();
    out_prob->resize(T);
    for (int i = 0; i < T; i++){
        (*out_prob)[i].resize(D);
        float* dst_ptr = (*out_prob)[i].data();
        float* src_ptr =  ctc_log_probs_ptr + (i*D);
        std::memcpy(dst_ptr, src_ptr, D * sizeof(float));
    }

    return;
}

// Debug api
void PaddleAsrModel::FeedEncoderOuts(paddle::Tensor& encoder_out){
    // encoder_out (T,D)
    encoder_outs_.clear();
    encoder_outs_.push_back(encoder_out);
}

float PaddleAsrModel::ComputePathScore(const paddle::Tensor& prob, const std::vector<int>& hyp, int eos){
    // sum `hyp` path scores in `prob`
    // prob (1, Umax, V)
    // hyp (U,)
    float score = 0.0f;
    std::vector<int64_t> dims = prob.shape();
    CHECK(dims.size() == 3);
    CHECK(dims[0] == 1);
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
    CHECK(rescoring_score != nullptr);

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

    std::vector<paddle::experimental::Tensor> inputs{hyps_tensor, hyps_lens, encoder_out};
    std::vector<paddle::Tensor> outputs = forward_attention_decoder_(inputs);
    CHECK(outputs.size() == 1); // not support backward decoder

#ifdef DEBUG
    float* fwd_ptr = outputs[0].data<float>();
    for (int i = 0; i < outputs[0].numel(); ++i){
         std::cout << fwd_ptr[i] << " ";
         if ((i+1)% 5 == 0){
             std::cout << std::endl;
         }
    }
    std::cout << std::endl;

    std::cout << "encoder_out  shape: "<< std::endl;
    for (auto d : encoder_out.shape()){
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "decoder output shape: "<< std::endl;
    for (auto d : outputs[0].shape()){
        std::cout << d << " ";
    }
    std::cout << std::endl;
#endif


    // (B, Umax, V)
    paddle::Tensor probs = outputs[0];
    std::vector<int64_t> probs_shape = probs.shape();
    CHECK(probs_shape.size() == 3);
    CHECK(probs_shape[0] == num_hyps);
    CHECK(probs_shape[1] == max_hyps_len);

    // fake reverse probs
    CHECK(std::fabs(reverse_weight - 0.0f) < std::numeric_limits<float>::epsilon());
    paddle::Tensor r_probs = outputs[0];
    std::vector<int64_t> r_probs_shape = r_probs.shape();
    CHECK(r_probs_shape.size() == 3);
    CHECK(r_probs_shape[0] == num_hyps);
    CHECK(r_probs_shape[1] == max_hyps_len);

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
