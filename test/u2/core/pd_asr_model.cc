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


#include "core/pd_asr_model.h"

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace ppspeech{

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

    std::cout << "Paddle Model Info: ";
    std::cout << "\tsubsampling_rate " << subsampling_rate_;
    std::cout << "\tright context " << right_context_;
    std::cout << "\tsos " << sos_;
    std::cout << "\teos " << eos_;
    std::cout << "\tis bidecoder " << is_bidecoder_;
}

}   // namespace ppspeech
