// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)

#include "core/asr_itf.h"

#include <limits>


namespace ppspeech{

int AsrModelItf::num_frames_for_chunk(bool start) const {
    int num_needed_frames = 0; // num feat frames
    bool first = !start; // start == false is first

    if (chunk_size_ > 0){
        // streaming mode
        if (first){ 
            // first chunk
            int context = this->context(); // 1 decoder frame need `context` feat frames
            num_needed_frames = (chunk_size_ - 1) * subsampling_rate_ + context;
        } else {
            // after first chunk, we need stride this num frames.
            num_needed_frames = chunk_size_ * subsampling_rate_;
        }
    } else {
        // non-streaming mode
        num_needed_frames = std::numeric_limits<int>::max();
    }

    return num_needed_frames;
}

// cache feats for next chunk
void AsrModelItf::CacheFeature(const std::vector<std::vector<float>>& chunk_feats){
    const int chunk_size = chunk_feats.size();
    const int cached_feat_size = this->context() - subsampling_rate_;
    if (chunk_feats.size() >= cached_feat_size){
        cached_feats_.resize(cached_feat_size);
        for (int i = 0; i < cached_feat_size; ++i){
            cached_feats_[i] = chunk_feats[chunk_size - cached_feat_size + i];
        }
    }
}

void AsrModelItf::ForwardEncoderChunk(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* ctc_probs){
        ctc_probs->clear();
        int num_frames = cached_feats_.size() + chunk_feats.size();
        if (num_frames >= this->context()){
            this->ForwardEncoderChunkImpl(chunk_feats, ctc_probs);
            this->CacheFeature(chunk_feats);
        }
}   

} // namespace ppspeech