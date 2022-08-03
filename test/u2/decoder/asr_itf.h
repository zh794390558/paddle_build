// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace ppspeech {

class AsrModelItf {
public:
    virtual int context() const {return right_context_ + 1;}
    virtual int right_context() const {return right_context_;}
    virtual int subsampling_rate() const {return subsampling_rate_; }
    virtual int eos() const { return eos_; }
    virtual int sos() const { return sos_; }
    virtual int is_bidecoder() const {return is_bidecoder_;}
    virtual int offset() const {return offset_;}

    virtual void set_chunk_size(int chunk_size) {
        chunk_size_ = chunk_size;
    }

    virtual void set_num_left_chunks(int num_left_chunks){
        num_left_chunks_ = num_left_chunks;
    }

    // start: false, it is the start chunk of one sentence, else true
    virtual int num_frames_for_chunk(bool start) const;

    virtual void Reset() = 0;

    virtual void ForwardEncoderChunk(
        const std::vector<std::vector<float>>& chunk_feats,
        std::vector<std::vector<float>>* ctc_probs);

    virtual void AttentionRescoring(
        const std::vector<std::vector<int>>& hyps,
        float reverse_weight,
        std::vector<float>* rescoring_score) = 0;

    virtual std::shared_ptr<AsrModelItf> Copy() const= 0;

protected:
    virtual void ForwardEncoderChunkImpl(
        const std::vector<std::vector<float>>& chunk_feats,
        std::vector<std::vector<float>>* ctc_probs) = 0;

    virtual void CacheFeature(const std::vector<std::vector<float>>& chunk_feats);

protected:
    // model specification
    int right_context_ = 1; 
    int subsampling_rate_ = 1;
    
    int sos_ = 0;
    int eos_ = 0;

    bool is_bidecoder_ = false;

    int chunk_size_ = 16; // num of decoder frames. If chunk_size > 0, streaming case. Otherwise, none streaming case
    int num_left_chunks_ = -1; // -1 means all left chunks

    // asr decoder state
    int offset_ = 0; // current offset in encoder output time stamp. Used by position embedding.
    std::vector<std::vector<float>> cached_feats_; // features cache
};

} // namespace ppspeech