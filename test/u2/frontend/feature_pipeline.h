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

#pragma once

#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "frontend/fbank.h"
#include "frontend/cmvn.h"
#include "utils/block_queue.h"
#include "utils/log.h"

namespace ppspeech {

struct FeaturePipelineConfig{
    int num_bins;       // 80 dim fbank
    int sample_rate;    // 16k 
    int frame_length;   // points in 25ms
    int frame_shift;    // points in 10ms
    std::string cmvn_path;      // cmvn path
    
    FeaturePipelineConfig(int num_bins, int sample_rate, const std::string& cmvn_path) 
    : num_bins(num_bins), sample_rate(sample_rate), cmvn_path(cmvn_path) {
        frame_length = sample_rate / 1000 * 25;
        frame_shift = sample_rate / 1000 * 10;
    }

    void Info() const {
        std::cout << "feature pipeline config"
        << " num_bins " << num_bins << " frame_length " << frame_length
        << " frame_shift " << frame_shift;
    }
};

// Typically, FeaturePipeline is used in two threads: one thread A calls
// AcceptWaveform() to add raw wav data and set_input_finished() to notice
// the end of input wav, another thread B (decoder thread) calls Read() to
// consume features. So a BlockingQueue is used to make this thread safe.

// The Read() is designed as a blocking method when there is no feature
// in feature_queue_ and the input is not finished.

class FeaturePipeline {
public:
    explicit FeaturePipeline(const FeaturePipelineConfig& config);

    // The feature extraction is done in AcceptWaveform().
    void AcceptWaveform(const float* pcm, const int& size);
    void AcceptWaveform(const int16_t* pcm, const int& size);

    // Current extracted frames number.
    int num_frames() const { return num_frames_; }
    int feature_dim() const { return feature_dim_; }
    const FeaturePipelineConfig& config() const { return config_; }

    // The caller should call thie method when speech input is end.
    // Never call AcceptWaveform() after calling SetInputFinished()
    void SetInputFinished();
    bool input_finished() const {return input_finished_; }

    // Return False if input is finished and no feature could be read.
    // Return True if a feature is read.
    // This function is a blocking method. It will block the thread when
    // there is no feature in feature_queue_ and the input is not finished.
    bool ReadOne(std::vector<float>* feat);

    // Read #num_frames frame features.
    // Return False if less than #num_frames features are read and the
    // input is finished.
    // Return True if #num_frames features are read.
    // This function is a blocking method when there is no feature
    // in feature_queue_ and the input is not finished.
    bool Read(int num_frames, std::vector<std::vector<float>>* feats);

    void Reset();
    bool IsLastFrame(int frame) const {
        return input_finished_ && (frame == num_frames_ -1);
    }

    int NumQueuedFrames() const { return feature_queue_.Size(); }

private:
    const FeaturePipelineConfig& config_;
    int feature_dim_;
    Fbank fbank_;
    Cmvn cmvn_;

    BlockingQueue<std::vector<float>> feature_queue_;
    int num_frames_;
    bool input_finished_;

    // The feature extraction is done in AcceptWaveform().
    // This waveform sample points are consumed by frame size.
    // The residual waveform sample points after framing are
    // kept to be used in next AcceptWaveform() calling.
    std::vector<float> remained_wav_;

    // Used to block the Read when there is no feature in feature_queue_
    // and the input is not finished.
    mutable std::mutex mutex_;
    std::condition_variable finish_condition_;

};


} // namespace ppspeech