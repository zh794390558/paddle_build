

// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/search_itf.h"
#include "utils/utils.h"

namespace ppspeech {

class ContextGraph;

struct CtcPrefixBeamSearchOptions {
  int blank = 0;
  int first_beam_size = 10;
  int second_beam_size = 10;
};

struct PrefixScore {
  // decoding, unit in log scale
  float b = -kFloatMax;   // blank ending score
  float nb = -kFloatMax;  // none blank ending score

  // timestamp, unit in log scale
  float v_b = -kFloatMax;             // viterbi blank ending score
  float v_nb = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_b;           // times of viterbi blank path
  std::vector<int> times_nb;          // times of viterbi none blank path

  // sum
  float score() const { return LogSumExp(b, nb); }

  // max
  float viterbi_score() const { return v_b > v_nb ? v_b : v_nb; }

  const std::vector<int>& times() const {
    return v_b > v_nb ? times_b : times_nb;
  }

  // context state
  bool has_context = false;
  int context_state = 0;
  float context_score = 0;
  std::vector<int> start_boundaries;
  std::vector<int> end_boundaries;

  void CopyContext(const PrefixScore& prefix_score) {
    context_state = prefix_score.context_state;
    context_score = prefix_score.context_score;
    start_boundaries = prefix_score.start_boundaries;
    end_boundaries = prefix_score.end_boundaries;
  }

  void UpdateContext(const std::shared_ptr<ContextGraph>& constext_graph,
                     const PrefixScore& prefix_score,
                     int word_id,
                     int prefix_len) {
    // TODO
  }

  float total_score() const { return score() + context_score; }
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // KB&DR has code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

using PrefixWithScoreType = std::pair<std::vector<int>, PrefixScore>;

class CtcPrefixBeamSearch : public SearchInterface {
 public:
  explicit CtcPrefixBeamSearch(
      const CtcPrefixBeamSearchOptions& opts,
      const std::shared_ptr<ContextGraph>& context_graph = nullptr);

  void Search(const std::vector<std::vector<float>>& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::kPrefixBeamSearch; }

  void UpdateOutputs(const std::pair<std::vector<int>, PrefixScore>& prefix);
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, PrefixScore>>& prefix);
  void UpdateFinalContext();

  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }

  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }

  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }

  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  const CtcPrefixBeamSearchOptions& opts_;
  int abs_time_step_ = 0;

  // n-best list and corresponding likelihood, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::shared_ptr<ContextGraph> context_graph_ = nullptr;

  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;

 public:
  DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace ppspeech