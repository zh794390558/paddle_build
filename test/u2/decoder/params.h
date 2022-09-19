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
#include <utility>
#include <vector>

#include "decoder/asr_decoder.h"
#include "decoder/pd_asr_model.h"
#include "frontend/feature_pipeline.h"

#include "utils/flags.h"
#include "utils/string.h"

DEFINE_int32(num_threads, 1, "num threads for ASR model");

// PaddleAsrModel flags
DEFINE_string(model_path, "", "paddle exported model path with suffix");

// OnnxAsrModel flags
DEFINE_string(onnx_dir, "", "directory where the onnx model is saved");

// FeaturePipelineConfig flags
DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(sample_rate, 16000, "sample rate for audio");
DEFINE_string(cmvn_path, "", "cmvn stats path.");
DEFINE_string(feature_pipeline_type,
              "kaldi",
              "using kaldi or graph feature pipeline. When graph mode, using "
              "FLAGS_model_path as feature model path");

// TLG fst
DEFINE_string(fst_path, "", "TLG fst path");

// DecodeOptions flags
DEFINE_int32(chunk_size, -1, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_double(ctc_weight,
              0.5,
              "ctc weight when combining ctc score and rescoring score");
DEFINE_double(rescoring_weight,
              1.0,
              "rescoring weight when combining ctc score and rescoring score");
DEFINE_double(reverse_weight,
              0.0,
              "used for bitransformer rescoring. it must be 0.0 if decoder is"
              "conventional transformer decoder, and only reverse_weight > 0.0"
              "dose the right to left decoder will be calculated and used");
DEFINE_int32(nbest, 10, "nbest for ctc wfst or prefix search");
// wfst
DEFINE_int32(max_active, 7000, "max active states in ctc wfst search");
DEFINE_int32(min_active, 200, "min active states in ctc wfst search");
DEFINE_double(beam, 16.0, "beam in ctc wfst search");
DEFINE_double(lattice_beam, 10.0, "lattice beam in ctc wfst search");
DEFINE_double(acoustic_scale, 1.0, "acoustic scale for ctc wfst search");
DEFINE_double(blank_skip_thresh,
              1.0,
              "blank skip thresh for ctc wfst search, 1.0 means no skip");
DEFINE_double(length_penalty,
              0.0,
              "length penalty ctc wfst search, will not"
              "apply on self-loop arc, for balancing the del/ins ratio, "
              "suggest set to -3.0");

// SymbolTable flags
DEFINE_string(dict_path,
              "",
              "dict symbol table path, required when LM is enabled");
DEFINE_string(unit_path,
              "",
              "e2e model unit symbol table, it is used in both "
              "with/without LM scenarios for context/timestamp");

// context flags
DEFINE_string(context_path, "", "context paht, is used to build context graph");
DEFINE_double(context_score, 3.0, "is used to rescore the decoded result");

// PostProcessOptions flags
DEFINE_int32(language_type,
             0,
             "remove spaces according to language type"
             "0 = kMandarinEnglish, "
             "1 = kIndoEuropean");
DEFINE_bool(lowercase, true, "lowercase final result if needed");

namespace ppspeech {

std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  auto feature_config =
      std::make_shared<FeaturePipelineConfig>(FLAGS_num_bins,
                                              FLAGS_sample_rate,
                                              FLAGS_cmvn_path,
                                              FLAGS_feature_pipeline_type,
                                              FLAGS_model_path);
  return feature_config;
}

std::shared_ptr<DecodeOptions> InitDecodeOptionsFromFlags() {
  auto decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = FLAGS_chunk_size;
  decode_config->num_left_chunks = FLAGS_num_left_chunks;
  decode_config->ctc_weight = FLAGS_ctc_weight;
  decode_config->reverse_weight = FLAGS_reverse_weight;
  decode_config->rescoring_weight = FLAGS_rescoring_weight;
  // ctc prefix beam search
  decode_config->ctc_prefix_search_opts.first_beam_size = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.second_beam_size = FLAGS_nbest;
  // ctc wfst
  // decode_config->ctc_wfst_search_opts.max_active = FLAGS_max_active;
  // decode_config->ctc_wfst_search_opts.min_active = FLAGS_min_active;
  // decode_config->ctc_wfst_search_opts.beam = FLAGS_beam;
  // decode_config->ctc_wfst_search_opts.lattice_beam = FLAGS_lattice_beam;
  // decode_config->ctc_wfst_search_opts.acoustic_scale = FLAGS_acoustic_scale;
  // decode_config->ctc_wfst_search_opts.blank_skip_thresh =
  //     FLAGS_blank_skip_thresh;
  // decode_config->ctc_wfst_search_opts.length_penalty = FLAGS_length_penalty;
  // decode_config->ctc_wfst_search_opts.nbest = FLAGS_nbest;

  return decode_config;
}

std::shared_ptr<DecodeResource> InitDecodeResourceFromFlags() {
  auto resource = std::make_shared<DecodeResource>();

  if (!FLAGS_onnx_dir.empty()) {
    LOG(FATAL) << "Not impl onnx.";
  } else {
    LOG(INFO) << "Reading paddle model " << FLAGS_model_path;
    CHECK(!FLAGS_model_path.empty());
    // PaddleAsrModel::InitEngineThreads(FLAGS_num_threads);
    auto model = std::make_shared<PaddleAsrModel>();
    model->Read(FLAGS_model_path);
    resource->model = model;
  }

  LOG(INFO) << "Reading unit table " << FLAGS_unit_path;
  auto unit_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_unit_path));
  CHECK(unit_table != nullptr);
  resource->unit_table = unit_table;

  if (!FLAGS_fst_path.empty()) {
    // with lm
    CHECK(!FLAGS_dict_path.empty());
    LOG(INFO) << "Reading fst " << FLAGS_fst_path;

    LOG(FATAL) << "not impl fst";

  } else {
    // w/o lm, symbol_table is the same as unit_table
    resource->symbol_table = unit_table;
  }

  // context
  if (!FLAGS_context_path.empty()) {
    LOG(INFO) << "Reading context " << FLAGS_context_path;
    std::vector<std::string> contexts;
    std::ifstream infile(FLAGS_context_path);
    std::string context;
    while (getline(infile, context)) {
      contexts.emplace_back(Trim(context));
    }
    // ContextConfig config;
    // config.context_score = FLAGS_context_score;
    // resource->context_graph = std::make_shared<CotextGraph>(config);
    // resource->context_graph->BuildContextGraph(contexts,
    //                                         resource->symbol_table);
  }

  // postprocess
  // PostProcessOptions post_process_opts;
  // post_process_opts.language_type =
  //     FLAGS_language_type == 0 ? kMandarinEnglish : kIndoEuropean;
  // post_process_opts.lowercase = FLAGS_lowercase;
  // resource->post_processor =
  //     std::make_shared<PostProcessor>(std::move(post_process_opts));

  return resource;
}

}  // namespace ppspeech
