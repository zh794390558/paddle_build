#include <algorithm>
#include "decoder/pd_asr_model.h"

int main(void) {
  ppspeech::PaddleAsrModel model;
  model.Read("asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model/export.jit");

  std::vector<std::vector<float>> chunk_feats;  // [T,D=80]
  std::vector<std::vector<float>> out_prob;

  int T = 7;
  int D = 80;
  chunk_feats.resize(T);
  for (int i = 0; i < T; ++i) {
    chunk_feats[i].resize(D);
    std::fill(chunk_feats[i].begin(), chunk_feats[i].end(), 0.1);
  }

  model.ForwardEncoderChunkImpl(chunk_feats, &out_prob);
  std::cout << "T: " << out_prob.size() << std::endl;
  std::cout << "D: " << out_prob[0].size() << std::endl;

  for (int i = 0; i < out_prob[0].size(); i++) {
    std::cout << out_prob[0][i] << " ";
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  std::cout << "==============================" << std::endl;

  std::vector<float> scores;
  std::vector<std::vector<int>> hyps(1, std::vector<int>(8, 10));
  paddle::Tensor encoder_out =
      paddle::full({20, 512}, 1.0, paddle::DataType::FLOAT32);
  model.FeedEncoderOuts(encoder_out);

  model.AttentionRescoring(hyps, 0, &scores);
  std::cout << "att rescore: " << scores.size() << " " << scores[0]
            << std::endl;

  return 0;
}