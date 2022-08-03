#include "decoder/pd_asr_model.h"

int main(void){
    ppspeech::PaddleAsrModel model;
    model.Read("chunk_wenetspeech_static/export.jit");


    // auto encoder_out = paddle::full({1, 20, 512}, 1, paddle::DataType::FLOAT32, phi::CPUPlace());

    std::vector<float> scores;
    std::vector<std::vector<int>> hyps(10, std::vector<int>(8, 10));
    model.AttentionRescoring(hyps, 0, &scores);

    std::cout << scores[0] << std::endl;
    return 0;
}