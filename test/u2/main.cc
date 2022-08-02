#include "core/pd_asr_model.h"

int main(void){
    ppspeech::PaddleAsrModel model;

    model.Read("chunk_wenetspeech_static/export.jit");
    
    return 0;
}