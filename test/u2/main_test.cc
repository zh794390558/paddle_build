#include "utils/log.h"

#include "paddle/extension.h"
#include "paddle/jit/all.h"
#include "paddle/phi/api/all.h"

#ifdef USE_PROFILING
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/profiler.h"

using paddle::platform::EnableHostEventRecorder;
using paddle::platform::Profiler;
using paddle::platform::ProfilerOptions;
using paddle::platform::ProfilerResult;
using paddle::platform::RecordEvent;
using paddle::platform::RecordInstantEvent;
using paddle::platform::TracerEventType;
#endif

int main() {
  paddle::jit::utils::InitKernelSignatureMap();

#ifdef USE_PROFILING
  EnableHostEventRecorder();
  ProfilerOptions options;
  options.trace_level = 2;
  options.trace_switch = 3;
  auto profiler = Profiler::Create(options);
  profiler->Prepare();
  profiler->Start();
#endif

  // tensor op
  std::cout << "Run Start" << std::endl;
  auto d = paddle::Tensor("test");
  auto a = paddle::full({3, 4}, 2.0);
  auto b = paddle::full({4, 5}, 3.0);
  auto out = paddle::matmul(a, b);
  std::cout << "Run End" << std::endl;

  // load model
  auto layer =
      paddle::jit::Load("asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model/export.jit", phi::CPUPlace());

  auto hyps =
      paddle::full({10, 8}, 10, paddle::DataType::INT64, phi::CPUPlace());
  auto hyps_lens =
      paddle::full({10}, 8, paddle::DataType::INT64, phi::CPUPlace());
  auto encoder_out =
      paddle::full({1, 20, 512}, 1, paddle::DataType::FLOAT32, phi::CPUPlace());

  std::vector<paddle::experimental::Tensor> inputs{
      hyps, hyps_lens, encoder_out};

  std::vector<paddle::experimental::Tensor> outputs =
      layer.Function("forward_attention_decoder")(inputs);
  std::cout << "forward_attention_decoder has " << outputs.size() << " outputs."
            << std::endl;
  std::cout << "output 0 has " << outputs[0].size() << " elements."
            << std::endl;
  auto data = outputs[0].data<float>();

  std::cout << "dtat[0]: " << data[0] << std::endl;

  // get attribute
  std::cout << "eos_symbol: " << layer.Attribute<int>("eos_symbol")
            << std::endl;
  std::cout << "sos_symbol: " << layer.Attribute<int>("sos_symbol")
            << std::endl;
  std::cout << "right_context: " << layer.Attribute<int>("right_context")
            << std::endl;
  std::cout << "subsampling_rate: " << layer.Attribute<int>("subsampling_rate")
            << std::endl;

  std::vector<paddle::Tensor> encoder_out_v = {
      paddle::full({1, 6, 5}, 3, paddle::DataType::FLOAT32),
      paddle::full({1, 6, 5}, 2, paddle::DataType::FLOAT32),
      paddle::full({1, 6, 5}, 1, paddle::DataType::FLOAT32),
  };
  out = paddle::concat(encoder_out_v, 1);

  float* out_ptr = out.data<float>();
  for (int i = 0; i < out.numel(); i++) {
    std::cout << out_ptr[i] << " ";
  }
  std::cout << std::endl;


 {
   std::vector<paddle::Tensor> encoder_outs;
   
    int feature_dim = 80;
    int frame_num = 16 * 4 + 3;  // chunk_size * downsample_rate +
                                 // (receptive_field - downsample_rate)
    paddle::Tensor feats = paddle::full(
        {1, frame_num, feature_dim}, 0.12f, paddle::DataType::FLOAT32);
    paddle::Tensor offset = paddle::zeros({1}, paddle::DataType::INT32);
    paddle::Tensor att_cache = paddle::zeros({0,0,0,0}, paddle::DataType::FLOAT32);
    paddle::Tensor cnn_cache = paddle::zeros({0,0,0,0}, paddle::DataType::FLOAT32);
    std::vector<paddle::Tensor> inputs = {
        feats, offset, /*required_cache_size, */ att_cache, cnn_cache};


    std::vector<paddle::Tensor> outputs = layer.Function("forward_encoder_chunk")(inputs);
    encoder_outs.push_back(outputs[0]);

    const float* out_ptr = outputs[0].data<float>();
    std::cout << out_ptr << std::endl;
    for (int i = 0; i < outputs[0].numel(); i++) {
      std::cout << out_ptr[i] << " ";
    }
    std::cout << std::endl;
      std::cout << std::endl;
        std::cout << std::endl;



    feats = paddle::full(
        {1, 67, 80}, 0.32f, paddle::DataType::FLOAT32);
    inputs = {
        feats, offset, /*required_cache_size, */ att_cache, cnn_cache};
    outputs = layer.Function("forward_encoder_chunk")(inputs);

    out_ptr = outputs[0].data<float>();
    std::cout << out_ptr << std::endl;
    for (int i = 0; i < outputs[0].numel(); i++) {
      std::cout << out_ptr[i] << " ";
    }
    std::cout << std::endl;
      std::cout << std::endl;
        std::cout << std::endl;


    auto encoder = paddle::concat(encoder_outs, 1);

    std::cout << std::endl;
    out_ptr = encoder.data<float>();
    std::cout << out_ptr << std::endl;
    for (int i = 0; i < encoder.numel(); i++) {
	    std::cout << out_ptr[i] << " ";
	    if ( (i+1) == outputs[0].numel()){
		    std::cout << std::endl;
		    std::cout << std::endl;
	    }
    }
 }



#ifdef USE_PROFILING
  auto profiler_result = profiler->Stop();
  profiler_result->Save("main.test.prof");
#endif
}
