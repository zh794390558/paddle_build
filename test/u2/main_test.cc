#include "utils/log.h"

#include "paddle/extension.h"
#include "paddle/jit/all.h"
#include "paddle/phi/api/all.h"


#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/profiler.h"

using paddle::platform::Profiler;
using paddle::platform::ProfilerOptions;
using paddle::platform::ProfilerResult;
using paddle::platform::RecordInstantEvent;
using paddle::platform::TracerEventType;
using paddle::platform::RecordEvent;

int main() {
  paddle::jit::utils::InitKernelSignatureMap();

  ProfilerOptions options;
  options.trace_level = 2;
  options.trace_switch = 3;
  auto profiler = Profiler::Create(options);
  profiler->Prepare();
  profiler->Start();

  // tensor op
  std::cout << "Run Start" << std::endl;
  auto d = paddle::Tensor("test");
  auto a = paddle::full({3, 4}, 2.0);
  auto b = paddle::full({4, 5}, 3.0);
  auto out = paddle::matmul(a, b);
  std::cout << "Run End" << std::endl;
 
  // load model
  auto layer =
      paddle::jit::Load("chunk_wenetspeech_static/export.jit", phi::CPUPlace());

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

  auto profiler_result = profiler->Stop(); 
  profiler_result->Save("test.prof");
}
