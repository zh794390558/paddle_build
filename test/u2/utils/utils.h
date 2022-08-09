#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace ppspeech {

#define DISALLOW_COPY_AND_ASSIGN(Type) \
    Type(const Type&) = delete;        \
    Type& operator=(const Type&) = delete;


const float kFloatMax = std::numeric_limits<float>::max();

// kSpaceSymbol in UTF-8 is: ▁
const char kSpaceSymbol[] = "\xe2\x96\x81";

// sum of two probabilities in log scale
float LogSumExp(float x, float y);

template<typename T>
void TopK(const std::vector<T>& data, int32_t k, std::vector<T>* values, std::vector<int>* indices);

} // namespace ppspeech