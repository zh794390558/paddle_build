#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace ppspeech {

// DISALLOW_COPY_AND_ASSIGN disallows the copy and operator= functions.
// It goes in the private: declarations in a class.
#define DISALLOW_COPY_AND_ASSIGN(Type) \
    Type(const Type&) = delete;        \
    Type& operator=(const Type&) = delete;

// A macro to disallow all the implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
//
// This should be used in the private: declarations for a class
// that wants to prevent anyone from instantiating it. This is
// especially useful for classes containing only static methods.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
  TypeName();                                    \
  DISALLOW_COPY_AND_ASSIGN(TypeName);


const float kFloatMax = std::numeric_limits<float>::max();

// kSpaceSymbol in UTF-8 is: ▁
const char kSpaceSymbol[] = "\xe2\x96\x81";

// sum of two probabilities in log scale
float LogSumExp(float x, float y);

template<typename T>
void TopK(const std::vector<T>& data, int32_t k, std::vector<T>* values, std::vector<int>* indices);

} // namespace ppspeech