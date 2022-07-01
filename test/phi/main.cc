#include "paddle/extension.h"
#include "paddle/phi/api/all.h"

int main (){
    std::cout << "Run Start" << std::endl;

    auto d = paddle::Tensor("test");

    auto a = paddle::full({3, 4}, 2.0);
    auto b = paddle::full({4, 5}, 3.0);
    auto out = paddle::matmul(a, b);

    std::cout << "Run End" << std::endl;
}
