/*
 * @Author: your name
 * @Date: 2020-11-27 20:40:44
 * @LastEditTime: 2020-11-28 15:37:39
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /exp2/extension/functional/src/utils.hpp
 */
#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor") \

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)                     \

#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                             \
           #x " must be an int tensor")

#define CHECK_IS_FLOAT(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                           \
           #x " must be a float tensor")

#endif
