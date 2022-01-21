#include <pybind11/pybind11.h>
#include "soft_mutual_score_dir/soft_mutual_score.hpp"
#include "soft_mutual_score_dir_On/soft_mutual_score_On.hpp"
#include "knn_dir/knn.hpp"
PYBIND11_MODULE(_multi_shape_pvcnn_backend, m) {
  m.def("knn_forward_cuda", &knn_forward_cuda,
        "K Nearest Neighbor forward(CUDA)");
  m.def("knn_backward_cuda", &knn_backward_cuda,
        "K Nearest Neighbor backward(CUDA)");
  m.def("soft_mutual_score_forward", &soft_mutual_score_forward,
        "Soft Mutual Score forward(CUDA)");
  m.def("soft_mutual_score_backward", &soft_mutual_score_backward,
        "Soft Mutual Score backward(CUDA)");
  m.def("soft_mutual_score_forward_On", &soft_mutual_score_forward_On,
        "Soft Mutual Score forward(CUDA)");
  m.def("soft_mutual_score_backward_On", &soft_mutual_score_backward_On,
        "Soft Mutual Score backward(CUDA)");
}
