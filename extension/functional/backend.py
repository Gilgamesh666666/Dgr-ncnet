import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_multi_shape_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'knn_dir/knn.cpp',
                    'knn_dir/knn.cu',
                    'soft_mutual_score_dir/soft_mutual_score.cpp',
                    'soft_mutual_score_dir/soft_mutual_score.cu',
                    'soft_mutual_score_dir_On/soft_mutual_score_On.cpp',
                    'soft_mutual_score_dir_On/soft_mutual_score_On.cu',
                    'bindings.cpp'
                ]]
                )

__all__ = ['_backend']
