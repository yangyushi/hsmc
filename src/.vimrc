let g:ale_linters = {'cpp': ['clang']}  " use clang compiler
let cpp_flags = '-std=c++11 -Wall '  " c++ standard & show all warnings
" for eigen
let cpp_flags = cpp_flags . '-I/usr/local/include/eigen3 '
let cpp_flags = cpp_flags . '-Wno-unknown-warning-option' . ' '  " supress eigen warnings
" for pybind11
let cpp_flags = cpp_flags . '-I/usr/local/include/python3.7m '
let cpp_flas = cpp_flags . '-I/usr/local/lib/python3.7/site-packages/pybind11/include '
let g:ale_cpp_clang_options = cpp_flags  " set the flag for linting

