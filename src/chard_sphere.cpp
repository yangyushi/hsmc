#include "hard_sphere.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(chard_sphere, m){
    py::class_<HSMCWet>(m, "HSMCWet")
        .def(py::init<int, vector<double>, double>())
        .def("remove_overlap", &HSMCWet::remove_overlap)
        .def("crush", &HSMCWet::crush)
        .def(
            "__repr__",
            [](const HSMCWet& obj){
                ostringstream str_stream;
                str_stream << "Hard Sphere MC Simulaion, PBC only with XY sides" << endl;
                str_stream << "N = " << obj.n_ << "; Box = (" << setprecision(4);
                for (int d = 0; d < obj.dim_; d++){
                    str_stream << obj.box_[d];
                    if (d < 2) {str_stream << ", ";}
                }
                str_stream << "); Volumn Fraction = " << obj.get_vf() << endl;
                return str_stream.str();
            }
        );
}
