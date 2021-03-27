#include "hard_sphere.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(chard_sphere, m){
    py::class_<HSMC>(m, "HSMC")
        .def(py::init<int, vector<double>, vector<bool>, double>())
        .def("fill_idea_gas", &HSMC::fill_ideal_gas)
        .def("fill_hs", &HSMC::fill_hs)
        .def("crush", &HSMC::crush)
        .def("sweep", &HSMC::sweep)
        .def("rebuild_nlist", &HSMC::rebuild_nlist)
        .def("get_vf", &HSMC::get_vf)
        .def("load_positions", &HSMC::load_positions)
        .def("copy_positions", &HSMC::get_positions)
        .def(
            "get_positions", &HSMC::get_positions,
            py::return_value_policy::reference_internal
        )
        .def(
            "view_positions", &HSMC::view_positions,
            py::return_value_policy::reference_internal
        )
        .def(
            "__repr__",
            [](const HSMC& obj){
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
