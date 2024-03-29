#include "hard_sphere.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(chard_sphere, m){
    py::class_<HSMC>(m, "HSMC")
        .def(
            py::init<
                int, vector<double>, vector<bool>, vector<bool>, double
            >(),
            pybind11::arg("n"),
            pybind11::arg("box"),
            pybind11::arg("is_pbc"),
            pybind11::arg("is_hard"),
            pybind11::arg("r_skin")=4
        )
        .def("fill_idea_gas", &HSMC::fill_ideal_gas)
        .def("fill_hs", &HSMC::fill_hs)
        .def("crush", &HSMC::crush)
        .def("crush_along_axis", &HSMC::crush_along_axis)
        .def("sweep", &HSMC::sweep)
        .def("rebuild_nlist", &HSMC::rebuild_nlist)
        .def("get_vf", &HSMC::get_vf)
        .def("load_positions", &HSMC::load_positions)
        .def("copy_positions", &HSMC::get_positions)
        .def("set_indices", &HSMC::set_indices)
        .def("report_overlap", &HSMC::report_overlap)
        .def("get_box", &HSMC::get_box)
        .def(
            "get_positions", &HSMC::get_positions,
            py::return_value_policy::reference_internal
        )
        .def(
            "view_positions", &HSMC::view_positions,
            py::return_value_policy::reference_internal
        )
        .def("__repr__", [](const HSMC& obj){ return obj.repr();})
        .def("__str__", [](const HSMC& obj){ return obj.str();} );
}
