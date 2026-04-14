#include "hard_sphere.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(chard_sphere, m) {
    py::class_<hsmc::HSMC>(m, "HSMC")
        .def(
            py::init<
                int, std::vector<double>, std::vector<bool>,
                std::vector<bool>, double
            >(),
            py::arg("n"),
            py::arg("box"),
            py::arg("is_pbc"),
            py::arg("is_hard"),
            py::arg("r_skin") = 4
        )
        .def("fill_ideal_gas", &hsmc::HSMC::fill_ideal_gas)
        .def("fill_hs", &hsmc::HSMC::fill_hs)
        .def("crush", &hsmc::HSMC::crush)
        .def("crush_along_axis", &hsmc::HSMC::crush_along_axis)
        .def("sweep", &hsmc::HSMC::sweep)
        .def("rebuild_nlist", &hsmc::HSMC::rebuild_nlist)
        .def("get_vf", &hsmc::HSMC::get_vf)
        .def("load_positions", &hsmc::HSMC::load_positions)
        .def("copy_positions", &hsmc::HSMC::get_positions)
        .def("set_indices", &hsmc::HSMC::set_indices)
        .def("report_overlap", &hsmc::HSMC::report_overlap)
        .def("get_box", &hsmc::HSMC::get_box)
        .def(
            "get_positions", &hsmc::HSMC::get_positions,
            py::return_value_policy::reference_internal
        )
        .def(
            "view_positions", &hsmc::HSMC::view_positions,
            py::return_value_policy::reference_internal
        )
        .def("__repr__", [](const hsmc::HSMC& obj) { return obj.repr(); })
        .def("__str__", [](const hsmc::HSMC& obj) { return obj.str(); });
}
