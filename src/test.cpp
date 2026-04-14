#include "hard_sphere.hpp"

int main() {
    int n = 100;
    double vf_init = 0.20;
    double vf_final = 0.3142;
    double box_size = std::pow(n * M_PI / 6 / vf_init, 1.0 / 3.0);
    std::vector<double> box{box_size, box_size, box_size};
    std::vector<bool> is_pbc_bulk{true, true, true};
    std::vector<bool> is_pbc_conf{true, true, false};
    std::vector<bool> is_hard_bulk{false, false, false};
    std::vector<bool> is_hard_conf{false, false, true};

    std::cout << "Creating bulk system" << std::endl;
    hsmc::HSMC bulk{n, box, is_pbc_bulk, is_hard_bulk, 4.0};
    bulk.fill_hs();
    bulk.crush(vf_final, 0.02);
    for (int i = 0; i < 1000; ++i) {
        bulk.sweep();
        if (i % 100 == 0) {
            hsmc::dump(bulk, "bulk.xyz");
        }
    }

    hsmc::HSMC confined{n, bulk.box_, is_pbc_conf, is_hard_conf, 4.0};
    hsmc::load(confined, "bulk.xyz");

    std::cout << "Writing Configurations" << std::endl;
    for (int i = 0; i < 1000; ++i) {
        confined.sweep();
        if (i % 100 == 0) {
            hsmc::dump(confined, "confined.xyz");
        }
    }
}
