#include "hard_sphere.hpp"

int main(){
    int n = 100;
    double vf_init = 0.20;
    double vf_final = 0.30;
    double box_size = pow(n * M_PI / 6 / vf_init, 1.0 / 3.0);
    vector<double> box {box_size, box_size, box_size};
    vector<bool> is_pbc_bulk {true, true, true};
    vector<bool> is_pbc_conf {true, true, false};  // hard wall in z-directin

    cout << "Creating bulk system" << endl;
    HSMC bulk{n, box, is_pbc_bulk, 1};
    bulk.fill_hs();
    bulk.crush(vf_final, 0.02);
    for (int i=0; i<1000; i++){
        bulk.sweep();
        if (i % 100 == 0){
            dump(bulk, "bulk.xyz");
        }
    }

    HSMC confined{n, bulk.box_, is_pbc_conf, 0.2};
    load(confined, "bulk.xyz");

    cout << "Writing Configurations" << endl;
    for (int i=0; i<1000; i++){
        confined.sweep();
        if (i % 100 == 0){
            dump(confined, "confined.xyz");
        }
    }
}
