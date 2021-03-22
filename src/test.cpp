#include "hard_sphere.hpp"

int main(){
    int n = 2000;
    double vf_init = 0.2;
    double box_size = pow(n * M_PI / 6 / vf_init, 1.0 / 3.0);
    vector<double> box {box_size, box_size, box_size};
    cout << "Initial Box size is: " << box_size << endl;
    HSConfined system{n, box, 0.2};
    cout << "System Made" << endl;
    system.crush(0.4, 0.02);

    load(system, "sample.xyz");

    cout << "Writing Configurations" << endl;
    for (int i=0; i<100; i++){
        system.sweep();
        if (i % 20 == 0){
            dump(system, "test.xyz");
        }
    }
}
