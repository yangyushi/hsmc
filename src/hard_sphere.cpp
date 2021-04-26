#include "hard_sphere.hpp"

vector<int> unravel_index(int index, const vector<int>& shape) {
    int dim = shape.size();
    vector<int> result;
    int size;
    int tmp;
    for (int d1 = 0; d1 < dim; d1++){
        size = 1;
        for (int d2 = d1 + 1; d2 < dim; d2++){
            size *= shape[d2];
        }
        tmp = floor(index / size);
        result.push_back(tmp);
        index -= tmp * size;
    }
    return result;
}

/*
 * Initialise the system by randomly populating the system
 */
HSMC::HSMC(int n, vector<double> box, vector<bool> is_pbc, vector<bool> is_hard)
    : n_{n}, box_{box}, positions_{dim_, n},
    boundary_{box, is_pbc}, total_disp_{dim_, n},
    is_pbc_{is_pbc}, is_hard_{is_hard} {
    step_ = 1;
    for (int i = 0; i < n_; i++){
        rand_indices_.push_back(i);
    }
    total_disp_.setZero();
    this->fill_ideal_gas();
    for (int d = 0; d < dim_; d++){
        if (is_hard_[d]){
            hard_dim_.push_back(d);
        }
    }
}

void HSMC::shuffle_indices(){
    random_device rd;
    mt19937 g(rd());
    shuffle(rand_indices_.begin(), rand_indices_.end(), g);
}


// Fill the system with random points
void HSMC::fill_ideal_gas(){
    positions_.setRandom(3, n_);  // ~U(-1, 1)
    positions_.array() += 1;  // ~U(0, 2)
    for (int d = 0; d < dim_; d++){
        positions_.row(d) *= box_[d] / 2;  // ~U(0, box)
    }
}


void HSMC::fill_hs(){
    if (this->get_vf() > 0.3){
        throw runtime_error(
            "Initial Volumn Fraction > 0.3, can't randomly initialise"
        );
    }
    for (auto i : rand_indices_){
        bool is_overlap = true;
        while (is_overlap) {
            is_overlap = false;
            positions_.col(i) = Eigen::MatrixXd::Random(3, 1).array() + 1; // ~U(0, 2)
            for (int d = 0; d < dim_; d++){
                positions_(d, i) *= box_[d] / 2.0;  // ~U(0, box)
            }
            for (int j = 0; j < i; j++){
                if (boundary_.get_dist_sq(positions_, i, j) < 1){
                    is_overlap = true;
                    break;
                }
            }
        }
    }
    this->rebuild_nlist();
}


bool HSMC::check_hardwall(){
    for (auto d : hard_dim_){  // check overlap with hard walls
        for (int i = 0; i < n_ - 1; i++){
            if (positions_(d, i) < 0) {
                return true;
            }
            if (positions_(d, i) > box_[d]) {
                return true;
            }
        }
    }
    return false;
}

bool HSMC::check_hardwall(int idx){
    for (auto d : hard_dim_){  // check overlap with hard walls
        if (positions_(d, idx) < 0) {
            return true;
        }
        if (positions_(d, idx) > box_[d]) {
            return true;
        }
    }
    return false;
}



bool HSMC::report_overlap(){
    for (int i = 0; i < n_ - 1; i++){
        if (this->check_hardwall(i)){
            return true;
        }
        for (int j = i + 1; j < n_; j++){  // check overlap with other particles
            if (boundary_.get_dist_sq(positions_, i, j) < 1){
                return true;
            }
        }
    }
    return false;
}


bool HSMC::check_overlap(int i){
    double dist_sq = 0;

    if (this->check_hardwall(i)) {
        return true;
    }

    for (auto j : this->get_neighbours(i)){
        dist_sq = boundary_.get_dist_sq(positions_, i, j);
        if (dist_sq < 1){
            return true;
        }
    }
    return false;
}

bool HSMC::check_overlap(){
    double dist_sq = 0;

    for (int i : rand_indices_){
        if (this->check_hardwall(i)) {
            return true;
        }
        for (auto j : this->get_neighbours(i)){
            dist_sq = boundary_.get_dist_sq(positions_, i, j);
            if (dist_sq < 1){
                return true;
            }
        }
    }
    return false;
}

bool HSMC::advance(int idx){
    Vec3D previous_pos = positions_.col(idx);
    Vec3D disp = Eigen::MatrixXd::Random(3, 1) * step_ / 2.0;
    positions_.col(idx) += disp;
    boundary_.fix_position(positions_, idx);
    this->check_disp_sum(idx, disp);
    if (this->check_overlap(idx)){ // reject trial movement
        this->check_disp_sum(idx, -disp);
        positions_.col(idx) = previous_pos;
        return false;
    } else { // confirm movement & update neighbour list
        return true;
    }
}

void HSMC::check_disp_sum(int idx, const Vec3D& disp){
    total_disp_.col(idx) += disp;
    double disp_sq = total_disp_.col(idx).squaredNorm();
    double disp_sq_max = total_disp_.col(ldi_).squaredNorm();

    if (disp_sq > disp_sq_max) {
        ldi_ = idx;
        disp_sq_max = disp_sq;
    }
    if (disp_sq_max * 4 >= vlist_.dr_sq_) {
        this->rebuild_nlist();
    }
}

void HSMC::adjust_step(int accept_number){
    double accept_ratio = (double) accept_number / n_;
    if (accept_ratio < 0.45) {
        step_ *= 0.95;
    } else if (accept_ratio > 0.55) {
        step_ *= 1.05;
    }
}

void HSMC::sweep(){
    this->shuffle_indices();
    int accept_number = 0;
    bool is_succeed = false;
    for (auto idx : rand_indices_){
        is_succeed = this->advance(idx);
        if (is_succeed){ accept_number++; }
    }
    this->adjust_step(accept_number);
}

/*
 * get all the overlapped indices with the neighbour list
 */
vector<array<int, 2>> HSMC::get_overlap_indices(){
    double dist_sq = 0;
    vector<array<int, 2>> overlap_indices {};
    for (int i = 0; i < n_; i++){
        for (auto j : this->get_neighbours(i)){
            dist_sq = boundary_.get_dist_sq(positions_, i, j);
            if (dist_sq < 1){
                overlap_indices.push_back(array<int, 2>{i, j});
            }
        }
    }
    return overlap_indices;
}

/*
 * Try to remove all the overlapped particles.
 * This method would be very slow in high density system.
 * Should only use this method to get a non-overlapping initial configuration.
 */
void HSMC::remove_overlap(){
    bool is_overlap = this->check_overlap();
    while (is_overlap){
        this->sweep();
        is_overlap = this->check_overlap();
    }
}

/*
 * Gradually increase the volume fraction by rescaling the system
 */
void HSMC::crush(double target_vf, double delta_vf){
    double vf = this->get_vf();
    double vf_new;
    while (vf < target_vf){
        if (target_vf - vf < delta_vf) {
            vf_new = target_vf;
        } else {
            vf_new = vf + delta_vf;
        }
        double scale = pow(vf / vf_new, 1.0 / 3.0);
        positions_.array() *= scale;
        boundary_.rescale(scale);
        box_ = boundary_.box_;
        boundary_.fix_position(positions_);
        this->rebuild_nlist();
        this->remove_overlap();
        vf = vf_new;
        cout << "Crushed to higher volume fraction, step: " << step_
             << "; vf: " << this->get_vf() * 100 << endl;
        }

    cout << "final box size: ";
    for (int d = 0; d < dim_; d++){
        cout << box_[d];
        if (d != dim_){
            cout << ", ";
        }
    }
    cout << endl;
}

/*
 * Gradually increase the volume fraction by rescaling the system
 */
void HSMC::crush_along_axis(double target_vf, double delta_vf, int axis){
    double vf = this->get_vf();
    double vf_new, scale;

    while (vf < target_vf){
        if (target_vf - vf < delta_vf) {
            vf_new = target_vf;
        } else {
            vf_new = vf + delta_vf;
        }
        scale = vf / vf_new;

        positions_.row(axis).array() *= scale;
        boundary_.rescale(scale, axis);
        box_ = boundary_.box_;

        boundary_.fix_position(positions_);
        this->rebuild_nlist();
        this->remove_overlap();

        vf = vf_new;
        cout << "Crushed to higher volume fraction, step: " << step_
             << "; vf: " << this->get_vf() * 100 << endl;
        }

    cout << "final box size: ";
    for (int d = 0; d < dim_; d++){
        cout << box_[d];
        if (d != dim_){
            cout << ", ";
        }
    }
    cout << endl;
}

string HSMC::str() const{
    ostringstream str_stream;
    vector<string> side_names {"X", "Y", "Z"};
    str_stream << "Hard Sphere MC Simulaion, with periodic boundary on ";
    for (int d = 0; d < dim_; d++){
        if (is_pbc_[d]){
            str_stream << side_names[d];
        }
    }
    str_stream << " sides" << endl;
    str_stream << "N = " << n_ << "; Box = (" << setprecision(8);
    for (int d = 0; d < dim_; d++){
        str_stream << box_[d];
        if (d < 2) {str_stream << ", ";}
    }
    str_stream << "); Volumn Fraction = " << get_vf() << endl;
    return str_stream.str();
}

string HSMC::repr() const{
    ostringstream str_stream;
    vector<string> side_names {"X", "Y", "Z"};
    str_stream << "Hard Sphere MC Simulaion, with periodic boundary on ";
    for (int d = 0; d < dim_; d++){
        if (is_pbc_[d]){
            str_stream << side_names[d];
        }
    }
    str_stream << " sides" << endl;
    str_stream << "N = " << n_ << "; Box = (" << setprecision(8);
    for (int d = 0; d < dim_; d++){
        str_stream << box_[d];
        if (d < 2) {str_stream << ", ";}
    }
    str_stream << "); Volumn Fraction = " << get_vf() << endl;
    str_stream << "(address: " << hex << &positions_ << ")" << endl;
    return str_stream.str();
}
