#include "hard_sphere.hpp"


/*
 * Initialise the system by randomly populating the system
 */
HSConfined::HSConfined(int n, vector<double> box, double step)
    : n_{n}, box_{box}, positions_{dim_, n}, step_{step},
    boundary_{box, is_pbc_}, total_disp_{dim_, n} {
    for (int i = 0; i < n_; i++){
        rand_indices_.push_back(i);
    }
    total_disp_.setZero();
    if (this->get_vf() > 0.3){
        throw runtime_error(
            "Initial Volumn Fraction > 0.3, can't randomly initialise"
        );
    }
    this->fill_hs();
    vlist_.build(positions_, boundary_);
}

void HSConfined::shuffle_indices(){
    random_device rd;
    mt19937 g(rd());
    shuffle(rand_indices_.begin(), rand_indices_.end(), g);
}

/*
 * Fill the system with random points, pretty useless method
 */
void HSConfined::fill_ideal_gas(){
    positions_.setRandom(3, n_);  // ~U(-1, 1)
    positions_.array() += 1;  // ~U(0, 2)
    for (int d = 0; d < dim_; d++){
        positions_.row(d) *= boundary_.box_[d] / 2;  // ~U(0, box)
    }
}


void HSConfined::fill_hs(){
    for (int i = 0; i < n_; i++){
        bool is_overlap = true;
        while (is_overlap) {
            is_overlap = false;
            positions_.col(i) = Eigen::MatrixXd::Random(3, 1).array() + 1;
            for (int d = 0; d < dim_; d++){
                positions_(d, i) *= boundary_.box_[d] / 2;  // ~U(0, box)
            }
            for (int j = 0; j < i; j++){
                if (boundary_.get_dist_sq(positions_, i, j) < 1){
                    is_overlap = true;
                    break;
                }
            }
        }
    }
}


bool HSConfined::check_overlap(int i){
    if ((positions_(2, i) < 0) or (positions_(2, i) > boundary_.box_[2])) {
        return true;
    }
    double dist_sq = 0;
    int j = 0;
    for (int k = vlist_.point_[i]; k < vlist_.point_[i+1]; k++){
        j = vlist_.nlist_[k];
        dist_sq = boundary_.get_dist_sq(positions_, i, j);
        if (dist_sq < 1){
            return true;
        }
    }
    return false;
}


bool HSConfined::check_overlap(){
    double dist_sq = 0;
    int j = 0;
    for (int i = 0; i < n_; i++){
        if ((positions_(2, i) < 0) or (positions_(2, i) > boundary_.box_[2])) {
            return true;
        }
        for (int k = vlist_.point_[i]; k < vlist_.point_[i+1]; k++){
            j = vlist_.nlist_[k];
            dist_sq = boundary_.get_dist_sq(positions_, i, j);
            if (dist_sq < 1){
                return true;
            }
        }
    }
    return false;
}


/*
 * Move a particle randomly, only accept without overlap.
 * keep a record of the total displacement for updating the neighbour list
 */
bool HSConfined::advance(int idx){
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

/*
 * Move a particle without checking the overlap
 */
void HSConfined::advance_no_check(int idx){
    Vec3D disp = Eigen::MatrixXd::Random(3, 1) * step_ / 2.0;
    positions_.col(idx) += disp;
    boundary_.fix_position(positions_, idx);
    this->check_disp_sum(idx, disp);
}


/*
 * Check the largest and second largest total displacement for updating the Verlet list
 */
void HSConfined::check_disp_sum(int idx, const Vec3D& disp){
    total_disp_.col(idx) += disp;
    double disp_sq = total_disp_.col(idx).squaredNorm();
    double disp_sq_max = total_disp_.col(ldi_).squaredNorm();

    if (disp_sq > disp_sq_max) {
        ldi_ = idx;
        disp_sq_max = disp_sq;
    }
    if (disp_sq_max * 4 >= vlist_.dr_sq_) {
        this->rebuild();
    }
}


void HSConfined::rebuild(){
    vlist_.build(positions_, boundary_);
    total_disp_.setZero();
}


/*
 * Adjust the movement step based on the accptance ratio
 *   following Allen & Tildesley's liquid simulation book
 */
void HSConfined::adjust_step(int accept_number){
    double accept_ratio = (double) accept_number / n_;
    if (accept_ratio < 0.45) {
        step_ *= 0.95;
    } else if (accept_ratio > 0.55) {
        step_ *= 1.05;
    }
}


void HSConfined::sweep(){
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
vector<IndexPair> HSConfined::get_overlap_indices(){
    double dist_sq = 0;
    int j = 0;
    vector<IndexPair> overlap_indices {};
    for (int i = 0; i < n_; i++){
        for (int k = vlist_.point_[i]; k < vlist_.point_[i+1]; k++){
            j = vlist_.nlist_[k];  // neighbour index
            dist_sq = boundary_.get_dist_sq(positions_, i, j);
            if (dist_sq < 1){
                overlap_indices.push_back(IndexPair{i, j});
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
void HSConfined::remove_overlap(){
    bool is_overlap = this->check_overlap();
    while (is_overlap){
        this->sweep();
        // force the system to try to move overlap indices
        for (auto idx_pair : this->get_overlap_indices()){
            advance(idx_pair[0]);
            advance(idx_pair[1]);
        }
        is_overlap = this->check_overlap();
    }
}


/*
 * Gradually increase the volumn fraction by rescaling the system
 */
void HSConfined::crush(double target_vf, double delta_vf){
    for (double vf = get_vf(); vf < target_vf; vf += delta_vf){
        double vf_new = vf + delta_vf;
        double scale = pow(vf / vf_new, 1.0 / 3.0);
        positions_.array() *= scale;
        boundary_.rescale(scale);
        box_ = boundary_.box_;
        boundary_.fix_position(positions_);
        vlist_.build(positions_, boundary_);
        this->remove_overlap();
        cout << "Crushed to higher volumn fraction, step: " << step_
             << "; vf: " << this->get_vf() * 100 << endl;
        }

    cout << "final box size: ";
    for (int d = 0; d < dim_; d++){
        cout << boundary_.box_[d];
        if (d != dim_){
            cout << ", ";
        }
    }
    cout << endl;
}


