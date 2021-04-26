template<class T>
void PBC<T>::update_volume(){
    volume_ = 1;
    for (auto size : box_){
        volume_ *= size;
    }
}

template<class T>
void PBC<T>::fix_position(T& positions) const{
    for (int d = 0; d < dim_; d++){
        if (is_pbc_[d]){
            // python equivalent: positions[d] %= box_[d]
            positions.row(d) = positions.row(d).unaryExpr(
                [&](double x){
                    double y = remainder(x, box_[d]);
                    if (y < 0) {y = y + box_[d];}
                    return y;
                }
            );
        }
    }
}

template<class T>
void PBC<T>::fix_position(T& positions, int i) const{
    for (int d = 0; d < dim_; d++){
        if (is_pbc_[d]){
            double pos_1d = positions(d, i);
            pos_1d = remainder(pos_1d, box_[d]);
            if (pos_1d < 0) {
                pos_1d = pos_1d + box_[d];
            }
            positions(d, i) = pos_1d;
        }
    }
}

template<class T>
double PBC<T>::get_dist_sq(const T& positions, int i, int j) const{
    double dist_sq = 0;
    double shift_1d = 0;
    for (int d = 0; d < dim_; d++){
        shift_1d = (positions(d, i) - positions(d, j)) / box_[d];
        if (is_pbc_[d]) {
            shift_1d -= rint(shift_1d);
        }
        shift_1d *= box_[d];
        dist_sq += shift_1d * shift_1d;
    }
    return dist_sq;
}

template<class T>
VerletList<T>::VerletList(double r_cut, double r_skin)
    : rc_{r_cut}, rl_{r_skin} {
        rl2_ = rl_ * rl_;
        rc2_ = rc_ * rc_;
        dr_sq_ = pow(rl_ - rc_, 2);
}

template<class T>
void VerletList<T>::build(const T& positions){
    double d2{0};
    size_ = positions.cols();
    point_.clear();
    nlist_.clear();

    point_.push_back(0);
    for (int i = 0; i < size_; i++){
        for (int j = 0; j < size_; j++){
            if (i != j){
                d2 = (positions.col(i) - positions.col(j)).array().pow(2).sum();
                if (d2 < rl2_){
                    nlist_.push_back(j);
                }
            }
        }
        point_.push_back(nlist_.size());
    }
    point_size_ = point_.size(); // particle number + 1
}

template<class T>
void VerletList<T>::build(const T& positions, const PBC<T>& boundary){
    double d2{0};
    size_ = positions.cols();
    point_.clear();
    nlist_.clear();

    point_.push_back(0);
    for (int i = 0; i < size_; i++){
        for (int j = 0; j < size_; j++){
            if (i != j){
                d2 = boundary.get_dist_sq(positions, i, j);
                if (d2 < rl2_){
                    nlist_.push_back(j);
                }
            }
        }
        point_.push_back(nlist_.size());
    }
    point_size_ = point_.size(); // particle number + 1
}

template<class T>
vector<int> VerletList<T>::get_neighbours(int i){
    vector<int> result {};
    for (int k = point_[i]; k < point_[i+1]; k++){
        result.push_back(nlist_[k]);
    }
    return result;
}

template<class T>
void dump(const T& system, string filename){
    ofstream f;
    f.open(filename, ios::out | ios::app);
    f << system.n_ << endl;
    if (system.positions_.rows() == 3){
        f << "id, x, y, z, vx, vy, vz" << endl;
        for (int i = 0; i < system.n_; i++ ) {
            f << i << " "
                << system.positions_(0, i)  << " "
                << system.positions_(1, i)  << " "
                << system.positions_(2, i)  << " " << endl;
        }
    } else if (system.positions_.rows() == 2){
        f << "id, x, y, vx, vy" << endl;
        for (int i = 0; i < system.n_; i++ ) {
            f << i << " "
              << system.positions_(0, i)  << " "
              << system.positions_(1, i)  << " " << endl;
        }
    } else {
        throw("invalid dimension");
    }
    f.close();
}


template<class T>
void load(T& system, string filename){
    ifstream f;
    string line;
    regex head_pattern{"\\d+"};
    smatch matched;
    int head_lines = 2;
    string num;
    int N = 0;
    int total_frame = 0;
    
    // find total number of frames
    f.open(filename, ios::in);
    while (f) {
        getline(f, line);
        if (regex_match(line, matched, head_pattern)){
            N = stoi(line);
            total_frame += 1;
            for (int i=0; i<N; i++) getline(f, line);
        }
    }
    f.close();
    
    // jump to the last frame 
    f.open(filename, ios::in);
    for (int i = 0; i < total_frame - 1; i++){
        for (int j = 0; j < N + head_lines; j++){
        getline(f, line);
        }
    }
    
    // load the data
    if (system.positions_.rows() == 3){
        for (int i = 0; i < N + head_lines; i++){
            getline(f, line);
            if (i > 1) {
                istringstream ss(line);
                ss >> num;
                for (int j = 0; j < 3; j++){
                    ss >> system.positions_(j, i - head_lines);
                }
            }
        }
    } else if (system.positions_.rows() == 2) {
        for (int i = 0; i < N + head_lines; i++){
            getline(f, line);
            
            if (i > 1) {
                istringstream ss(line);
                ss >> num;
                for (int j = 0; j < 2; j++){
                    ss >> system.positions_(j, i - head_lines);
                }
            }
        }
    } else {
        throw("invalid dimension");
    }
    system.rebuild_nlist();  // rebuild the neighbour list
    f.close();
}


template<class T>
void recursive_product(
    const vector<vector<T>>& arrays, vector<vector<T>>& result, int idx=0
){
    if (idx == arrays.size()) {
        return;
    }
    else if (idx == 0){
        for (auto val : arrays[0]){
            result.push_back( vector<T> {val} );
        }
    } else {
        vector<vector<T>> new_result;
        for (auto x : result){  // {1}, {2}
            for (auto val : arrays[idx]){
                x.push_back(val);
                new_result.push_back(x);
                x.pop_back();
            }
        }
        result = new_result;
    }
    recursive_product(arrays, result, ++idx);
}

template<class T>
vector<vector<T>> product_nd(const vector<vector<T>>& arrays){
    vector<vector<T>> result;
    recursive_product(arrays, result);
    return result;
}



template<class T>
CellList<T>::CellList(double r_cut, vector<double> box, vector<bool> is_pbc)
    : r_cut_{r_cut}, box_{box}, is_pbc_{is_pbc},
      boundary_{box, is_pbc} {
    dim_ = box.size();
    size_ = 0;
    sc_ = 1;
    // sc_ = floor(side_min / r_cut_ / 2);
    if (sc_ < 1) {
        sc_ = 1;
    }
    for (int d = 0; d < dim_; d++){
        head_shape_[d] = sc_;
    }
}


