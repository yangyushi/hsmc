template<class T>
void PBC<T>::update_volume() {
    volume_ = 1.0;
    for (double size : box_) {
        volume_ *= size;
    }
}

template<class T>
void PBC<T>::refresh_cache() {
    for (int axis = 0; axis < 3; ++axis) {
        if (axis < dim_) {
            box_cache_[axis] = box_[axis];
            inv_box_cache_[axis] = 1.0 / box_[axis];
            is_pbc_cache_[axis] = is_pbc_[axis];
            continue;
        }
        box_cache_[axis] = 1.0;
        inv_box_cache_[axis] = 1.0;
        is_pbc_cache_[axis] = false;
    }
}

template<class T>
double PBC<T>::axis_box(int axis) const {
    return axis < 3 ? box_cache_[axis] : box_[axis];
}

template<class T>
double PBC<T>::axis_inv_box(int axis) const {
    return axis < 3 ? inv_box_cache_[axis] : 1.0 / box_[axis];
}

template<class T>
bool PBC<T>::axis_is_pbc(int axis) const {
    return axis < 3 ? is_pbc_cache_[axis] : is_pbc_[axis];
}

template<class T>
void PBC<T>::fix_position(T& positions) const {
    for (int axis = 0; axis < dim_; ++axis) {
        if (!axis_is_pbc(axis)) {
            continue;
        }
        const double box = axis_box(axis);
        positions.row(axis) = positions.row(axis).unaryExpr([box](double value) {
            double wrapped = std::remainder(value, box);
            if (wrapped < 0.0) {
                wrapped += box;
            }
            return wrapped;
        });
    }
}

template<class T>
void PBC<T>::fix_position(T& positions, int i) const {
    for (int axis = 0; axis < dim_; ++axis) {
        if (!axis_is_pbc(axis)) {
            continue;
        }
        const double box = axis_box(axis);
        double value = positions(axis, i);
        if (value >= 0.0 && value < box) {
            continue;
        }
        if (value >= -box && value < 2.0 * box) {
            if (value < 0.0) {
                value += box;
            } else if (value >= box) {
                value -= box;
            }
        } else {
            value = std::remainder(value, box);
            if (value < 0.0) {
                value += box;
            }
        }
        positions(axis, i) = value;
    }
}

template<class T>
double PBC<T>::get_dist_sq(const T& positions, int i, int j) const {
    double dist_sq = 0.0;
    for (int axis = 0; axis < dim_; ++axis) {
        double delta = positions(axis, i) - positions(axis, j);
        if (axis_is_pbc(axis)) {
            delta -= axis_box(axis) * std::nearbyint(delta * axis_inv_box(axis));
        }
        dist_sq += delta * delta;
    }
    return dist_sq;
}

template<class T>
VerletList<T>::VerletList(double r_cut, double r_skin)
    : dr_sq_{std::pow(r_skin - r_cut, 2)},
      rc_{r_cut},
      rc2_{r_cut * r_cut},
      rl_{r_skin},
      rl2_{r_skin * r_skin} {}

template<class T>
void VerletList<T>::build(const T& positions) {
    size_ = positions.cols();
    std::vector<std::vector<int>> adjacency(size_);
    point_.clear();
    point_.reserve(size_ + 1);

    for (int i = 0; i < size_; ++i) {
        for (int j = i + 1; j < size_; ++j) {
            const double d2 =
                (positions.col(i) - positions.col(j)).squaredNorm();
            if (d2 >= rl2_) {
                continue;
            }
            adjacency[i].push_back(j);
            adjacency[j].push_back(i);
        }
    }

    point_.push_back(0);
    nlist_.clear();
    for (const std::vector<int>& neighbours : adjacency) {
        nlist_.insert(nlist_.end(), neighbours.begin(), neighbours.end());
        point_.push_back(static_cast<int>(nlist_.size()));
    }
    point_size_ = point_.size();
}

template<class T>
void VerletList<T>::build(const T& positions, const PBC<T>& boundary) {
    size_ = positions.cols();
    std::vector<std::vector<int>> adjacency(size_);
    point_.clear();
    point_.reserve(size_ + 1);

    for (int i = 0; i < size_; ++i) {
        for (int j = i + 1; j < size_; ++j) {
            const double d2 = boundary.get_dist_sq(positions, i, j);
            if (d2 >= rl2_) {
                continue;
            }
            adjacency[i].push_back(j);
            adjacency[j].push_back(i);
        }
    }

    point_.push_back(0);
    nlist_.clear();
    for (const std::vector<int>& neighbours : adjacency) {
        nlist_.insert(nlist_.end(), neighbours.begin(), neighbours.end());
        point_.push_back(static_cast<int>(nlist_.size()));
    }
    point_size_ = point_.size();
}

template<class T>
std::vector<int> VerletList<T>::get_neighbours(int i) const {
    return std::vector<int>(nlist_.begin() + point_[i], nlist_.begin() + point_[i + 1]);
}

template<class T>
void dump(const T& system, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::app);
    file << system.n_ << std::endl;
    if (system.positions_.rows() == 3) {
        file << "id, x, y, z, vx, vy, vz" << std::endl;
        for (int i = 0; i < system.n_; ++i) {
            file << i << " "
                 << system.positions_(0, i) << " "
                 << system.positions_(1, i) << " "
                 << system.positions_(2, i) << " " << std::endl;
        }
        return;
    }
    if (system.positions_.rows() == 2) {
        file << "id, x, y, vx, vy" << std::endl;
        for (int i = 0; i < system.n_; ++i) {
            file << i << " "
                 << system.positions_(0, i) << " "
                 << system.positions_(1, i) << " " << std::endl;
        }
        return;
    }
    throw std::runtime_error("invalid dimension");
}

template<class T>
void load(T& system, const std::string& filename) {
    std::ifstream file(filename, std::ios::in);
    std::string line;
    std::regex head_pattern{"\\d+"};
    std::smatch matched;
    const int head_lines = 2;
    std::string num;
    int n_particles = 0;
    int total_frame = 0;

    while (file) {
        std::getline(file, line);
        if (std::regex_match(line, matched, head_pattern)) {
            n_particles = std::stoi(line);
            ++total_frame;
            for (int i = 0; i < n_particles; ++i) {
                std::getline(file, line);
            }
        }
    }
    file.close();

    file.open(filename, std::ios::in);
    for (int frame = 0; frame < total_frame - 1; ++frame) {
        for (int row = 0; row < n_particles + head_lines; ++row) {
            std::getline(file, line);
        }
    }

    if (system.positions_.rows() == 3) {
        for (int row = 0; row < n_particles + head_lines; ++row) {
            std::getline(file, line);
            if (row <= 1) {
                continue;
            }
            std::istringstream stream(line);
            stream >> num;
            for (int axis = 0; axis < 3; ++axis) {
                stream >> system.positions_(axis, row - head_lines);
            }
        }
    } else if (system.positions_.rows() == 2) {
        for (int row = 0; row < n_particles + head_lines; ++row) {
            std::getline(file, line);
            if (row <= 1) {
                continue;
            }
            std::istringstream stream(line);
            stream >> num;
            for (int axis = 0; axis < 2; ++axis) {
                stream >> system.positions_(axis, row - head_lines);
            }
        }
    } else {
        throw std::runtime_error("invalid dimension");
    }

    system.rebuild_nlist();
}

template<class T>
void recursive_product(
    const std::vector<std::vector<T>>& arrays,
    std::vector<std::vector<T>>& result,
    int idx = 0
) {
    if (idx == static_cast<int>(arrays.size())) {
        return;
    }
    if (idx == 0) {
        result.reserve(arrays[0].size());
        for (const T& value : arrays[0]) {
            result.push_back(std::vector<T>{value});
        }
    } else {
        std::vector<std::vector<T>> new_result;
        new_result.reserve(result.size() * arrays[idx].size());
        for (std::vector<T> values : result) {
            for (const T& value : arrays[idx]) {
                values.push_back(value);
                new_result.push_back(values);
                values.pop_back();
            }
        }
        result = std::move(new_result);
    }
    recursive_product(arrays, result, idx + 1);
}

template<class T>
std::vector<std::vector<T>> product_nd(const std::vector<std::vector<T>>& arrays) {
    std::vector<std::vector<T>> result;
    recursive_product(arrays, result);
    return result;
}

template<class T>
CellList<T>::CellList(double r_cut, std::vector<double> box, std::vector<bool> is_pbc)
    : dim_{static_cast<int>(box.size())},
      r_cut_{r_cut},
      sc_{1},
      size_{0},
      head_shape_(box.size(), 1),
      box_{std::move(box)},
      is_pbc_{std::move(is_pbc)},
      boundary_{box_, is_pbc_} {
    if (sc_ < 1) {
        sc_ = 1;
    }
    for (int axis = 0; axis < dim_; ++axis) {
        head_shape_[axis] = sc_;
    }
}
