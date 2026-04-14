#include "hard_sphere.hpp"

namespace hsmc {

std::vector<int> unravel_index(int index, const std::vector<int>& shape) {
    const int dim = static_cast<int>(shape.size());
    std::vector<int> result;
    result.reserve(dim);

    for (int d1 = 0; d1 < dim; ++d1) {
        int size = 1;
        for (int d2 = d1 + 1; d2 < dim; ++d2) {
            size *= shape[d2];
        }
        const int value = static_cast<int>(std::floor(index / size));
        result.push_back(value);
        index -= value * size;
    }
    return result;
}

HSMC::HSMC(
    int n, std::vector<double> box,
    std::vector<bool> is_pbc, std::vector<bool> is_hard,
    double r_skin
) : n_{n},
    box_{std::move(box)},
    positions_{dim_, n},
    boundary_{box_, is_pbc},
    total_disp_{dim_, n},
    is_pbc_{std::move(is_pbc)},
    is_hard_{std::move(is_hard)},
    vlist_{1.0, r_skin},
    rng_{std::random_device{}()},
    unit_dist_{0.0, 1.0},
    symmetric_dist_{-1.0, 1.0} {
    rand_indices_.resize(n_);
    std::iota(rand_indices_.begin(), rand_indices_.end(), 0);
    total_disp_.setZero();
    fill_ideal_gas();
    for (int axis = 0; axis < dim_; ++axis) {
        if (is_hard_[axis]) {
            hard_dim_.push_back(axis);
        }
    }
}

Vec3D HSMC::random_position() {
    Vec3D position;
    for (int axis = 0; axis < dim_; ++axis) {
        position(axis) = unit_dist_(rng_) * box_[axis];
    }
    return position;
}

Vec3D HSMC::random_displacement() {
    Vec3D displacement;
    for (int axis = 0; axis < dim_; ++axis) {
        displacement(axis) = symmetric_dist_(rng_) * step_ * 0.5;
    }
    return displacement;
}

void HSMC::shuffle_indices() {
    std::shuffle(rand_indices_.begin(), rand_indices_.end(), rng_);
}

void HSMC::fill_ideal_gas() {
    for (int i = 0; i < n_; ++i) {
        positions_.col(i) = random_position();
    }
}

void HSMC::fill_hs() {
    if (get_vf() > 0.3) {
        throw std::runtime_error(
            "Initial Volumn Fraction > 0.3, can't randomly initialise"
        );
    }

    for (int i : rand_indices_) {
        bool is_overlap = true;
        while (is_overlap) {
            is_overlap = false;
            positions_.col(i) = random_position();
            for (int j = 0; j < i; ++j) {
                if (boundary_.get_dist_sq(positions_, i, j) < 1.0) {
                    is_overlap = true;
                    break;
                }
            }
        }
    }
    rebuild_nlist();
}

bool HSMC::check_hardwall() {
    for (int axis : hard_dim_) {
        for (int i = 0; i < n_; ++i) {
            if (positions_(axis, i) < 0 || positions_(axis, i) > box_[axis]) {
                return true;
            }
        }
    }
    return false;
}

bool HSMC::check_hardwall(int idx) {
    for (int axis : hard_dim_) {
        if (positions_(axis, idx) < 0 || positions_(axis, idx) > box_[axis]) {
            return true;
        }
    }
    return false;
}

bool HSMC::report_overlap() {
    for (int i = 0; i < n_; ++i) {
        if (check_hardwall(i)) {
            return true;
        }
        for (int j = i + 1; j < n_; ++j) {
            if (boundary_.get_dist_sq(positions_, i, j) < 1.0) {
                return true;
            }
        }
    }
    return false;
}

bool HSMC::check_overlap(int i) {
    if (check_hardwall(i)) {
        return true;
    }

    for (int offset = vlist_.begin_offset(i); offset < vlist_.end_offset(i); ++offset) {
        const int j = vlist_.neighbour_at(offset);
        if (boundary_.get_dist_sq(positions_, i, j) < 1.0) {
            return true;
        }
    }
    return false;
}

bool HSMC::check_overlap() {
    for (int i : rand_indices_) {
        if (check_hardwall(i)) {
            return true;
        }
        for (int offset = vlist_.begin_offset(i); offset < vlist_.end_offset(i); ++offset) {
            const int j = vlist_.neighbour_at(offset);
            if (boundary_.get_dist_sq(positions_, i, j) < 1.0) {
                return true;
            }
        }
    }
    return false;
}

bool HSMC::advance(int idx) {
    const Vec3D previous_pos = positions_.col(idx);
    const Vec3D disp = random_displacement();
    positions_.col(idx) += disp;
    boundary_.fix_position(positions_, idx);
    check_disp_sum(idx, disp);
    if (check_overlap(idx)) {
        check_disp_sum(idx, -disp);
        positions_.col(idx) = previous_pos;
        return false;
    }
    return true;
}

void HSMC::check_disp_sum(int idx, const Vec3D& disp) {
    total_disp_.col(idx) += disp;
    const double disp_sq = total_disp_.col(idx).squaredNorm();
    double disp_sq_max = total_disp_.col(ldi_).squaredNorm();

    if (disp_sq > disp_sq_max) {
        ldi_ = idx;
        disp_sq_max = disp_sq;
    }
    if (disp_sq_max * 4.0 >= vlist_.dr_sq_) {
        rebuild_nlist();
    }
}

void HSMC::adjust_step(int accept_number) {
    if (rand_indices_.empty()) {
        return;
    }

    const double accept_ratio =
        static_cast<double>(accept_number) / rand_indices_.size();
    if (accept_ratio < 0.45) {
        step_ *= 0.95;
    } else if (accept_ratio > 0.55) {
        step_ *= 1.05;
    }
}

void HSMC::sweep() {
    shuffle_indices();
    int accept_number = 0;
    for (int idx : rand_indices_) {
        if (advance(idx)) {
            ++accept_number;
        }
    }
    adjust_step(accept_number);
}

std::vector<std::array<int, 2>> HSMC::get_overlap_indices() {
    std::vector<std::array<int, 2>> overlap_indices;
    for (int i = 0; i < n_; ++i) {
        for (int offset = vlist_.begin_offset(i); offset < vlist_.end_offset(i); ++offset) {
            const int j = vlist_.neighbour_at(offset);
            if (boundary_.get_dist_sq(positions_, i, j) < 1.0) {
                overlap_indices.push_back(std::array<int, 2>{{i, j}});
            }
        }
    }
    return overlap_indices;
}

void HSMC::remove_overlap() {
    while (check_overlap()) {
        sweep();
    }
}

void HSMC::crush(double target_vf, double delta_vf) {
    double vf = get_vf();
    while (vf < target_vf) {
        const double vf_new = (target_vf - vf < delta_vf) ? target_vf : vf + delta_vf;
        const double scale = std::pow(vf / vf_new, 1.0 / 3.0);
        positions_.array() *= scale;
        boundary_.rescale(scale);
        box_ = boundary_.box_;
        boundary_.fix_position(positions_);
        rebuild_nlist();
        remove_overlap();
        vf = vf_new;
        std::cout << "Crushed to higher volume fraction, step: " << step_
                  << "; vf: " << get_vf() * 100 << std::endl;
    }

    std::cout << "final box size: ";
    for (int d = 0; d < dim_; ++d) {
        std::cout << box_[d];
        if (d != dim_) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

void HSMC::crush_along_axis(double target_vf, double delta_vf, int axis) {
    double vf = get_vf();
    while (vf < target_vf) {
        const double vf_new = (target_vf - vf < delta_vf) ? target_vf : vf + delta_vf;
        const double scale = vf / vf_new;

        positions_.row(axis).array() *= scale;
        boundary_.rescale(scale, axis);
        box_ = boundary_.box_;

        boundary_.fix_position(positions_);
        rebuild_nlist();
        remove_overlap();

        vf = vf_new;
        std::cout << "Crushed to higher volume fraction, step: " << step_
                  << "; vf: " << get_vf() * 100 << std::endl;
    }

    std::cout << "final box size: ";
    for (int d = 0; d < dim_; ++d) {
        std::cout << box_[d];
        if (d != dim_) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

std::string HSMC::str() const {
    std::ostringstream str_stream;
    const std::array<std::string, 3> side_names{{"X", "Y", "Z"}};
    str_stream << "Hard Sphere MC Simulaion, with periodic boundary on ";
    for (int d = 0; d < dim_; ++d) {
        if (is_pbc_[d]) {
            str_stream << side_names[d];
        }
    }
    str_stream << " sides" << std::endl;
    str_stream << "N = " << n_ << "; Box = (" << std::setprecision(8);
    for (int d = 0; d < dim_; ++d) {
        str_stream << box_[d];
        if (d < 2) {
            str_stream << ", ";
        }
    }
    str_stream << "); Volumn Fraction = " << get_vf() << std::endl;
    return str_stream.str();
}

std::string HSMC::repr() const {
    std::ostringstream str_stream;
    const std::array<std::string, 3> side_names{{"X", "Y", "Z"}};
    str_stream << "Hard Sphere MC Simulaion, with periodic boundary on ";
    for (int d = 0; d < dim_; ++d) {
        if (is_pbc_[d]) {
            str_stream << side_names[d];
        }
    }
    str_stream << " sides" << std::endl;
    str_stream << "N = " << n_ << "; Box = (" << std::setprecision(8);
    for (int d = 0; d < dim_; ++d) {
        str_stream << box_[d];
        if (d < 2) {
            str_stream << ", ";
        }
    }
    str_stream << "); Volumn Fraction = " << get_vf() << std::endl;
    str_stream << "(address: " << std::hex << &positions_ << ")" << std::endl;
    return str_stream.str();
}

}  // namespace hsmc
