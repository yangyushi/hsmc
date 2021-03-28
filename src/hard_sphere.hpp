#ifndef HARD_SPHERE
#define HARD_SPHERE
#include <regex>
#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using Coord3D = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;  // (3, n)
using CellIndices = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec3D = Eigen::Vector3d;  // (3, 1)
using Vec2D = Eigen::Vector2d;  // (2, 1)


/*
 * Calculate the N-Dimensional Cartesian product
 *
 * Example:
 *      product_nd([[1, 2], [3, 4]]) -> [[1, 3], [1, 4], [2, 3], [2, 4]]
 */
template<class T>
vector<vector<T>> product_nd(const vector<vector<T>>& arrays);


/*
* Bheaving like ``numpy.unravel_index`` with 'C' order (last axis changing fastest)
*/
vector<int> unravel_index(int index, const vector<int>& shape);


/*
 * Applying periodic boundary conditions in any dimension, the pbc
 *      in each dimension can be turned off
 * The dimension is inferred from class T, which is an Eigen matrix
 *      with shape (dimension, n_particles)
 */
template<class T>
class PBC{
    public:
        PBC(vector<double> box, vector<bool> is_pbc)
            : box_{box}, is_pbc_{is_pbc}
        {
            if (box_.size() != is_pbc_.size()) {
                throw("PBC box size mismatch");
            }
            update_volumn();
            dim_ = box_.size();
        }
        vector<double> box_;
        void fix_position(T& positions) const;
        void fix_position(T& positions, int i) const;
        double get_dist_sq(const T& positions, int i, int j) const;
        // rescale the box homogeneously
        void rescale(double scale){
            for (auto& side : box_) {
                side *= scale;
            }
            this->update_volumn();
        }
        double volumn_;
    private:
        int dim_;
        void update_volumn();
        vector<bool> is_pbc_;
};

/*
 * Using the Verlet list to accelerate distance calculation with a
 *     cutoff for simulation in all dimensions
 * The dimension is inferred from class T, which is an Eigen matrix
 *      with shape (dimension, n_particles)
 */
template<class T>
class VerletList{
    public:
        VerletList(double r_cut, double r_skin);
        /*
         * Build Verlet list without any periodic boundary
         */
        void build(const T& positions);
        /*
         * Build Verlet list while applying PBC for distance calculation
         */
        void build(const T& positions, const PBC<T>& boundary);
        /*
         * For particle [i], retrieve the indices of neighbour particles
         */
        vector<int> get_neighbours(int i);
        double dr_sq_;  // (r_skin - r_cut) ^ 2 for auto-update
    private:
        vector<int> nlist_;  // a collection of all the neighbour indices
        vector<int> point_;  // nlist_[ point_[i] ] -> the first neighbour of particle i
        double rc_;
        double rc2_;
        double rl_;
        double rl2_;
        int point_size_ = 0;  // particle number + 1, point_.size()
        int size_ = 0;  // particle number
};


/*
 * Using the Cell linked list to accelerate the distance calculation
 *      for simulation in all dimensions
 * The dimension is inferred from class T, which is an Eigen matrix
 *      with shape (dimension, n_particles)
 */
template<class T>
class CellList{
    public:
        CellList(double r_cut, vector<double> box, vector<bool> is_pbc);
        void build(const T& positions);
    private:
        PBC<T> boundadry_;
};


/*
 * Dump the current phase point to an xyz file
 * It works with both 2D and 3D system
 */
template<class T>
void dump(const T& system, string filename);


/*
 * Load the phase point from the *last* frame of an xyz file, then rebuild the
 *      neighbour list.
 * It works with both 2D and 3D system, and the xyz file should be generated by
 *      the `dump` function
 */
template<class T>
void load(T& system, string filename);


/*
 * Hard Sphere Monte-Carlo simulation without neighbout lists
 */
class HSMC{
    public:
        HSMC(int n, vector<double> box, vector<bool> is_pbc);
        int dim_ = 3;
        int n_;
        vector<double> box_;
        Coord3D positions_;

        void fill_ideal_gas(); // radomly fill the box with ideal gas 
        void fill_hs(); // radomly fill the box with hard spheres
        void sweep();  // try to make N movements
        void crush(double target_vf, double delta_vf);  // rescale box to the target vf
        inline double get_vf() const{  // get the volumn fraction
            return (double) n_ / boundary_.volumn_ / 6 * M_PI;
        }
        // rebuild the neighbour list
        inline void rebuild_nlist(){
            vlist_.build(positions_, boundary_);
            total_disp_.setZero();
        }
        /*
         * these functions were created for the python end
         */
        Coord3D& get_positions() {return positions_;}
        const Coord3D& view_positions() const {return positions_;}
        void load_positions(Coord3D positions){
            positions_ = positions;
            this->rebuild_nlist();
        }
        string repr() const;
        string str() const;
    private:
        double step_;  // maximum random movement
        PBC<Coord3D> boundary_;
        Coord3D total_disp_;  // total diplacements of each particle
        vector<bool> is_pbc_;
        vector<int> rand_indices_;
        VerletList<Coord3D> vlist_{1.0, 3.0};
        int ldi_ = 0;  // largest displacement index
        vector<array<int, 2>> get_overlap_indices();
        vector<int> get_neighbours(int i){
            return vlist_.get_neighbours(i);
        }
        // try to move 1 paticle, return if succeed
        bool advance(int idx);
        // try to remove all the overlapped particles
        void remove_overlap(); 
        // randomly change the order of movement during a sweep
        void shuffle_indices();
        // check if one particle is overlapping
        bool check_overlap(int idx);
        // check if overlap exist at all
        bool check_overlap();
        // check if the verlet list needs update, after moveing a single particle
        void check_disp_sum(int idx, const Vec3D& disp);
        void adjust_step(int accept_number);
};


#include "hard_sphere.tpp"
#endif
