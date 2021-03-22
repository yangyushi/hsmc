#ifndef HARD_SPHERE
#define HARD_SPHERE
#include <regex>
#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using Coord3D = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;  // (3, n)
using Vec3D = Eigen::Vector3d;  // (3, 1)
using Vec2D = Eigen::Vector2d;  // (2, 1)
using IndexPair = array<int, 2>;
using IndexTriplet = array<int, 3>;

/*
 * T should be a class for a the positions of many single particle
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

template<class T>
class VerletList{
    /*
     * Using the Verlet list to accelerate distance calculation with a
     * cutoff for 3D simulation
     * This is suitable for accelerating the simulation without a box
     */
    private:
        double rc_;
        double rc2_;
        double rl_;
        double rl2_;
        int point_size_ = 0;  // particle number + 1, point_.size()
        int size_ = 0;  // particle number

    public:
        VerletList(double r_cut, double r_skin);
        void build(const T& positoins);
        void build(const T& positoins, const PBC<T>& boundary);
        vector<int> nlist_;  // a collection of all the neighbour indices
        vector<int> point_;  // nlist_[ point_[i] ] -> the first neighbour of particle i
        double dr_sq_;
};

template<class T>
void dump(const T& system, string filename);

template<class T>
void load(T& system, string filename);


/*
 * Perform standard MC simulation between two hard walls
 *   for hard spheres with a diameter of 1
 */
class HSConfined{
    public:
        HSConfined(int n, vector<double> box, double step);
        void remove_overlap(); // try to remove all the overlapped particles
        void fill_ideal_gas(); // radomly fill the box with ideal gas 
        bool advance(int idx);  // try to move 1 paticle, return if succeed
        void fill_hs(); // radomly fill the box with hard spheres
        void sweep();  // try to make N movements
        void rebuild();  // rebuild the neighbour list
        void crush(double target_vf, double delta_vf);  // rescale box to the target vf
        int dim_ = 3;
        int n_;
        vector<double> box_;
        Coord3D positions_;
        inline double get_vf() const{  // get the volumn fraction
            return (double) n_ / boundary_.volumn_ / 6 * M_PI;
        }
    private:
        double step_;  // maximum random movement
        vector<bool> is_pbc_{true, true, false};
        PBC<Coord3D> boundary_;
        Coord3D total_disp_;  // total diplacements of each particle
        vector<int> rand_indices_;
        VerletList<Coord3D> vlist_{1.0, 5.0};
        int ldi_ = 0;  // largest displacement index
        vector<IndexPair> get_overlap_indices();
        // randomly change the order of movement during a sweep
        void shuffle_indices();
        // check if one particle is overlapping
        bool check_overlap(int idx);
        // check if overlap exist at all
        bool check_overlap();
        // check if the verlet list needs update, after moveing a single particle
        void check_disp_sum(int idx, const Vec3D& disp);
        void advance_no_check(int idx);  // move 1 paticle
        void adjust_step(int accept_number);
};

/*
 * Perform standard MC simulation in PBCs for hard spheres with a diameter of 1
 */
class HSMC{
    public:
        HSMC(int n, vector<double> box, double step);
};


#include "hard_sphere.tpp"
#endif
