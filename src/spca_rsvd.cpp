// Sparse PCA via regularized SVD (sPCA-rSVD, Shen and Huang 2008, Algorithm 1)
// in RcppEigen. The tuning parameter is the degree of sparsity (number of zero
// loadings), either fixed per component as in Guerra's sPCArSVD.m, or chosen
// by the ad hoc CPEV rule of Shen and Huang (Algorithm 3). X is passed by
// reference (Eigen::Map onto R's memory) and is never copied.
//
// Memory: the residual X - U Vt' is never formed; it is applied implicitly
// (U is n x k, Vt is p x k). The start vector of each component is the leading
// singular vector of the residual, taken from its min(n, p) square Gram matrix.
//
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <algorithm>
#include <vector>

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

namespace {

// The residual R_k = X - U_k Vt_k', never formed.
struct Residual {
  const MapMat& X;
  MatrixXd U, Vt;  // n x m and p x m; only the first k columns are used
  Index k = 0;
  Residual(const MapMat& X_, Index m)
    : X(X_), U(X_.rows(), m), Vt(X_.cols(), m) {}

  void tmul(const VectorXd& u, VectorXd& out) const {        // out = R_k' u
    out.noalias() = X.transpose() * u;
    if (k) out.noalias() -= Vt.leftCols(k) * (U.leftCols(k).transpose() * u);
  }
  void mul(const VectorXd& v, VectorXd& out) const {         // out = R_k v
    out.noalias() = X * v;
    if (k) out.noalias() -= U.leftCols(k) * (Vt.leftCols(k).transpose() * v);
  }
  void deflate(const VectorXd& u, const VectorXd& vt) {
    U.col(k) = u; Vt.col(k) = vt; ++k;
  }
};

// soft thresholding h(y) = sign(y) (|y| - l)_+
inline VectorXd soft(const VectorXd& y, double l) {
  return y.array().sign() * (y.array().abs() - l).max(0.0);
}

// MATLAB: AXu = sort(abs(y)); l = AXu(n_zeros), the n_zeros-th smallest |y|
inline double kth_smallest_abs(const VectorXd& y, int n_zeros) {
  std::vector<double> a(y.size());
  for (Index i = 0; i < y.size(); ++i) a[i] = std::abs(y(i));
  std::nth_element(a.begin(), a.begin() + (n_zeros - 1), a.end());
  return a[n_zeros - 1];
}

// Leading left singular vector u of the residual R, from the Gram matrix of R
// on the smaller side (G0 is the Gram matrix of X on that side).
VectorXd start_vector(const Residual& R, const MatrixXd& G0, bool wide) {
  const MapMat& X = R.X;
  MatrixXd G = G0;
  const Index k = R.k;
  if (k) {
    const auto Uk = R.U.leftCols(k);
    const auto Vk = R.Vt.leftCols(k);
    if (wide) {        // R R' = X X' - X Vk Uk' - Uk Vk' X' + Uk Vk'Vk Uk'
      MatrixXd XV = X * Vk;
      G.noalias() -= XV * Uk.transpose();
      G.noalias() -= Uk * XV.transpose();
      G.noalias() += Uk * (Vk.transpose() * Vk) * Uk.transpose();
    } else {           // R'R = X'X - X'Uk Vk' - Vk Uk'X + Vk Uk'Uk Vk'
      MatrixXd XU = X.transpose() * Uk;
      G.noalias() -= XU * Vk.transpose();
      G.noalias() -= Vk * XU.transpose();
      G.noalias() += Vk * (Uk.transpose() * Uk) * Vk.transpose();
    }
  }
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(G);
  const Index top = G.rows() - 1;                    // eigenvalues ascending
  if (wide) return es.eigenvectors().col(top);
  VectorXd Xv(X.rows());
  R.mul(es.eigenvectors().col(top), Xv);
  return Xv / Xv.norm();
}

// Algorithm 1 for one component of the residual R, from the start vector u0,
// with nz zero loadings. Returns u (unit) and v~ (not normalised).
struct Fit {
  VectorXd u, v;
  double l;
  int iter;
};

Fit fit_component(const Residual& R, const VectorXd& u0, int nz,
                  int iter_max, double eps) {
  Fit f;
  f.u = u0;
  VectorXd Xu(R.X.cols()), Xv(R.X.rows()), u_old;
  R.tmul(f.u, Xu);                                   // R'u = d1 v1
  // threshold for nz zero loadings (l = 0 when nz = 0)
  f.l = nz > 0 ? kth_smallest_abs(Xu, nz) : 0.0;

  // alternate (a) v = h_l(R'u) and (b) u = R v / ||R v||
  f.iter = 0;
  double dif = 1.0;
  while (dif > eps && f.iter < iter_max) {
    ++f.iter;
    f.v = soft(Xu, f.l);                             // (a); Xu = R'u_old
    if (nz > 0 && (f.v.array() == 0.0).count() != nz) {
      f.l = kth_smallest_abs(Xu, nz);
      f.v = soft(Xu, f.l);
    }
    R.mul(f.v, Xv);                                  // (b)
    u_old = f.u;
    f.u = Xv / Xv.norm();
    R.tmul(f.u, Xu);                                 // R'u_new, reused in (a)
    dif = (f.u - u_old).norm();
  }
  return f;
}

// CPEV of the first k loading vectors (Shen and Huang 2008, Section 2.3):
// tr(X_k'X_k) / tr(X'X), with X_k = X V (V'V)^{-1} V'
double cpev(const MapMat& X, const MatrixXd& V, double totvar) {
  MatrixXd XV = X * V;
  MatrixXd M = XV.transpose() * XV;
  MatrixXd VtV = V.transpose() * V;
  return VtV.ldlt().solve(M).trace() / totvar;
}

} // namespace

// n_zeros: zero loadings per component (fixed). If grid is not empty, the
// number of zeros of each component is chosen from grid by the ad hoc rule:
// the largest j whose CPEV is at least (1 - drop) times the CPEV at j = 0.
// [[Rcpp::export(.spca_rsvd_cpp)]]
Rcpp::List spca_rsvd_cpp(const Eigen::Map<Eigen::MatrixXd> X,
                         const Eigen::Map<Eigen::VectorXi> n_zeros,
                         const Eigen::Map<Eigen::VectorXi> grid, double drop,
                         int iter_max, double eps) {
  const Index n = X.rows(), p = X.cols(), m = n_zeros.size();
  const Index ng = grid.size();
  const bool wide = n <= p;
  // Gram matrix of X on the smaller side, used for the start vectors
  MatrixXd G0 = wide ? MatrixXd(X * X.transpose())
                     : MatrixXd(X.transpose() * X);
  const double totvar = G0.trace();                  // tr(X'X)
  Residual R(X, m);
  MatrixXd V(p, m), cpev_grid(ng, m);
  VectorXd lambda(m), d(m), cpev_k(m);
  Rcpp::IntegerVector iterations(m), zeros(m);

  for (Index c = 0; c < m; ++c) {
    // 1. start: leading singular vector of the residual
    const VectorXd u0 = start_vector(R, G0, wide);
    int nz = n_zeros(c);
    if (ng) {
      // Algorithm 3: CPEV of the first c + 1 PCs for each j in grid
      // (grid[0] = 0 is the peak value)
      MatrixXd Vc(p, c + 1);
      Vc.leftCols(c) = V.leftCols(c);
      for (Index g = 0; g < ng; ++g) {
        Fit f = fit_component(R, u0, grid(g), iter_max, eps);
        Vc.col(c) = f.v / f.v.norm();
        cpev_grid(g, c) = cpev(X, Vc, totvar);
      }
      nz = grid(0);
      for (Index g = 1; g < ng; ++g)
        if (cpev_grid(g, c) >= (1.0 - drop) * cpev_grid(0, c)) nz = grid(g);
    }
    // 2. Algorithm 1 with nz zero loadings
    Fit f = fit_component(R, u0, nz, iter_max, eps);

    // 3. unit-length loading vector; deflate with the rank-one fit u v~'
    d(c) = f.v.norm();
    V.col(c) = f.v / d(c);
    lambda(c) = f.l;
    iterations[c] = f.iter;
    zeros[c] = nz;
    cpev_k(c) = cpev(X, V.leftCols(c + 1), totvar);
    R.deflate(f.u, f.v);
  }
  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = R.U,
                            Rcpp::Named("d") = d,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("n_zeros") = zeros,
                            Rcpp::Named("cpev") = cpev_k,
                            Rcpp::Named("cpev_grid") = cpev_grid,
                            Rcpp::Named("iterations") = iterations);
}
