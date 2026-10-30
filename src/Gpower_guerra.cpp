// GPower2.m of Guerra-Urzola et al. (2024), github.com/RosemberGuerra/sparsePCA, in
// RcppEigen: GPower.m single-unit l1 algorithm with deflation, where component k starts
// at a given vector U[, k] (they use the PCA left singular vectors) instead of the
// largest-norm column. Same iteration, stopping rule and pattern filling as GPower.m and
// gpower_byref.cpp. A is passed by reference (Eigen::Map) and the deflation
// A - Y Z' is applied implicitly, so A is never copied.
//
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolVec;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

namespace {

// MATLAB: iter>2 && (f(iter)-f(iter-1))/f(iter-1)<eps || iter>iter_max
inline bool stop_rule(double f, double f_prev, int iter, int iter_max, double eps, bool abs_rel) {
  double rel = abs_rel ? std::abs(f - f_prev) / std::abs(f_prev) : (f - f_prev) / f_prev;
  return (iter > 2 && rel < eps) || iter > iter_max;
}

// deflated matrix A_k = A - Y_k Z_k', never formed
struct Deflated {
  const MapMat& A;
  MatrixXd Y, Z;
  Index k = 0;
  Deflated(const MapMat& A_, Index m) : A(A_), Y(A_.rows(), m), Z(A_.cols(), m) {}
  void tmul(const VectorXd& x, VectorXd& out) const {        // out = A_k' x
    out.noalias() = A.transpose() * x;
    if (k) out.noalias() -= Z.leftCols(k) * (Y.leftCols(k).transpose() * x);
  }
  void mul(const VectorXd& v, VectorXd& out) const {         // out = A_k v
    out.noalias() = A * v;
    if (k) out.noalias() -= Y.leftCols(k) * (Z.leftCols(k).transpose() * v);
  }
  VectorXd col(Index i) const {                              // A_k(:, i)
    VectorXd c = A.col(i);
    if (k) c.noalias() -= Y.leftCols(k) * Z.row(i).head(k).transpose();
    return c;
  }
  VectorXd col_norms() const {
    if (!k) return A.colwise().norm().transpose();
    VectorXd nrm(A.cols());
    for (Index i = 0; i < A.cols(); ++i) nrm(i) = col(i).norm();
    return nrm;
  }
  void deflate(const VectorXd& y, const VectorXd& z) { Y.col(k) = y; Z.col(k) = z; ++k; }
};

// pattern_filling, single-unit case: power iteration on the support
VectorXd fill_single(const Deflated& D, const BoolVec& P, const VectorXd& z0,
                     int iter_max = 1000, double eps = 1e-6) {
  std::vector<Index> support;
  for (Index i = 0; i < P.size(); ++i) if (P(i)) support.push_back(i);
  VectorXd z = VectorXd::Zero(P.size());
  if (support.size() == 1) { z(support[0]) = 1.0; return z; }
  if (support.empty()) return z;
  MatrixXd Ar(D.A.rows(), support.size());
  for (size_t j = 0; j < support.size(); ++j) Ar.col(j) = D.col(support[j]);
  VectorXd u(support.size()), tmp(support.size()), Au(Ar.rows());
  for (size_t j = 0; j < support.size(); ++j) u(j) = z0(support[j]);
  double f, f_prev = 0;
  for (int iter = 1;; ++iter) {
    Au.noalias() = Ar * u;
    tmp.noalias() = Ar.transpose() * Au;
    u = tmp / tmp.norm();
    f = -2.0 * u.dot(tmp);
    if (stop_rule(f, f_prev, iter, iter_max, eps, true)) break;
    f_prev = f;
  }
  for (size_t j = 0; j < support.size(); ++j) z(support[j]) = u(j);
  return z;
}

} // namespace

// A: p x n data (observations in rows); rho: relative l1 parameters, one per component;
// U: p x m start vectors, column k starts component k.
// [[Rcpp::export]]
Rcpp::List gpower_guerra(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXd> rho,
                         const Eigen::Map<Eigen::MatrixXd> U,
                         int iter_max = 1000, double eps = 1e-4) {
  const Index p = A.rows(), n = A.cols(), m = U.cols();
  if (U.rows() != p) Rcpp::stop("U must have nrow(A) rows");
  if (rho.size() != m) Rcpp::stop("rho must have one value per column of U");
  Deflated D(A, m);
  VectorXd thresholds(m), Ax(n), tr(n), grad(p), x(p), y(p), z(n);
  Rcpp::IntegerVector iters(m);

  for (Index comp = 0; comp < m; ++comp) {
    const double r = rho(comp) * D.col_norms().maxCoeff();
    thresholds(comp) = r;
    x = U.col(comp) / U.col(comp).norm();                   // GPower2.m: x = V(:, comp)
    double f, f_prev = 0;
    int iter = 1;
    for (;; ++iter) {
      D.tmul(x, Ax);
      tr = Ax.array().sign() * (Ax.array().abs() - r).max(0.0);
      f = tr.squaredNorm();
      if (f == 0) break;
      D.mul(tr, grad);
      x = grad / grad.norm();
      if (stop_rule(f, f_prev, iter, iter_max, eps, false)) break;
      f_prev = f;
    }
    iters[comp] = iter;
    D.tmul(x, Ax);
    BoolVec P = (Ax.array().abs() - r) > 0;
    z = Ax.array().sign() * (Ax.array().abs() - r).max(0.0);
    if ((z.array() > 0).any()) z /= z.norm();               // MATLAB: max(abs(z>0))>0 (sic)
    z = fill_single(D, P, z);
    D.mul(z, y);
    D.deflate(y, z);
  }
  return Rcpp::List::create(Rcpp::Named("Z") = D.Z, Rcpp::Named("thresholds") = thresholds,
                            Rcpp::Named("iterations") = iters);
}
