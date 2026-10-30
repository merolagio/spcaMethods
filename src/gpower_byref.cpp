// GPower (Journee, Nesterov, Richtarik, Sepulchre 2010) in RcppEigen, with the
//   data matrix
// passed by reference: A is an Eigen::Map onto R's memory, so it is never 
//   copied.
//
// Memory: besides A itself, the largest objects are n x m and p x m. The single
  //-unit
// deflation A <- A - y z' is never formed; the deflated matrix is applied
//  implicitly as
// A - Y Z' (Y is p x k, Z is n x k), so no p x n copy is needed.
// Behaviour matches GPower.m except for the fixes listed in README.md.
//
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> BoolMat;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

namespace {

// MATLAB: iter>2 && (f(iter)-f(iter-1))/f(iter-1)<eps || iter>iter_max  (&& 
//  binds tighter)
inline bool stop_rule(double f, double f_prev, int iter, int iter_max, double eps, bool abs_rel) {
  double rel = abs_rel ? std::abs(f - f_prev) / std::abs(f_prev) : 
  (f - f_prev) / f_prev;
  return (iter > 2 && rel < eps) || iter > iter_max;
}

// polar factor U V' of the thin SVD of G (p x m, m small)
inline MatrixXd polar(const MatrixXd& G) {
  Eigen::JacobiSVD<MatrixXd> svd(G, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU() * svd.matrixV().transpose();
}

inline void normalize_cols(MatrixXd& Z) {
  for (Index i = 0; i < Z.cols(); ++i) {
    double nz = Z.col(i).norm();
    if (nz > 0) Z.col(i) /= nz;
  }
}

// The deflated matrix A_k = A - Y_k Z_k' of the single-unit algorithm, never formed.
struct Deflated {
  const MapMat& A;
  MatrixXd Y, Z;   // p x m and n x m; only the first k columns are used
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
  MatrixXd cols(const std::vector<Index>& idx) const { // A_k(:, idx), 
    //copied (|idx| is small)
    MatrixXd out(A.rows(), idx.size());
    for (size_t j = 0; j < idx.size(); ++j) out.col(j) = col(idx[j]);
    return out;
  }
  void deflate(const VectorXd& y, const VectorXd& z) { Y.col(k) = y; 
    Z.col(k) = z; ++k; }
};

// pattern_filling, single-unit case: power iteration on A_red' A_red on the support
VectorXd fill_single(const Deflated& D, const BoolMat& P, const VectorXd& z0,
                     int iter_max = 1000, double eps = 1e-6) {
  std::vector<Index> support;
  for (Index i = 0; i < P.rows(); ++i) if (P(i, 0)) support.push_back(i);
  VectorXd z = VectorXd::Zero(P.rows());
  if (support.size() == 1) { z(support[0]) = 1.0; return z; }
  if (support.empty()) return z;
  MatrixXd Ar = D.cols(support);
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

// pattern_filling, block case (w = mu, or all ones)
void fill_block(const MapMat& A, const BoolMat& P, MatrixXd& Z, 
                const VectorXd& w,
                int iter_max = 1000, double eps = 1e-6) {
  MatrixXd AZ(A.rows(), Z.cols());
  double f, f_prev = 0;
  for (int iter = 1;; ++iter) {
    AZ.noalias() = A * Z;
    AZ = AZ * w.asDiagonal();
    MatrixXd X = polar(AZ);
    f = X.cwiseProduct(AZ).sum();
    Z.noalias() = A.transpose() * X;
    Z = P.select(Z * w.asDiagonal(), 0.0);
    normalize_cols(Z);
    if (stop_rule(f, f_prev, iter, iter_max, eps, true)) break;
    f_prev = f;
  }
}

} // namespace

// [[Rcpp::export(.gpower_single)]]
Rcpp::List gpower_single(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXd> rho,
                         int m, bool l1,
                         int iter_max, double eps) {
  const Index p = A.rows(), n = A.cols();
  Deflated D(A, m);
  VectorXd thresholds(m), Ax(n), tr(n), grad(p), x(p), y(p), z(n);
  Rcpp::IntegerVector iters(m);

  for (int comp = 0; comp < m; ++comp) {
    Index i_max;
    double rho_max = D.col_norms().maxCoeff(&i_max);
    double r = l1 ? rho(comp) * rho_max : rho(comp) * rho_max * rho_max;
    thresholds(comp) = r;
    x = D.col(i_max) / rho_max;
    double f, f_prev = 0;
    int iter = 1;
    for (;; ++iter) {
      D.tmul(x, Ax);
      if (l1) tr = Ax.array().sign() * (Ax.array().abs() - r).max(0.0);
      else    tr = (Ax.array().square() - r).max(0.0);
      f = l1 ? tr.squaredNorm() : tr.sum();
      if (f == 0) break;
      if (!l1) tr = (tr.array() > 0).select(Ax, 0.0); // gradient direction 
      //  for l0
      D.mul(tr, grad);
      x = grad / grad.norm();
      if (stop_rule(f, f_prev, iter, iter_max, eps, false)) break;
      f_prev = f;
    }
    iters[comp] = iter;
    D.tmul(x, Ax);
    if (l1) {
      BoolMat P = (Ax.array().abs() - r) > 0;
      z = Ax.array().sign() * (Ax.array().abs() - r).max(0.0);
      if ((z.array() > 0).any()) z /= z.norm(); // MATLAB: max(abs(z>0))>0 (sic)
      z = fill_single(D, P, z);
      D.mul(z, y);
    } else {
      z = ((Ax.array().square() - r) > 0).select(Ax, 0.0);
      double nz = z.norm();
      if (nz > 0) { z /= nz; y = x * nz; }
      else y.setZero();     // MATLAB returns NaN here
    }
    D.deflate(y, z);
  }
  return Rcpp::List::create(Rcpp::Named("Z") = D.Z, 
                            Rcpp::Named("thresholds") = thresholds,
                            Rcpp::Named("iterations") = iters);
}

// Exported signatures spell out the Eigen types so RcppExports.cpp also compiles inside a package.
// [[Rcpp::export(.gpower_block)]]
Rcpp::List gpower_block(const Eigen::Map<Eigen::MatrixXd> A, 
                        const Eigen::Map<Eigen::VectorXd> rho,
                        const Eigen::Map<Eigen::VectorXd> mu, 
                        const Eigen::Map<Eigen::MatrixXd> G,
                        bool l1, bool matlab_scale, int iter_max, double eps) {
  const Index p = A.rows(), n = A.cols(), m = rho.size();
  const bool all_ones = (mu.array() == 1.0).all();
  const VectorXd w = all_ones ? VectorXd::Ones(m) : VectorXd(mu);

  // initialisation: [x, R] = qr([a_imax / ||a_imax||, G], 0)
  Index i_max;
  const double max_norm = A.colwise().norm().maxCoeff(&i_max);
  MatrixXd x(p, m);
  x << A.col(i_max) / max_norm, G;
  Eigen::HouseholderQR<MatrixXd> qr(x);
  x = qr.householderQ() * MatrixXd::Identity(p, m);

  // thresholds: paper's bound by default, MATLAB's RHO * R (QR factor) if asked
  MatrixXd scale = matlab_scale ? 
  MatrixXd(qr.matrixQR().topRows(m).triangularView<Eigen::Upper>())
  : MatrixXd(MatrixXd::Identity(m, m) * max_norm);
  Eigen::RowVectorXd rr = rho.transpose(), wt = w.transpose(), RHO;
  if (l1)            RHO = rr.cwiseProduct(wt) * scale;
  else if (all_ones) RHO = rr * (scale * scale);
  else               RHO = rr.cwiseProduct((wt * scale).cwiseAbs2());

  MatrixXd Ax(n, m), T(n, m), grad(p, m);
  double f, f_prev = 0;
  int iter = 1;
  for (;; ++iter) {
    Ax.noalias() = A.transpose() * x;
    Ax = Ax * w.asDiagonal();
    if (l1) T = (Ax.array().abs().rowwise() - RHO.array()).max(0.0);
    else    T = (Ax.array().square().rowwise() - RHO.array()).max(0.0);
    f = l1 ? T.squaredNorm() : T.sum();
    if (f == 0) break;
    if (l1) T = T.array() * Ax.array().sign();
    else    T = (T.array() > 0).select(Ax, 0.0);
    grad.noalias() = A * T; // one BLAS-3 style product
    x = polar(grad * w.asDiagonal());
    if (stop_rule(f, f_prev, iter, iter_max, eps, false)) break;
    f_prev = f;
  }

  Ax.noalias() = A.transpose() * x;
  Ax = Ax * w.asDiagonal();
  MatrixXd Z(n, m);
  if (l1) {
    T = (Ax.array().abs().rowwise() - RHO.array());
    BoolMat P = T.array() > 0;
    Z = Ax.array().sign() * T.array().max(0.0);
    normalize_cols(Z);
    fill_block(A, P, Z, w);
  } else {
    BoolMat P = (Ax.array().square().rowwise() - RHO.array()) > 0;
    Z.noalias() = A.transpose() * x;
    Z = P.select(Z, 0.0);
    normalize_cols(Z);
  }
  return Rcpp::List::create(
    Rcpp::Named("Z") = Z, 
    Rcpp::Named("thresholds") = VectorXd(RHO.transpose()),
    Rcpp::Named("iterations") = iter);
}
