// Sparse PCA of Zou, Hastie and Tibshirani (2006), RcppEigen port of two implementations:
//
//  * spca_zou_cpp:  elasticnet::spca (Zou & Hastie, R package elasticnet 1.3, predictor
//                   type). Each sparse weight vector is an elastic-net (LARS-EN) fit with
//                   ridge lambda, stopped either at an l1 penalty or at a number of
//                   non-zero weights. SpaSM's spca.m (used by Camacho et al. as
//                   spca_zouhastie) implements the same algorithm.
//  * spca_soft_cpp: spcaZou.m of the MEDA Toolbox v1.8 (Camacho et al.), the ridge = Inf
//                   case, where the elastic net reduces to soft thresholding of X'X a_k.
//
// X is passed by reference (Eigen::Map) and is never copied; X'X is never formed.

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

// ---------------------------------------------------------------- helpers

// solve R' z = b with R upper triangular (lars::backsolvet)
inline VectorXd backsolvet(const MatrixXd& R, const VectorXd& b) {
  return R.transpose().triangularView<Eigen::Lower>().solve(b);
}

// solve R z = b with R upper triangular (backsolve)
inline VectorXd backsolve(const MatrixXd& R, const VectorXd& b) {
  return R.triangularView<Eigen::Upper>().solve(b);
}

// remove column k of the Cholesky factor R and restore the triangular form with
// Givens rotations (lars::downdateR / delcol). Row signs may differ from the Fortran,
// which leaves R'R, and hence every quantity used by LARS, unchanged.
MatrixXd downdateR(const MatrixXd& R, int k) {
  const int p = R.rows();
  if (p == 1) return MatrixXd(0, 0);
  MatrixXd H(p, p - 1);
  H.leftCols(k) = R.leftCols(k);
  H.rightCols(p - 1 - k) = R.rightCols(p - 1 - k);
  for (int j = k; j < p - 1; ++j) {
    Eigen::JacobiRotation<double> G;
    G.makeGivens(H(j, j), H(j + 1, j));
    H.applyOnTheLeft(j, j + 1, G.adjoint());
    H(j + 1, j) = 0.0;
  }
  return H.topRows(p - 1);
}

// columns scaled to unit norm, zero columns left at zero
void normalize_cols(MatrixXd& B) {
  for (int j = 0; j < B.cols(); ++j) {
    double nb = B.col(j).norm();
    if (nb == 0.0) nb = 1.0;
    B.col(j) /= nb;
  }
}

// elasticnet::convcheck
double convcheck(const MatrixXd& b1, const MatrixXd& b2) {
  double out = -std::numeric_limits<double>::infinity();
  for (int j = 0; j < b1.cols(); ++j) {
    double a = (b1.col(j) + b2.col(j)).cwiseAbs().maxCoeff();
    double b = (b1.col(j) - b2.col(j)).cwiseAbs().maxCoeff();
    out = std::max(out, std::min(a, b));
  }
  return out;
}

// polar factor U V' of G (m x K)
MatrixXd polar(const MatrixXd& G) {
  Eigen::JacobiSVD<MatrixXd> s(G, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return s.matrixU() * s.matrixV().transpose();
}

// first K right singular vectors of X, from the smaller of XX' and X'X
MatrixXd right_sv(const MapMat& X, int K) {
  const int n = X.rows(), m = X.cols();
  if (n < m) {
    MatrixXd G = X * X.transpose();
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(G);
    MatrixXd V(m, K);
    for (int j = 0; j < K; ++j) {
      const int c = n - 1 - j;  // eigenvalues come in increasing order
      V.col(j) = X.transpose() * es.eigenvectors().col(c);
      V.col(j) /= V.col(j).norm();
    }
    return V;
  }
  MatrixXd G = X.transpose() * X;
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(G);
  return es.eigenvectors().rowwise().reverse().leftCols(K);
}

// PEV of the model X ~ X P (Q'P)^+ Q' (Camacho et al. 2025, eq. 21), total and per
// component, without forming the n x m reconstruction (Q'Q = I)
void pev_pq(const MapMat& X, const MatrixXd& P, const MatrixXd& Q, double totvar,
            double& total, VectorXd& comp) {
  MatrixXd QtP = Q.transpose() * P;
  MatrixXd M = QtP.completeOrthogonalDecomposition().pseudoInverse();
  MatrixXd T = (X * P) * M;            // X W with W = P (Q'P)^+
  MatrixXd XQ = X * Q;
  double cross = (T.cwiseProduct(XQ)).sum();
  total = 1.0 - (totvar - 2.0 * cross + T.squaredNorm()) / totvar;
  comp = T.colwise().squaredNorm().transpose() / totvar;
}

// ---------------------------------------------------------------- LARS-EN

// elasticnet:::solvebeta: elastic-net coefficients of y on x with ridge lambda, stopped
// at the l1 penalty `para` (varnum = false) or at `para` non-zero coefficients
// (varnum = true). The ridge is applied through the implicit augmented design
// [x; sqrt(lambda) I] / sqrt(1 + lambda), never formed.
VectorXd solvebeta(const MapMat& x, const VectorXd& y, double lambda, double para,
                   bool varnum, int max_steps) {
  const int n = x.rows(), m = x.cols();
  const double eps = std::numeric_limits<double>::epsilon();
  int maxvars = (lambda > 0) ? m : ((m == n) ? m : std::min(m, n - 1));
  const double d1 = std::sqrt(lambda), d2 = 1.0 / std::sqrt(1.0 + lambda);
  VectorXd Cvec = (x.transpose() * y) * d2;
  VectorXd r1 = y;                     // residuals, data part
  VectorXd r2 = VectorXd::Zero(m);     // residuals, ridge part
  if (max_steps <= 0) max_steps = 50 * maxvars;
  std::vector<double> penalty(1, Cvec.cwiseAbs().maxCoeff());
  VectorXd beta = VectorXd::Zero(m);
  if (!varnum && penalty[0] * 2.0 / d2 <= para) return beta;

  std::vector<int> active, ignores;
  std::vector<double> Sign;
  std::vector<char> state(m, 0);       // 0 inactive, 1 active, 2 ignored
  MatrixXd R(0, 0);
  bool anydrops = false;
  int k = 0;
  while (k < max_steps && (int)active.size() < maxvars - (int)ignores.size()) {
    ++k;
    std::vector<int> inactive;
    for (int i = 0; i < m; ++i) if (state[i] == 0) inactive.push_back(i);
    double Cmax = 0.0;
    for (int i : inactive) Cmax = std::max(Cmax, std::abs(Cvec(i)));
    std::vector<int> rest;             // inactive variables that stay inactive
    if (!anydrops) {
      std::vector<int> newv;
      for (int i : inactive) {
        if (std::abs(Cvec(i)) == Cmax) newv.push_back(i); else rest.push_back(i);
      }
      for (int inew : newv) {
        const double xtx = (x.col(inew).squaredNorm() + lambda) / (1.0 + lambda);
        if (R.size() == 0) {
          R = MatrixXd::Constant(1, 1, std::sqrt(xtx));
        } else {
          const int na = active.size();
          VectorXd Xtx(na);
          for (int j = 0; j < na; ++j)
            Xtx(j) = x.col(active[j]).dot(x.col(inew)) / (1.0 + lambda);
          VectorXd r = backsolvet(R, Xtx);
          const double rpp = xtx - r.squaredNorm();
          if (rpp <= eps) {            // collinear with the active set: ignore it
            ignores.push_back(inew);
            state[inew] = 2;
            continue;
          }
          MatrixXd R2 = MatrixXd::Zero(na + 1, na + 1);
          R2.topLeftCorner(na, na) = R;
          R2.topRightCorner(na, 1) = r;
          R2(na, na) = std::sqrt(rpp);
          R.swap(R2);
        }
        active.push_back(inew);
        state[inew] = 1;
        Sign.push_back((Cvec(inew) > 0) - (Cvec(inew) < 0));
      }
    } else {
      rest = inactive;
    }
    const int na = active.size();
    VectorXd S = Eigen::Map<VectorXd>(Sign.data(), na);
    VectorXd Gi1 = backsolve(R, backsolvet(R, S));
    const double A = 1.0 / std::sqrt(Gi1.dot(S));
    VectorXd w = A * Gi1;
    VectorXd u1 = VectorXd::Zero(n);
    for (int j = 0; j < na; ++j) u1.noalias() += x.col(active[j]) * w(j);
    u1 *= d2;
    const int nign = ignores.size();
    maxvars = (lambda > 0) ? m - nign : std::min(m - nign, n - 1);
    double gamhat = Cmax / A;
    if (na != maxvars - nign) {        // (sic) as in elasticnet
      for (int i : rest) {
        const double a = x.col(i).dot(u1) * d2;  // ridge part of u is zero off the active set
        const double C = Cvec(i);
        const double g1 = (Cmax - C) / (A - a), g2 = (Cmax + C) / (A + a);
        if (g1 > eps && g1 < gamhat) gamhat = g1;
        if (g2 > eps && g2 < gamhat) gamhat = g2;
      }
    }
    // variables whose coefficient would cross zero (lasso modification)
    double zmin = gamhat;
    VectorXd z1(na);
    for (int j = 0; j < na; ++j) {
      z1(j) = -beta(active[j]) / w(j);
      if (z1(j) > eps && z1(j) < zmin) zmin = z1(j);
    }
    std::vector<char> drops(na, 0);
    anydrops = false;
    if (zmin < gamhat) {
      gamhat = zmin;
      for (int j = 0; j < na; ++j) if (z1(j) == zmin) { drops[j] = 1; anydrops = true; }
    }
    VectorXd beta2 = beta;
    for (int j = 0; j < na; ++j) {
      beta(active[j]) += gamhat * w(j);
      r2(active[j]) -= gamhat * d1 * d2 * w(j);
    }
    r1 -= gamhat * u1;
    Cvec.noalias() = x.transpose() * r1;
    Cvec = (Cvec + d1 * r2) * d2;
    penalty.push_back(penalty.back() - std::abs(gamhat * A));
    if (!varnum && penalty.back() * 2.0 / d2 <= para) {
      const double s1 = penalty.back() * 2.0 / d2;
      const double s2 = penalty[penalty.size() - 2] * 2.0 / d2;
      beta = (s2 - para) / (s2 - s1) * beta + (para - s1) / (s2 - s1) * beta2;
      beta *= d2;
      break;
    }
    if (anydrops) {
      for (int j = na - 1; j >= 0; --j) if (drops[j]) R = downdateR(R, j);
      std::vector<int> a2;
      std::vector<double> s2;
      for (int j = 0; j < na; ++j) {
        if (drops[j]) {
          beta(active[j]) = 0.0;
          state[active[j]] = 0;
        } else {
          a2.push_back(active[j]);
          s2.push_back(Sign[j]);
        }
      }
      active.swap(a2);
      Sign.swap(s2);
    }
    if (varnum && (double)active.size() >= para) break;
  }
  return beta;
}

// ---------------------------------------------------------------- exported

// [[Rcpp::export(.spca_zou_cpp)]]
Rcpp::List spca_zou_cpp(const Eigen::Map<Eigen::MatrixXd> X, int K,
                        const Eigen::Map<Eigen::VectorXd> para, double lambda,
                        bool varnum, int max_iter, double eps_conv, int max_steps) {
  const double totvar = X.squaredNorm();
  MatrixXd alpha = right_sv(X, K);
  MatrixXd beta(X.cols(), K);
  for (int i = 0; i < K; ++i)
    beta.col(i) = solvebeta(X, X * alpha.col(i), lambda, para(i), varnum, max_steps);
  MatrixXd temp = beta;
  normalize_cols(temp);
  int k = 0;
  double diff = 1.0;
  while (k < max_iter && diff > eps_conv) {
    ++k;
    MatrixXd XB = X * beta;
    alpha = polar(X.transpose() * XB);
    for (int i = 0; i < K; ++i)
      beta.col(i) = solvebeta(X, X * alpha.col(i), lambda, para(i), varnum, max_steps);
    MatrixXd beta2 = beta;
    normalize_cols(beta2);
    diff = convcheck(beta2, temp);
    temp.swap(beta2);
  }
  normalize_cols(beta);
  // adjusted variance of Zou et al.: QR of the scores
  MatrixXd U = X * beta;
  Eigen::HouseholderQR<MatrixXd> qr(U);
  MatrixXd Rq = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
  VectorXd pev = Rq.diagonal().array().square() / totvar;
  // orthonormal loadings and PEV_pq of Camacho et al.
  MatrixXd Q = polar(X.transpose() * U);
  double pq;
  VectorXd pq_comp;
  pev_pq(X, beta, Q, totvar, pq, pq_comp);
  return Rcpp::List::create(Rcpp::Named("P") = beta, Rcpp::Named("Q") = Q,
                            Rcpp::Named("pev") = pev, Rcpp::Named("pev_pq") = pq,
                            Rcpp::Named("pev_pq_comp") = pq_comp,
                            Rcpp::Named("iterations") = k, Rcpp::Named("totvar") = totvar);
}

// [[Rcpp::export(.spca_soft_cpp)]]
Rcpp::List spca_soft_cpp(const Eigen::Map<Eigen::MatrixXd> X, int K,
                         const Eigen::Map<Eigen::VectorXd> stop, double tol, int max_iter) {
  const int m = X.cols();
  const double totvar = X.squaredNorm();
  MatrixXd A = right_sv(X, K);
  MatrixXd B(m, K);
  std::vector<double> work(m);
  auto update_B = [&]() {
    for (int k = 0; k < K; ++k) {
      VectorXd AXX = X.transpose() * (X * A.col(k));
      double thr = stop(k);
      if (stop(k) < 0 && -stop(k) < m) {  // keep the -stop(k) largest |AXX|
        const int pos = (int)(-std::floor(stop(k)));   // 0-based index of the (pos+1)-th largest
        for (int i = 0; i < m; ++i) work[i] = std::abs(AXX(i));
        std::nth_element(work.begin(), work.begin() + pos, work.end(), std::greater<double>());
        thr = work[pos];
      }
      B.col(k) = AXX.array().sign() * (AXX.array().abs() - thr).max(0.0);
      double nb = B.col(k).norm();
      if (nb == 0.0) nb = 1.0;
      B.col(k) /= nb;
    }
  };
  update_B();
  int iter = 0;
  while (true) {
    MatrixXd XB = X * B;
    A = polar(X.transpose() * XB);
    MatrixXd Bold = B;
    update_B();
    ++iter;
    MatrixXd D = B - Bold;             // MATLAB norm(): largest singular value
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(D.transpose() * D, Eigen::EigenvaluesOnly);
    const double nd = std::sqrt(std::max(0.0, es.eigenvalues().maxCoeff()));
    if (nd < tol || iter > max_iter) break;
  }
  double pq;
  VectorXd pq_comp;
  pev_pq(X, B, A, totvar, pq, pq_comp);
  return Rcpp::List::create(Rcpp::Named("P") = B, Rcpp::Named("Q") = A,
                            Rcpp::Named("pev_pq") = pq, Rcpp::Named("pev_pq_comp") = pq_comp,
                            Rcpp::Named("iterations") = iter, Rcpp::Named("totvar") = totvar);
}
