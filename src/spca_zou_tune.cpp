// Sparse PCA of Zou, Hastie and Tibshirani (2006) with the number of non-zero
// weights of each component chosen by a stopping rule (sPCA_Z and sPCA_C).
//
// Rule, component by component (k = 1, ..., K), the cardinalities of the
// earlier components kept: the smallest cardinality c_k such that
//   PEV(k) >= alpha * PEV of the first k PCs (PCA),
// with PEV either Zou's adjusted variance (sPCA_Z) or PEV_pq of Camacho et al.
// (2025, sPCA_C). Search:
//  (a) candidate from one LARS-EN path with alpha_k fixed: the PEV is evaluated
//      after every step and the path stops at the first c meeting the rule;
//  (b) confirmation with full fits (alternating algorithm to convergence),
//      stepping down (or up) from the candidate by 1, 2, 4, ... and then
//      bisecting, to find the smallest c meeting the rule;
//  (c) every full fit starts from the alpha of the previous one (warm start).
//
// The file also has zSPCA_seq, the sequential sparse PCA of Sjostrand et al.
// (2018, Algorithm 4), with the cardinality either given (as in SpaSM) or
// chosen by the same rule (sPCA_M).
//
// The LARS-EN code (solvebeta and helpers) is copied from spca_zou.cpp, inside
// an unnamed namespace so that both files can live in the same package.
// X is passed by reference (Eigen::Map) and is never copied; X'X is never
// formed.

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

namespace {

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

// called after each LARS-EN step with the current coefficients and the number
// of active variables; returning true stops the path (added for the tuning rule)
typedef std::function<bool(const VectorXd&, int)> StepFn;

// elasticnet:::solvebeta: elastic-net coefficients of y on x with ridge lambda, stopped
// at the l1 penalty `para` (varnum = false) or at `para` non-zero coefficients
// (varnum = true). The ridge is applied through the implicit augmented design
// [x; sqrt(lambda) I] / sqrt(1 + lambda), never formed.
VectorXd solvebeta(const MapMat& x, const VectorXd& y, double lambda, double para,
                   bool varnum, int max_steps, const StepFn& stop_fn = nullptr) {
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
    if (varnum && stop_fn && stop_fn(beta, (int)active.size())) break;
    if (varnum && (double)active.size() >= para) break;
  }
  return beta;
}

// ---------------------------------------------------------------- tuning

// Zou's adjusted variance of the weights B (columns normalised): QR of XB
double pev_zou(const MapMat& X, const MatrixXd& B, double totvar) {
  MatrixXd U = X * B;
  Eigen::HouseholderQR<MatrixXd> qr(U);
  const MatrixXd& R = qr.matrixQR();
  double s = 0.0;
  for (int j = 0; j < B.cols(); ++j) s += R(j, j) * R(j, j);
  return s / totvar;
}

// PEV_pq of Camacho et al. (2025) with Q = polar(X'X B), as in spca_zou_cpp
double pev_camacho(const MapMat& X, const MatrixXd& B, double totvar) {
  MatrixXd Q = polar(X.transpose() * (X * B));
  double total;
  VectorXd comp;
  pev_pq(X, B, Q, totvar, total, comp);
  return total;
}

struct ZouFit {
  MatrixXd B, alpha;  // B with normalised columns
  double pev;
  int iterations;
};

// elasticnet::spca (varnum) for K = cards.size() components, started at alpha0
ZouFit zou_fit(const MapMat& X, const std::vector<int>& cards, double lambda,
               const MatrixXd& alpha0, bool camacho, double totvar,
               int max_iter, double eps_conv, int max_steps) {
  const int K = cards.size();
  ZouFit f;
  f.alpha = alpha0;
  f.B.resize(X.cols(), K);
  for (int i = 0; i < K; ++i)
    f.B.col(i) = solvebeta(X, X * f.alpha.col(i), lambda, cards[i], true,
                           max_steps);
  MatrixXd temp = f.B;
  normalize_cols(temp);
  int k = 0;
  double diff = 1.0;
  while (k < max_iter && diff > eps_conv) {
    ++k;
    MatrixXd XB = X * f.B;
    f.alpha = polar(X.transpose() * XB);
    for (int i = 0; i < K; ++i)
      f.B.col(i) = solvebeta(X, X * f.alpha.col(i), lambda, cards[i], true,
                             max_steps);
    MatrixXd B2 = f.B;
    normalize_cols(B2);
    diff = convcheck(B2, temp);
    temp.swap(B2);
  }
  normalize_cols(f.B);
  f.iterations = k;
  f.pev = camacho ? pev_camacho(X, f.B, totvar) : pev_zou(X, f.B, totvar);
  return f;
}

// ---------------------------------------------------------------- zSPCA_seq

// beta for the response X alpha: elastic net by LARS-EN with c non-zero
// weights, or, for lambda = Inf, soft thresholding of X'X alpha keeping its c
// largest entries (as SpaSM spca.m and MEDA spcaZou)
VectorXd seq_beta(const MapMat& X, const VectorXd& alpha, double lambda, int c,
                  int max_steps) {
  const VectorXd Xa = X * alpha;
  if (!std::isinf(lambda)) return solvebeta(X, Xa, lambda, c, true, max_steps);
  VectorXd g = X.transpose() * Xa;
  const int p = g.size();
  if (c >= p) return g;
  std::vector<double> a(p);
  for (int i = 0; i < p; ++i) a[i] = std::abs(g(i));
  std::nth_element(a.begin(), a.begin() + c, a.end(), std::greater<double>());
  const double thr = a[c];                         // (c + 1)-th largest |g|
  return g.array().sign() * (g.array().abs() - thr).max(0.0);
}

struct SeqFit {
  VectorXd beta, alpha;  // beta with unit length (or zero)
  int iterations;
};

// Sjostrand et al. (2018), Algorithm 4, one component: alternate
//   beta_k = EN(X alpha_k), normalised, and
//   alpha_k = (I - A A') X'X beta_k, normalised,
// with A = [alpha_1, ..., alpha_{k-1}] fixed, until beta_k converges
SeqFit seq_component(const MapMat& X, const MatrixXd& A,
                     const VectorXd& alpha0, double lambda, int c,
                     int max_iter, double eps_conv, int max_steps) {
  SeqFit f;
  f.alpha = alpha0;
  f.beta = VectorXd::Zero(X.cols());
  VectorXd beta_old;
  f.iterations = 0;
  while (f.iterations < max_iter) {
    ++f.iterations;
    beta_old = f.beta;
    f.beta = seq_beta(X, f.alpha, lambda, c, max_steps);
    const double nb = f.beta.norm();
    if (nb > 0) f.beta /= nb;
    VectorXd s = X.transpose() * (X * f.beta);
    if (A.cols()) s -= A * (A.transpose() * s);
    const double ns = s.norm();
    if (ns > 0) f.alpha = s / ns;
    // change of beta, up to its sign
    const double d = std::min((f.beta - beta_old).cwiseAbs().maxCoeff(),
                              (f.beta + beta_old).cwiseAbs().maxCoeff());
    if (d < eps_conv) break;
  }
  return f;
}

} // namespace

// [[Rcpp::export(.spca_zou_tune_cpp)]]
Rcpp::List spca_zou_tune_cpp(const Eigen::Map<Eigen::MatrixXd> X, int K,
                             double lambda, bool camacho, double alpha,
                             int max_iter, double eps_conv, int max_steps) {
  const int n = X.rows(), p = X.cols();
  // small-ridge trick: with lambda = 0 and p >= n, LARS-EN (a lasso) can
  // select at most n - 1 variables; a ridge of 1e-6 removes the limit
  const double lam = (lambda == 0.0 && p >= n) ? 1e-6 : lambda;
  const int cmax = (lam > 0) ? p : std::min(p, n - 1);
  const double totvar = X.squaredNorm();
  MatrixXd V = right_sv(X, K);                     // PCA start of elasticnet
  VectorXd pev_pca(K);
  for (int k = 0; k < K; ++k)
    pev_pca(k) = (X * V.leftCols(k + 1)).squaredNorm() / totvar;

  std::vector<int> cards;
  Rcpp::IntegerVector candidate(K), n_fits(K);
  ZouFit best;
  MatrixXd alpha_warm = V.leftCols(1);
  for (int k = 0; k < K; ++k) {
    const double target = alpha * pev_pca(k);
    // warm start: alpha of the last fit, plus the k-th PCA direction
    MatrixXd alpha0(p, k + 1);
    if (k) alpha0.leftCols(k) = best.alpha;
    alpha0.col(k) = V.col(k);

    // (a) candidate from one LARS-EN path with alpha_k fixed
    MatrixXd Bk(p, k + 1);
    if (k) Bk.leftCols(k) = best.B;
    int c0 = cmax;
    StepFn fn = [&](const VectorXd& beta, int na) {
      const double nb = beta.norm();
      if (nb == 0.0) return false;
      Bk.col(k) = beta / nb;
      const double pv = camacho ? pev_camacho(X, Bk, totvar)
                                : pev_zou(X, Bk, totvar);
      if (pv >= target) { c0 = na; return true; }
      return false;
    };
    solvebeta(X, X * alpha0.col(k), lam, cmax, true, max_steps, fn);
    candidate[k] = c0;

    // (b) confirm with full fits: smallest c with PEV >= target
    int fits = 0;
    auto fit_c = [&](int c, ZouFit& out) {
      std::vector<int> cc = cards;
      cc.push_back(c);
      out = zou_fit(X, cc, lam, alpha0, camacho, totvar, max_iter, eps_conv,
                    max_steps);
      alpha0 = out.alpha;                          // (c) warm start
      ++fits;
      return out.pev >= target;
    };
    ZouFit f_hi, f;
    int lo, hi;                                    // lo fails, hi passes
    if (fit_c(c0, f_hi)) {
      hi = c0; lo = 0;                             // c = 0 counts as failing
      for (int step = 1; hi - step > 0; step *= 2) {
        if (fit_c(hi - step, f)) { hi -= step; f_hi = f; }
        else { lo = hi - step; break; }
      }
    } else {
      lo = c0; hi = -1;
      for (int step = 1; hi < 0; step *= 2) {
        const int c = std::min(lo + step, cmax);
        if (fit_c(c, f)) { hi = c; f_hi = f; }
        else if (c == cmax) { hi = cmax; f_hi = f; }  // never met: densest
        else lo = c;
      }
    }
    while (hi - lo > 1) {
      const int mid = lo + (hi - lo) / 2;
      if (fit_c(mid, f)) { hi = mid; f_hi = f; } else lo = mid;
    }
    cards.push_back(hi);
    best = f_hi;
    n_fits[k] = fits;
  }

  // loadings Q and PEV_pq of the final model, as in spca_zou_cpp
  MatrixXd U = X * best.B;
  MatrixXd Q = polar(X.transpose() * U);
  double pq;
  VectorXd pq_comp;
  pev_pq(X, best.B, Q, totvar, pq, pq_comp);
  return Rcpp::List::create(Rcpp::Named("P") = best.B,
                            Rcpp::Named("Q") = Q,
                            Rcpp::Named("cardinality") = cards,
                            Rcpp::Named("pev") = pev_zou(X, best.B, totvar),
                            Rcpp::Named("pev_pq") = pq,
                            Rcpp::Named("pev_pca") = pev_pca,
                            Rcpp::Named("lambda") = lam,
                            Rcpp::Named("candidate") = candidate,
                            Rcpp::Named("n_fits") = n_fits,
                            Rcpp::Named("iterations") = best.iterations);
}

// zSPCA_seq: sequential sparse PCA of Sjostrand et al. (2018, Algorithm 4).
// card: number of non-zero weights per component (SpaSM's -stop); if empty,
// each cardinality is chosen by the sPCA_M rule: the smallest c_k such that
// Zou's adjusted PEV of the first k components >= alpha * PEV of k PCs.
// [[Rcpp::export(.spca_zou_seq_cpp)]]
Rcpp::List spca_zou_seq_cpp(const Eigen::Map<Eigen::MatrixXd> X, int K,
                            const Eigen::Map<Eigen::VectorXi> card,
                            double lambda, double alpha, int max_iter,
                            double eps_conv, int max_steps) {
  const int n = X.rows(), p = X.cols();
  const bool rule = card.size() == 0;
  // small-ridge trick, as in spca_zou_tune_cpp
  const double lam = (lambda == 0.0 && p >= n) ? 1e-6 : lambda;
  const int cmax = (lam > 0) ? p : std::min(p, n - 1);
  const double totvar = X.squaredNorm();
  MatrixXd V = right_sv(X, K);                     // A starts at PCA loadings
  VectorXd pev_pca(K);
  for (int k = 0; k < K; ++k)
    pev_pca(k) = (X * V.leftCols(k + 1)).squaredNorm() / totvar;

  MatrixXd B(p, K), A(p, K);
  Rcpp::IntegerVector cards(K), candidate(K), n_fits(K), iterations(K);
  VectorXd pev(K);
  for (int k = 0; k < K; ++k) {
    const MatrixXd Ak = A.leftCols(k);
    MatrixXd Bk(p, k + 1);
    Bk.leftCols(k) = B.leftCols(k);
    VectorXd alpha0 = V.col(k);
    int fits = 0;
    auto fit_c = [&](int c, SeqFit& out) {
      out = seq_component(X, Ak, alpha0, lam, c, max_iter, eps_conv,
                          max_steps);
      alpha0 = out.alpha;                          // warm start
      ++fits;
      Bk.col(k) = out.beta;
      return pev_zou(X, Bk, totvar);
    };
    SeqFit best;
    if (!rule) {
      cards[k] = std::min(card(k), cmax);
      fit_c(cards[k], best);
    } else {
      const double target = alpha * pev_pca(k);
      // (a) candidate from one LARS-EN path with alpha_k fixed
      int c0 = cmax / 2;
      if (!std::isinf(lam)) {
        c0 = cmax;
        StepFn fn = [&](const VectorXd& beta, int na) {
          const double nb = beta.norm();
          if (nb == 0.0) return false;
          Bk.col(k) = beta / nb;
          if (pev_zou(X, Bk, totvar) >= target) { c0 = na; return true; }
          return false;
        };
        solvebeta(X, X * alpha0, lam, cmax, true, max_steps, fn);
      }
      c0 = std::max(c0, 1);
      candidate[k] = c0;
      // (b) smallest c with PEV >= target: step down or up from the
      // candidate by 1, 2, 4, ..., then bisect (lo fails, hi passes)
      SeqFit f;
      int lo, hi;
      if (fit_c(c0, f) >= target) {
        hi = c0; lo = 0; best = f;
        for (int step = 1; hi - step > 0; step *= 2) {
          if (fit_c(hi - step, f) >= target) { hi -= step; best = f; }
          else { lo = hi - step; break; }
        }
      } else {
        lo = c0; hi = -1;
        for (int step = 1; hi < 0; step *= 2) {
          const int c = std::min(lo + step, cmax);
          if (fit_c(c, f) >= target || c == cmax) { hi = c; best = f; }
          else lo = c;
        }
      }
      while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (fit_c(mid, f) >= target) { hi = mid; best = f; } else lo = mid;
      }
      cards[k] = hi;
    }
    B.col(k) = best.beta;
    A.col(k) = best.alpha;
    iterations[k] = best.iterations;
    n_fits[k] = fits;
    pev(k) = pev_zou(X, B.leftCols(k + 1), totvar);
  }

  // loadings Q and PEV_pq of the final model, as in spca_zou_tune_cpp
  MatrixXd Q = polar(X.transpose() * (X * B));
  double pq;
  VectorXd pq_comp;
  pev_pq(X, B, Q, totvar, pq, pq_comp);
  return Rcpp::List::create(Rcpp::Named("P") = B,
                            Rcpp::Named("A") = A,
                            Rcpp::Named("Q") = Q,
                            Rcpp::Named("cardinality") = cards,
                            Rcpp::Named("pev") = pev,
                            Rcpp::Named("pev_pq") = pq,
                            Rcpp::Named("pev_pca") = pev_pca,
                            Rcpp::Named("lambda") = lam,
                            Rcpp::Named("candidate") = candidate,
                            Rcpp::Named("n_fits") = n_fits,
                            Rcpp::Named("iterations") = iterations);
}
