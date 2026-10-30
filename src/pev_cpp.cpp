// [[Rcpp::depends(RcppEigen)]]

#include "validators.h"  // get validators

// Percentage explained variance (pev) of the components t_j = X w_j,
// as in Zou, Hastie and Tibshirani (2006). The adjusted variance of t_j is
// R_jj^2, where T = QR is the unpivoted QR decomposition of T = X W. The
// pivoting order is the natural column order, so component j is adjusted
// only for components 1, ..., j - 1. pev_j = R_jj^2 / total variance.
//
// The same R_jj^2 come from the covariance matrix alone: T'T = W' X'X W,
// so with S = X'X (up to a scale factor n - 1 that cancels in the ratio),
// the Cholesky factor L of W' S W satisfies L L' = W' S W, with
// L_jj = R_jj.

using namespace Rcpp;
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::HouseholderQR;
using Eigen::LLT;

// [[Rcpp::export]]
Eigen::VectorXd pevDataC(const Eigen::Map<Eigen::MatrixXd>& W,
                         const Eigen::Map<Eigen::MatrixXd>& X) {
  try {
    validate_data_mat(X, "X");
    validate_mult_dims(X, W, "X", "W");
    const int r = W.cols();
    if (X.rows() < r)
      stop("pevDataC: X must have at least as many rows as W has columns");

    // T = X W, the n x r matrix of component scores
    MatrixXd T(X.rows(), r);
    T.noalias() = X * W;

    // unpivoted Householder QR (no column reordering)
    HouseholderQR<MatrixXd> qr(T);

    // the diagonal of the compact QR storage holds R_jj (may be negative)
    VectorXd rjj = qr.matrixQR().diagonal();

    // total variance of X, the same scale as the squared R_jj
    const double total = X.squaredNorm();
    if (total <= 0.0)
      stop("pevDataC: X has zero total variance");

    return rjj.array().square() / total;
  }
  catch (std::exception& ex) {
    Rcpp::stop("pevDataC failed: " + std::string(ex.what()));
  }
  catch (...) {
    Rcpp::stop("pevDataC failed: unknown C++ error");
  }
}

// [[Rcpp::export]]
Eigen::VectorXd pevCovC(const Eigen::Map<Eigen::MatrixXd>& W,
                        const Eigen::Map<Eigen::MatrixXd>& S) {
  try {
    validate_cov_mat(S, "S");
    validate_mult_dims(S, W, "S", "W");
    const int r = W.cols();

    // M = W' S W, the r x r Gram matrix of the component scores
    MatrixXd M(r, r);
    M.noalias() = W.transpose() * S * W;

    // unpivoted Cholesky factorisation M = L L'; L_jj = R_jj
    LLT<MatrixXd> llt(M);
    if (llt.info() != Eigen::Success)
      stop("pevCovC: W' S W is not positive definite");

    VectorXd rjj = llt.matrixLLT().diagonal();

    // total variance of X, the same scale as the squared L_jj
    const double total = S.trace();

    return rjj.array().square() / total;
  }
  catch (std::exception& ex) {
    Rcpp::stop("pevCovC failed: " + std::string(ex.what()));
  }
  catch (...) {
    Rcpp::stop("pevCovC failed: unknown C++ error");
  }
}
