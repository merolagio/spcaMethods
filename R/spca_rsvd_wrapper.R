#' Sparse PCA via regularized SVD (sPCA-rSVD)
#'
#' Computes \code{n_comps} sparse weight vectors with Algorithm 1 of Shen and
#' Huang (2008), soft-thresholding penalty, ported to C++ (RcppEigen). As in
#' Guerra-Urzola's \code{sPCArSVD.m}, the tuning parameter is the degree of
#' sparsity (number of zero weights), and the threshold \eqn{\lambda} is reset
#' at each iteration to the corresponding order statistic of \eqn{|X'u|}.
#'
#' The degree of sparsity is either
#' \itemize{
#'   \item fixed: component j has \code{floor(ncol(X) * ps[j])} zero weights;
#'   \item or, when \code{ps = NULL}, chosen by the ad hoc rule of Shen and
#'     Huang (2008, Algorithm 3): for component k, the CPEV of the first k
#'     components is computed for each number of zeros j in
#'     \code{n_zeros_grid}, and the largest j is kept whose CPEV is at least
#'     \code{(1 - drop)} times the CPEV at j = 0 (the paper suggests a drop of
#'     5\% or 10\%). Earlier components keep the sparsity already chosen.
#' }
#' CPEV is \eqn{tr(X_k'X_k) / tr(X'X)}, with
#' \eqn{X_k = X V_k (V_k'V_k)^{-1} V_k'} (Shen and Huang 2008, Section 2.3).
#'
#' Components are computed one at a time. Each starts from the best rank-one
#' approximation of the current residual matrix, and the next residual is
#' \eqn{X - u \tilde v'} (Shen and Huang 2008, Section 2.2).
#'
#' The data matrix is passed to C++ by reference (an \code{Eigen::Map} onto R's
#' memory), so it is not copied as long as it is already a double-precision
#' matrix. The residual matrices are never formed.
#'
#' @section Differences from sPCArSVD.m:
#' \itemize{
#'   \item The residual is \eqn{X - u \tilde v'}, as in the paper. MATLAB uses
#'     \eqn{X - u (d_1 \tilde v / \|\tilde v\|)'}, with \eqn{d_1} the largest
#'     singular value of the current residual, so from component 2 on the
#'     results differ.
#'   \item Iterations stop when \eqn{\|u_{new} - u_{old}\|} < \code{eps}.
#'     MATLAB stops on the relative change of \eqn{\|X - u v'\|_2 + 2
#'     \lambda \|v\|_1}, which needs the spectral norm of an
#'     \eqn{n \times p} matrix at every iteration.
#'   \item Weights are returned with unit length; MATLAB scales them by
#'     \eqn{d_1}.
#'   \item \code{ps} can differ between components.
#' }
#'
#' @param X numeric matrix, \eqn{n \times p}: observations in rows, variables
#'   in columns. Must be column-centred.
#' @param n_comps number of components.
#' @param ps proportion of zero weights, in \eqn{[0, 1)}; recycled to length
#'   \code{n_comps}. \code{NULL} chooses the number of zeros by the ad hoc
#'   CPEV rule.
#' @param drop largest relative drop of CPEV from its value at j = 0 allowed
#'   by the ad hoc rule. Ignored when \code{ps} is given.
#' @param n_zeros_grid numbers of zeros tried by the ad hoc rule; 0 is always
#'   added. The default \code{0:(ncol(X) - 1)} costs one fit of Algorithm 1
#'   per value and component, so a coarser grid is advisable for large p.
#'   Ignored when \code{ps} is given.
#' @param iter_max maximum number of iterations per component.
#' @param eps tolerance on the change of the unit vector u between iterations.
#'
#' @return an object of class \code{c("cspca", "spca", "list")} with
#'   \item{weights}{\eqn{p \times n_comps} matrix of sparse weights, unit
#'     length (\eqn{v} in Shen and Huang 2008).}
#'   \item{n_comps}{number of components.}
#'   \item{cardinality}{number of non-zero weights of each component.}
#'   \item{pev}{Zou's adjusted proportion of explained variance of each
#'     component, from the QR decomposition of \code{X \%*\% weights}
#'     (\code{\link{compute_pev}}); the total is \code{sum(X^2)}.}
#'   \item{own_pev}{the method's own measure: CPEV of the first k
#'     components, k = 1, ..., n_comps (cumulative).}
#'   \item{U}{\eqn{n \times n_comps} matrix of unit-length component vectors
#'     (\eqn{u = X_k \tilde v / \|X_k \tilde v\|}, \eqn{X_k} the residual).}
#'   \item{d}{\eqn{\|\tilde v\|} for each component.}
#'   \item{lambda}{the thresholds at convergence.}
#'   \item{n_zeros}{the number of zero weights of each component.}
#'   \item{cpev_grid}{ad hoc rule only: CPEV for each value of
#'     \code{n_zeros_grid} (rows) and component (columns).}
#'   \item{iterations}{iterations used for each component.}
#'   \item{method_name}{short label of the method, \code{"rSVD"}.}
#'   \item{method_descrip}{one-line description of the method.}
#'
#' @references H. Shen, J. Z. Huang (2008). Sparse principal component
#'   analysis via regularized low rank matrix approximation. \emph{Journal of
#'   Multivariate Analysis}, 99, 1015--1034.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("spca_rsvd.cpp")   # not needed when built as a package
#' set.seed(1)
#' X = scale(matrix(rnorm(100 * 40), 100), scale = FALSE)
#' fit = spca_rsvd(X, n_comps = 3, ps = 0.8)
#' fit$cardinality
#' fit2 = spca_rsvd(X, n_comps = 3, drop = 0.05)   # ad hoc CPEV rule
#' fit2$n_zeros
#' }
#'
#' @export
spca_rsvd = function(X, n_comps = 1L, ps = NULL, drop = 0.05,
                     n_zeros_grid = NULL, iter_max = 1000L, eps = 1e-7) {
  tryCatch({
  if (is.data.frame(X)) {
    X = as.matrix(X)
    # only copies when the input is not already double
    storage.mode(X) = "double"
  }
  if (anyNA(X)) stop("X must not contain missing values")
  p = ncol(X)
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(nrow(X), p))
    stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  if (is.null(ps)) {
    if (is.null(n_zeros_grid)) n_zeros_grid = 0:(p - 1)
    n_zeros_grid = sort(unique(as.integer(c(0, n_zeros_grid))))
    if (any(n_zeros_grid < 0 | n_zeros_grid >= p))
      stop("n_zeros_grid must be in 0, ..., ncol(X) - 1")
    if (drop < 0 || drop >= 1) stop("drop must be in [0, 1)")
    n_zeros = integer(n_comps)
  } else {
    ps = rep_len(ps, n_comps)
    if (any(ps < 0 | ps >= 1)) stop("ps must be in [0, 1)")
    n_zeros = as.integer(floor(p * ps))
    n_zeros_grid = integer(0)
  }
  fit = .spca_rsvd_cpp(X, n_zeros, n_zeros_grid, as.double(drop),
                       as.integer(iter_max), as.double(eps))
  W = fit$V
  dimnames(W) = list(colnames(X), paste0("sPC", seq_len(n_comps)))
  structure(list(
    weights = W, n_comps = n_comps, cardinality = colSums(W != 0),
    pev = compute_pev(W, X), own_pev = fit$cpev, U = fit$U, d = fit$d,
    lambda = fit$lambda, n_zeros = fit$n_zeros, cpev_grid = fit$cpev_grid,
    iterations = fit$iterations, method_name = "rSVD",
    method_descrip = paste0("sPCA-rSVD (Shen and Huang 2008), soft ",
                            "thresholding, ",
                            if (is.null(ps)) "ad hoc CPEV rule"
                            else "fixed proportion of zeros")),
    class = c("cspca", "spca", "list"))
  }, error = function(e) {
    stop("spca_rsvd(): ", conditionMessage(e), call. = FALSE)
  })
}
