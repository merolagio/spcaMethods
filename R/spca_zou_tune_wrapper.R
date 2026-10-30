#' Zou's sparse PCA with the cardinality chosen by a PEV stopping rule
#'
#' Fits the sparse PCA of Zou, Hastie and Tibshirani (2006) (elastic-net
#' weights computed by LARS-EN, as in \code{spca_zou()}), choosing the number of
#' non-zero weights of each component by a stopping rule. Component by
#' component, with the cardinalities of the earlier components kept, the
#' cardinality \eqn{c_k} is the smallest one such that
#' \deqn{PEV(k) \ge \alpha PEV_{PCA}(k),}
#' where \eqn{PEV_{PCA}(k)} is the variance explained by the first k PCs and
#' PEV is
#' \itemize{
#'   \item \code{method = "Z"} (sPCA_Z): Zou's adjusted variance, from the QR
#'     decomposition of the scores \eqn{XP};
#'   \item \code{method = "C"} (sPCA_C): \eqn{PEV_{pq}} of Camacho et al.
#'     (2025), the variance of the model \eqn{X P (Q'P)^{-1} Q'}.
#' }
#' The search takes a candidate from one LARS-EN path (PEV evaluated after
#' every step, with column k of Zou's matrix \eqn{A} fixed), then confirms it
#' with full fits, stepping down or up by 1, 2, 4, ... and bisecting. Each
#' full fit starts from the matrix \eqn{A} of the previous one.
#'
#' With \code{lambda = 0} and \code{ncol(X) >= nrow(X)}, LARS-EN is a lasso and
#' selects at most \code{nrow(X) - 1} variables, so a ridge of 1e-6 is used
#' instead (returned as \code{lambda}).
#'
#' X is passed to C++ by reference and is not copied; \eqn{X'X} is never
#' formed.
#'
#' @param X numeric matrix, observations in rows, already preprocessed
#'   (column-centred, and scaled if wanted); it is used as it is.
#' @param n_comps number of components.
#' @param method \code{"Z"} (Zou's adjusted variance) or \code{"C"}
#'   (Camacho's \eqn{PEV_{pq}}).
#' @param lambda ridge parameter of the elastic net.
#' @param alpha smallest fraction of the PEV of PCA that the first k
#'   components must explain.
#' @param max_iter maximum number of iterations of the alternating algorithm.
#' @param eps_conv convergence tolerance (as \code{elasticnet::spca}).
#' @param max_steps maximum number of LARS-EN steps; 0 means 50 times the
#'   largest number of variables.
#'
#' @return a list with
#'   \item{P}{weights, columns of unit length.}
#'   \item{Q}{orthonormal loadings, \eqn{Q = polar(X'XP)}.}
#'   \item{cardinality}{chosen number of non-zero weights per component.}
#'   \item{pev}{Zou's adjusted PEV of the final model.}
#'   \item{pev_pq}{Camacho's \eqn{PEV_{pq}} of the final model.}
#'   \item{pev_pca}{PEV of the first k PCs, k = 1, ..., n_comps.}
#'   \item{lambda}{ridge used.}
#'   \item{candidate}{cardinalities proposed by the LARS-EN paths.}
#'   \item{n_fits}{full fits used per component.}
#'   \item{iterations}{iterations of the final fit.}
#'
#' @references H. Zou, T. Hastie, R. Tibshirani (2006). Sparse principal
#'   component analysis. \emph{Journal of Computational and Graphical
#'   Statistics}, 15, 265--286.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("spca_zou_tune.cpp")   # not needed when built as a package
#' set.seed(1)
#' X = scale(matrix(rnorm(100 * 40), 100))
#' fz = spca_zou_tune(X, n_comps = 2, method = "Z")
#' fc = spca_zou_tune(X, n_comps = 2, method = "C")
#' fz$cardinality
#' fc$cardinality
#' }
#'
#' @export
spca_zou_tune = function(X, n_comps = 1L, method = c("Z", "C"), lambda = 0,
                         alpha = 0.95, max_iter = 200L, eps_conv = 1e-3,
                         max_steps = 0L) {
  tryCatch({
  method = match.arg(method)
  if (is.data.frame(X)) {
    X = as.matrix(X)
    # only copies when the input is not already double
    storage.mode(X) = "double"
  }
  if (anyNA(X)) stop("X must not contain missing values")
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(dim(X)))
    stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  if (lambda < 0 || is.infinite(lambda)) stop("lambda must be finite, >= 0")
  if (alpha <= 0 || alpha > 1) stop("alpha must be in (0, 1]")
  .spca_zou_tune_cpp(X, n_comps, as.double(lambda), method == "C",
                     as.double(alpha), as.integer(max_iter),
                     as.double(eps_conv), as.integer(max_steps))
  }, error = function(e) {
    stop("spca_zou_tune(): ", conditionMessage(e), call. = FALSE)
  })
}

#' Sequential sparse PCA (zSPCA_seq) with sPCA_M cardinality selection
#'
#' Fits the sequential sparse PCA of Sjostrand et al. (2018, Algorithm 4),
#' the \code{spca} function of the SpaSM toolbox, ported to C++ (RcppEigen).
#' Components are computed one at a time, the earlier ones kept fixed: for
#' component k, the algorithm alternates
#' \itemize{
#'   \item \eqn{\beta_k}: elastic net (LARS-EN) of \eqn{X\alpha_k} on X with
#'     ridge \code{lambda} and \eqn{c_k} non-zero weights, normalised; for
#'     \code{lambda = Inf}, soft thresholding of \eqn{X'X\alpha_k} keeping its
#'     \eqn{c_k} largest entries;
#'   \item \eqn{\alpha_k = (I - A_{k-1}A_{k-1}')X'X\beta_k}, normalised, where
#'     \eqn{A_{k-1}} holds the earlier \eqn{\alpha}'s,
#' }
#' until \eqn{\beta_k} converges. A starts at the PCA loadings.
#'
#' The cardinalities are either given (\code{card}, as SpaSM's negative
#' \code{stop}), or, by default, chosen by the sPCA_M rule: \eqn{c_k} is the
#' smallest cardinality such that Zou's adjusted PEV of the first k
#' components is at least \code{alpha} times the PEV of the first k principal
#' components. The search takes a candidate from one LARS-EN path (PEV
#' evaluated after every step, with \eqn{\alpha_k} fixed; for
#' \code{lambda = Inf} the candidate is half the number of variables), then
#' confirms it with fits of component k, stepping down or up by 1, 2, 4, ...
#' and bisecting; each fit starts from the \eqn{\alpha_k} of the previous one.
#'
#' With \code{lambda = 0} and \code{ncol(X) >= nrow(X)}, LARS-EN is a lasso and
#' selects at most \code{nrow(X) - 1} variables, so a ridge of 1e-6 is used
#' instead (returned as \code{lambda}).
#'
#' The convergence criterion of SpaSM is not documented; here iterations stop
#' when the largest change of \eqn{\beta_k} (up to its sign) is below
#' \code{eps_conv}. X is passed to C++ by reference and is not copied;
#' \eqn{X'X} is never formed.
#'
#' @param X numeric matrix, observations in rows, already preprocessed
#'   (column-centred, and scaled if wanted); it is used as it is.
#' @param n_comps number of components.
#' @param card number of non-zero weights of each component (recycled to
#'   length \code{n_comps}); \code{NULL} (default) chooses them by the sPCA_M
#'   rule.
#' @param alpha smallest fraction of the PEV of PCA that the first k
#'   components must explain (sPCA_M rule). Ignored when \code{card} is
#'   given.
#' @param lambda ridge parameter of the elastic net (\eqn{\delta} in SpaSM);
#'   \code{Inf} gives soft thresholding.
#' @param max_iter maximum number of iterations per component.
#' @param eps_conv convergence tolerance on \eqn{\beta_k}.
#' @param max_steps maximum number of LARS-EN steps; 0 means 50 times the
#'   largest number of variables.
#'
#' @return a list with
#'   \item{P}{weights \eqn{\beta}, columns of unit length.}
#'   \item{A}{the matrix A of the algorithm (orthonormal columns).}
#'   \item{Q}{orthonormal loadings, \eqn{Q = polar(X'XP)}.}
#'   \item{cardinality}{number of non-zero weights per component.}
#'   \item{pev}{Zou's adjusted PEV of the first k components,
#'     k = 1, ..., n_comps.}
#'   \item{pev_pq}{Camacho's \eqn{PEV_{pq}} of the final model.}
#'   \item{pev_pca}{PEV of the first k PCs, k = 1, ..., n_comps.}
#'   \item{lambda}{ridge used.}
#'   \item{candidate}{sPCA_M rule only: cardinalities proposed by the
#'     LARS-EN paths.}
#'   \item{n_fits}{fits of each component.}
#'   \item{iterations}{iterations of the final fit of each component.}
#'
#' @references K. Sjostrand, L. H. Clemmensen, R. Larsen, G. Einarsson,
#'   B. Ersboll (2018). SpaSM: A MATLAB toolbox for sparse statistical
#'   modeling. \emph{Journal of Statistical Software}, 84(10), 1--37.
#'
#'   H. Zou, T. Hastie, R. Tibshirani (2006). Sparse principal component
#'   analysis. \emph{Journal of Computational and Graphical Statistics}, 15,
#'   265--286.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("spca_zou_tune.cpp")   # not needed when built as a package
#' set.seed(1)
#' X = scale(matrix(rnorm(100 * 40), 100))
#' fm = spca_zou_seq(X, n_comps = 2)                   # sPCA_M rule
#' fs = spca_zou_seq(X, n_comps = 2, card = c(10, 5))  # as SpaSM
#' fm$cardinality
#' }
#'
#' @export
spca_zou_seq = function(X, n_comps = 1L, card = NULL, alpha = 0.95,
                        lambda = 0, max_iter = 3000L, eps_conv = 1e-9,
                        max_steps = 0L) {
  tryCatch({
  if (is.data.frame(X)) {
    X = as.matrix(X)
    # only copies when the input is not already double
    storage.mode(X) = "double"
  }
  if (anyNA(X)) stop("X must not contain missing values")
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(dim(X)))
    stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  if (is.null(card)) {
    card = integer(0)
    if (alpha <= 0 || alpha > 1) stop("alpha must be in (0, 1]")
  } else {
    card = as.integer(rep_len(card, n_comps))
    if (any(card < 1L)) stop("card must be positive")
  }
  if (lambda < 0) stop("lambda must be >= 0")
  .spca_zou_seq_cpp(X, n_comps, card, as.double(lambda), as.double(alpha),
                    as.integer(max_iter), as.double(eps_conv),
                    as.integer(max_steps))
  }, error = function(e) {
    stop("spca_zou_seq(): ", conditionMessage(e), call. = FALSE)
  })
}
