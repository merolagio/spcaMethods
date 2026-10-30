#' Sparse PCA of Zou, Hastie and Tibshirani (elastic-net / LARS-EN version)
#'
#' Port to C++ (RcppEigen) of \code{spca()} from the R package \pkg{elasticnet}
#' (Zou and Hastie), the reference implementation of Zou, Hastie and Tibshirani (2006).
#' SpaSM's \code{spca.m}, which Camacho et al. call \code{spca_zouhastie}, implements the
#' same alternating algorithm: an elastic-net fit (ridge \code{lambda}) of each score
#' \eqn{X a_k} on \eqn{X} gives the sparse weights, and a Procrustes step gives the
#' orthonormal coefficients.
#'
#' The data matrix is passed to C++ by reference (\code{Eigen::Map}), and \eqn{X'X} is
#' never formed, so the function works with many more variables than observations.
#' \code{X} is copied only when \code{center} or \code{scale} is \code{TRUE}; pass
#' preprocessed data with \code{center = FALSE} to avoid the copy.
#'
#' @param X numeric matrix, observations in rows. Ignored when \code{gram} is given.
#' @param n_comps number of components.
#' @param para for \code{sparse = "penalty"}, the l1 penalties; for
#'   \code{sparse = "varnum"}, the numbers of non-zero weights. Recycled to length
#'   \code{n_comps}.
#' @param gram optional covariance or correlation matrix, used instead of \code{X}
#'   (\code{type = "Gram"} in \pkg{elasticnet}): \code{X} is then its square root.
#' @param lambda ridge penalty (\eqn{\lambda_2}, SpaSM's \code{delta}). 0 gives the
#'   lasso. Use \code{\link{spca_soft}} for \code{lambda = Inf}.
#' @param sparse \code{"penalty"} or \code{"varnum"}, as in \pkg{elasticnet}.
#' @param center,scale centre and scale the columns of \code{X} first
#'   (\pkg{elasticnet} always centres, and scales when \code{use.corr = TRUE}).
#' @param max_iter maximum number of alternating iterations.
#' @param eps_conv convergence tolerance (\pkg{elasticnet}'s \code{eps.conv}).
#' @param max_steps maximum number of LARS-EN steps per fit; 0 means
#'   \pkg{elasticnet}'s default, 50 times the maximal number of variables.
#'
#' @return an object of class \code{c("cspca", "spca", "list")} with
#'   \item{weights}{\eqn{p \times n_comps} matrix of sparse weights,
#'     unit-norm columns (signs are arbitrary). This is \eqn{B} (\code{P} in
#'     Camacho et al.) of Zou et al. (2006).}
#'   \item{n_comps}{number of components.}
#'   \item{cardinality}{number of non-zero weights of each component.}
#'   \item{pev}{Zou's adjusted proportion of explained variance of each
#'     component, from the QR decomposition of \code{X \%*\% weights}
#'     (Zou et al. 2006; \code{$pev} in \pkg{elasticnet});
#'     the total is \code{sum(X^2)}.}
#'   \item{coefficients}{\eqn{p \times n_comps} orthonormal matrix
#'     \eqn{C = A} of the regression model \eqn{\hat X = X W C'} of Zou et
#'     al., the polar factor of \eqn{X'XW} (\code{Q} in Camacho et al.).}
#'   \item{pev_pq}{proportion of explained variance of the model
#'     \eqn{X P (Q'P)^{-1} Q'} (Camacho et al. 2025, eq. 21).}
#'   \item{pev_pq_comp}{per-component variances \eqn{\|X w_k\|^2 / \|X\|^2}
#'     with \eqn{W = P (Q'P)^{-1}}, as \code{expvar} in MEDA.}
#'   \item{iterations}{alternating iterations used.}
#'   \item{totvar}{total sum of squares of \code{X}.}
#'   \item{method_name}{short label of the method, \code{"sPCA_Z"}.}
#'   \item{method_descrip}{one-line description of the method.}
#'
#' @references H. Zou, T. Hastie, R. Tibshirani (2006). Sparse principal component
#'   analysis. \emph{Journal of Computational and Graphical Statistics}, 15, 265--286.
#'
#'   J. Camacho et al. (2025). All sparse PCA models are wrong, but some are useful.
#'   Part III: model interpretation. \emph{Chemometrics and Intelligent Laboratory
#'   Systems}, 266, 105498.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("spca_zou.cpp")   # not needed when built as a package
#' data(pitprops, package = "elasticnet")
#' fit = spca_zou(gram = pitprops, n_comps = 6, para = c(0.06, 0.16, 0.1, 0.5, 0.5, 0.5))
#' round(100 * fit$pev, 1)           # 28.0 14.0 13.3 7.4 6.8 6.2
#' }
#'
#' @export
spca_zou = function(X = NULL, n_comps = 1L, para, gram = NULL, lambda = 1e-6,
                    sparse = c("penalty", "varnum"), 
                    center = TRUE, scale = FALSE,
                    max_iter = 200L, eps_conv = 1e-3, max_steps = 0L) {
  tryCatch({
  sparse = match.arg(sparse)
  if (!is.null(gram)) {
    e = spca:::eigen_sym(gram)
    d = (e$values + abs(e$values)) / 2
    X = e$vectors %*% (sqrt(d) * t(e$vectors))   # elasticnet rootmatrix()
  } else {
    if (is.null(X)) stop("give X or gram")
    if (is.data.frame(X)) X = as.matrix(X)
    if (center || scale) X = base::scale(X, center = center, scale = scale)
    if (!is.double(X)) storage.mode(X) = "double"
  }
  if (anyNA(X)) stop("X must not contain missing values")
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(dim(X)))
    stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  if (lambda < 0) stop("lambda must be non-negative")
  if (is.infinite(lambda)) stop("for lambda = Inf use spca_soft()")
  para = as.double(rep_len(para, n_comps))
  fit = .spca_zou_cpp(X, n_comps, para, as.double(lambda),
                      sparse == "varnum", as.integer(max_iter),
                      as.double(eps_conv), as.integer(max_steps))
  W = fit$P
  dimnames(W) = list(colnames(X), paste0("sPC", seq_len(n_comps)))
  structure(list(
    weights = W, n_comps = n_comps, cardinality = colSums(W != 0),
    pev = fit$pev, coefficients = fit$Q, pev_pq = fit$pev_pq,
    pev_pq_comp = fit$pev_pq_comp, iterations = fit$iterations,
    totvar = fit$totvar, method_name = "sPCA_Z",
    method_descrip = paste0("Sparse PCA of Zou et al. (2006), elastic net ",
                            "(LARS-EN), ridge ", lambda)),
    class = c("cspca", "spca", "list"))
  }, error = function(e) {
    stop("spca_zou(): ", conditionMessage(e), call. = FALSE)
  })
}

#' Sparse PCA by soft thresholding (Zou et al. with ridge = Inf, MEDA spcaZou)
#'
#' Port to C++ (RcppEigen) of \code{spcaZou.m} from the MEDA Toolbox v1.8 (Camacho et
#' al.), which is the \eqn{\lambda_2 = \infty} case of Zou et al. (2006): the sparse
#' weights are soft-thresholded \eqn{X'X a_k}, followed by a Procrustes step for the
#' coefficients. Iterations, stopping rule (spectral norm of the change in the weights)
#' and outputs follow the MATLAB. \code{X} is passed by reference and is not
#' preprocessed, as in MEDA; autoscale it first if needed.
#'
#' @param X numeric matrix, observations in rows. Ignored when \code{gram} is given.
#' @param n_comps number of components.
#' @param stop per component: a negative value \code{-k} keeps the \code{k} largest
#'   weights; a non-negative value is the soft threshold. Recycled to length
#'   \code{n_comps}.
#' @param gram optional cross-product matrix \eqn{X'X}, used instead of \code{X}
#'   (MEDA takes \eqn{X = V |D|^{1/2} V'} from its eigendecomposition).
#' @param tol stopping tolerance (MEDA default 1e-15).
#' @param max_iter maximum number of iterations (MEDA default 1000).
#'
#' @return an object of class \code{c("cspca", "spca", "list")} with the
#'   elements of \code{\link{spca_zou}}: \code{weights}, \code{n_comps},
#'   \code{cardinality}, \code{pev} (Zou's adjusted PEV, computed by
#'   \code{\link{compute_pev}}), \code{coefficients} (MEDA \code{altloads}),
#'   \code{pev_pq}, \code{pev_pq_comp} (MEDA \code{expvar} divided by
#'   \code{var}), \code{iterations}, \code{totvar}, \code{method_name}
#'   (\code{"sPCA_soft"}) and \code{method_descrip}.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("spca_zou.cpp")
#' X = scale(matrix(rnorm(30 * 200), 30))
#' fit = spca_soft(X, n_comps = 3, stop = -4)
#' fit$cardinality                   # 4 4 4
#' }
#'
#' @export
spca_soft = function(X = NULL, n_comps = 1L, stop, gram = NULL, tol = 1e-15,
                     max_iter = 1000L) {
  tryCatch({
  if (!is.null(gram)) {
    e = eigen(gram, symmetric = TRUE)
    X = e$vectors %*% (sqrt(abs(e$values)) * t(e$vectors))
  } else {
    if (is.null(X)) base::stop("give X or gram")
    if (is.data.frame(X)) X = as.matrix(X)
    if (!is.double(X)) storage.mode(X) = "double"
  }
  if (anyNA(X)) base::stop("X must not contain missing values")
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(dim(X)))
    base::stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  stop = as.double(rep_len(stop, n_comps))
  fit = .spca_soft_cpp(X, n_comps, stop, as.double(tol),
                       as.integer(max_iter))
  W = fit$P
  dimnames(W) = list(colnames(X), paste0("sPC", seq_len(n_comps)))
  structure(list(
    weights = W, n_comps = n_comps, cardinality = colSums(W != 0),
    pev = compute_pev(W, X), coefficients = fit$Q, pev_pq = fit$pev_pq,
    pev_pq_comp = fit$pev_pq_comp, iterations = fit$iterations,
    totvar = fit$totvar, method_name = "sPCA_soft",
    method_descrip = paste0("Sparse PCA of Zou et al. (2006) with ridge ",
                            "Inf (soft thresholding, MEDA spcaZou)")),
    class = c("cspca", "spca", "list"))
  }, error = function(e) {
    base::stop("spca_soft(): ", conditionMessage(e), call. = FALSE)
  })
}
