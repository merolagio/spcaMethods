#' Sparse PCA by the generalized power method (GPower)
#'
#' Computes \code{n_comps} sparse weight vectors with the generalized power 
#' method of Journee, Nesterov, Richtarik and Sepulchre (2010), ported from the 
#' authors'
#' MATLAB code \code{GPower.m} to C++ (RcppEigen). Four variants are available:
#' single-unit with deflation or block (all components at once), each with an
#' \eqn{\ell_1} or \eqn{\ell_0} penalty, and the block variants optionally
#'  weighted by \code{mu}.
#'
#' The data matrix is passed to C++ by reference (an \code{Eigen::Map} onto R's
#' memory), so it is not copied as long as it is already a double-precision
#' matrix. An integer or logical matrix, or a data frame, is converted once.
#' The deflated matrices of the single-unit algorithm are never formed, so the
#' extra memory is of order \eqn{(n + p) n_comps}.
#'
#' @section Differences from GPower.m:
#' \itemize{
#'   \item In the block algorithms, MATLAB multiplies \code{rho} by the R factor
#'    of the QR decomposition used for the random start (\code{RHO*rho_max} is a
#'     matrix product), so the first component gets a negative threshold and is
#'     never sparse. By default (\code{rho_scale = "colnorm"}) the thresholds 
#'     are
#'     \eqn{\rho_j \mu_j \max_i \|a_i\|} (\eqn{\ell_1}) and
#'     \eqn{\rho_j \mu_j^2 \max_i \|a_i\|^2} (\eqn{\ell_0}), the bounds of the
#'      paper.
#'     \code{rho_scale = "matlab"} reproduces the MATLAB behaviour.
#'   \item In the \eqn{\ell_0} single-unit algorithm, a fully thresholded
#'    component is returned as a zero column; MATLAB returns NaN for it and for
#'     every later component.
#'   \item The unweighted block algorithm is used only when all \code{mu} 
#'   equal 1; MATLAB tests \code{sum(mu - 1) == 0}, which ignores e.g. 
#'   \code{mu = c(0.5, 1.5)}.
#' }
#'
#' @param X numeric matrix, \eqn{p \times n}: observations in rows, variables in
#'   columns (as in \code{GPower.m}). Usually column-centred.
#' @param rho sparsity parameters, relative to their upper bound, so values in
#'   \eqn{[0, 1)}; recycled to length \code{n_comps}. 
#'   Larger values give sparser weights.
#' @param n_comps number of components.
#' @param penalty \code{"l1"} or \code{"l0"}.
#' @param block if \code{FALSE}, components are computed one at a time with
#'   deflation; if \code{TRUE}, all \code{n_comps} components are computed
#'    together.
#'   Ignored when \code{n_comps = 1}.
#' @param mu weights of the block algorithm, length \code{n_comps}. Distinct weights
#'   break the symmetry between components (the paper uses decreasing values).
#'   Ignored when \code{block = FALSE}.
#' @param rho_scale \code{"colnorm"} (default) or \code{"matlab"}; 
#'   see the section on differences from GPower.m. Only used by the block
#'   algorithms.
#' @param init optional \eqn{p \times (n_comps - 1)} matrix used, together with the
#'   variable of largest norm, to start the block algorithm. By default it is
#'   drawn with \code{rnorm}, so \code{set.seed()} makes results reproducible.
#' @param iter_max maximum number of iterations.
#' @param eps tolerance on the relative change of the objective.
#'
#' @return an object of class \code{c("cspca", "spca", "list")} with
#'   \item{weights}{\eqn{ncol(X) \times n_comps} matrix of sparse weights,
#'     columns of unit norm (or zero when everything is thresholded). Signs
#'     are arbitrary. This is \code{Z} in \code{GPower.m}.}
#'   \item{n_comps}{number of components.}
#'   \item{cardinality}{number of non-zero weights of each component.}
#'   \item{pev}{Zou's adjusted proportion of explained variance of each
#'     component, from the QR decomposition of \code{X \%*\% weights}
#'     (\code{\link{compute_pev}}); the total is \code{sum(X^2)}.}
#'   \item{thresholds}{the absolute thresholds actually applied to each
#'     component.}
#'   \item{iterations}{iterations of the main loop (per component for the
#'     single-unit algorithm).}
#'   \item{method_name}{short label of the method, e.g.
#'     \code{"GPower_l1"} or \code{"GPower_l0_block"}.}
#'   \item{method_descrip}{one-line description of the method.}
#'
#' @references M. Journee, Y. Nesterov, P. Richtarik, R. Sepulchre (2010).
#'   Generalized power method for sparse principal component analysis.
#'   \emph{Journal of Machine Learning Research}, 11, 517--553.
#'
#' @examples
#' \dontrun{
#' Rcpp::sourceCpp("gpower_byref.cpp")   # not needed when built as a package
#' set.seed(1)
#' X = scale(matrix(rnorm(100 * 40), 100), scale = FALSE)
#' fit = gpower(X, rho = 0.2, n_comps = 3, penalty = "l1")
#' fit$cardinality
#' fit2 = gpower(X, rho = 0.1, n_comps = 3, penalty = "l0", block = TRUE,
#'               mu = c(1, 0.5, 0.25))
#' }
#'
#' @export
gpower = function(X, rho, n_comps = 1L, penalty = c("l1", "l0"),
                  block = FALSE, mu = rep(1, n_comps),
                  rho_scale = c("colnorm", "matlab"),
                  init = NULL, iter_max = 1000L, eps = 1e-4) {
  tryCatch({
  penalty = match.arg(penalty)
  rho_scale = match.arg(rho_scale)
  if (is.data.frame(X)) {
    X = as.matrix(X)
    # only copies when the input is not already double
    storage.mode(X) = "double"
  }
  if (anyNA(X)) stop("X must not contain missing values")
  
  
  p = nrow(X); 
  n = ncol(X)
  
  n_comps = as.integer(n_comps)
  if (n_comps < 1L || n_comps > min(p, n)) 
    stop("n_comps must be between 1 and min(nrow(X), ncol(X))")
  rho = as.double(rep_len(rho, n_comps))
  
  if (any(rho < 0)) stop("rho must be non-negative")
  
  iter_max = as.integer(iter_max)
  eps = as.double(eps)

  if (!block || n_comps == 1L) {
    fit = .gpower_single(X, rho, n_comps, penalty == "l1", iter_max, eps)
  } else {
    mu = as.double(mu)
    if (length(mu) != n_comps || any(mu <= 0))
      stop("mu must be a positive vector of length n_comps")
    if (is.null(init)) {
      init = matrix(rnorm(p * (n_comps - 1L)), p, n_comps - 1L)
    } else {
      init = as.matrix(init)
      storage.mode(init) = "double"
      if (!identical(dim(init), c(p, n_comps - 1L)))
        stop("init must be a nrow(X) x (n_comps - 1) matrix")
    }
    fit = .gpower_block(X, rho, mu, init, penalty == "l1",
                        rho_scale == "matlab", iter_max, eps)
  }
  block = block && n_comps > 1L
  W = fit$Z
  dimnames(W) = list(colnames(X), paste0("sPC", seq_len(n_comps)))
  structure(list(
    weights = W, n_comps = n_comps, cardinality = colSums(W != 0),
    pev = compute_pev(W, X), thresholds = fit$thresholds,
    iterations = fit$iterations,
    method_name = paste0("GPower_", penalty, if (block) "_block"),
    method_descrip = paste0("GPower (Journee et al. 2010), ", penalty,
                            " penalty, ",
                            if (block) "block" else "single-unit",
                            " algorithm")),
    class = c("cspca", "spca", "list"))
  }, error = function(e) {
    stop("gpower(): ", conditionMessage(e), call. = FALSE)
  })
}

#' Single-unit GPower step with a chosen start point
#'
#' Computes one sparse weight vector with the single-unit GPower iteration of
#' \code{GPower.m} (same iteration, stopping rule and pattern filling as \code{gpower()}
#' with \code{block = FALSE}), but starting from \code{x0} instead of the largest-norm
#' column of \code{X}. Plain R, for the uses \code{gpower()} does not cover:
#' \itemize{
#'   \item the PCA start of Guerra-Urzola et al. (2024), \code{GPower2.m}: component k
#'     starts at the k-th left singular vector of the data;
#'   \item warm starts along a path of \code{rho} values, by passing the \code{x} of the
#'     previous fit (needed to reach a given cardinality, since it jumps with \code{rho}).
#' }
#' Deflation is left to the caller: after a fit, use \code{X - fit$y \%*\% t(fit$z)} for
#' the next component.
#'
#' @param X numeric matrix, \eqn{p \times n}: observations in rows, variables in columns.
#' @param rho sparsity parameter relative to its upper bound, in \eqn{[0, 1)}.
#' @param x0 start vector of length \code{nrow(X)}.
#' @param penalty \code{"l1"} or \code{"l0"}.
#' @param iter_max maximum number of iterations.
#' @param eps tolerance on the relative change of the objective.
#'
#' @return a list with
#'   \item{z}{sparse weight vector of length \code{ncol(X)}, unit norm (or zero).}
#'   \item{y}{deflation vector: \code{X - y t(z)} is the deflated matrix.}
#'   \item{x}{final iterate, to warm-start a fit with a nearby \code{rho}.}
#'   \item{f}{final value of the objective.}
#'   \item{iter}{iterations used.}
#'
#' @examples
#' \dontrun{
#' X = scale(matrix(rnorm(30 * 200), 30))
#' U = svd(X, nu = 3, nv = 0)$u
#' fit1 = gpower_su(X, 0.5, U[, 1])                # PCA start
#' fit2 = gpower_su(X, 0.55, fit1$x)               # warm start at a larger rho
#' X2 = X - fit1$y %*% t(fit1$z)                   # deflate, then fit component 2
#' fit3 = gpower_su(X2, 0.5, U[, 2])
#' }
#'
#' @export
gpower_su = function(X, rho, x0, penalty = c("l1", "l0"), iter_max = 1000L,
                     eps = 1e-4) {
  penalty = match.arg(penalty)
  soft = function(a, t) sign(a) * pmax(abs(a) - t, 0)
  vnorm = function(a) sqrt(sum(a^2))
  na = sqrt(colSums(X^2))
  r = if (penalty == "l1") rho * max(na) else rho * max(na)^2
  x = x0 / vnorm(x0)
  f = numeric(0)
  iter = 1
  repeat {
    Ax = drop(crossprod(X, x))
    if (penalty == "l1") {
      tr = soft(Ax, r)
      f[iter] = sum(tr^2)
    } else {
      tr = pmax(Ax^2 - r, 0)
      f[iter] = sum(tr)
    }
    if (f[iter] == 0) break
    grad = if (penalty == "l1") X %*% tr else X %*% ((tr > 0) * Ax)
    x = drop(grad) / vnorm(grad)
    if ((iter > 2 && (f[iter] - f[iter - 1]) / f[iter - 1] < eps) ||
        iter > iter_max) break
    iter = iter + 1
  }
  Ax = drop(crossprod(X, x))
  if (penalty == "l1") {
    support = which((abs(Ax) - r) > 0)
    z = soft(Ax, r)
    if (any(z > 0)) z = z / vnorm(z)            # MATLAB: max(abs(z>0))>0 (sic)
    # pattern filling: power iteration on the support (single-unit case)
    zf = numeric(length(z))
    if (length(support) == 1) {
      zf[support] = 1
    } else if (length(support) > 1) {
      u = z[support]
      Xr = X[, support, drop = FALSE]
      fp = numeric(0)
      it = 1
      repeat {
        tmp = drop(crossprod(Xr, Xr %*% u))
        u = tmp / vnorm(tmp)
        fp[it] = -2 * sum(u * tmp)
        if ((it > 2 && abs(fp[it] - fp[it - 1]) / abs(fp[it - 1]) < 1e-6) ||
            it > 1000) break
        it = it + 1
      }
      zf[support] = u
    }
    z = zf
    y = drop(X %*% z)
  } else {
    z = Ax
    z[(Ax^2 - r) <= 0] = 0
    nz = vnorm(z)
    if (nz > 0) {
      z = z / nz
      y = x * nz
    } else y = x * 0
  }
  list(z = z, y = y, x = x, f = f[iter], iter = iter)
}
