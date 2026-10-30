#' Percentage explained variance of sparse components
#'
#' Computes the adjusted percentage explained variance (pev) of the components
#' t_j = X w_j, following Zou, Hastie and Tibshirani (2006). The adjusted
#' variance of t_j is R_jj^2 from the unpivoted QR decomposition T = QR of
#' T = X W. Components are taken in column order, so component j is adjusted
#' only for components 1, ..., j - 1. The pev is that adjusted variance divided
#' by the total variance of X.
#'
#' Supply exactly one of X (data) or S (covariance matrix).
#'
#' @param W numeric matrix of weights, p x r (a vector is treated as p x 1).
#'   Each column holds the weights of one component.
#' @param X numeric data matrix, n x p, centred by the caller. Default NULL.
#'   Used when S is NULL. The total variance is sum(X^2).
#' @param S numeric covariance matrix, p x p, symmetric and positive
#'   diagonal. Default NULL. Used when X is NULL. The total variance is
#'   trace(S), so S must have the same scale as the data used to build it.
#'
#' @return numeric vector of length r, the pev of each component in column
#'   order. Values are not normalised, so they can sum to more than 1 if W
#'   is not normalised.
#'
#' @details Calls the C++ functions \code{pevDataC} (data route, QR of X W) or
#'   \code{pevCovC} (covariance route, Cholesky of W' S W). Both give the same
#'   result when S = X'X / (n - 1).
compute_pev = function(W, X = NULL, S = NULL)
{
  W = as.matrix(W)
  storage.mode(W) = "double"
  if (is.null(X) == is.null(S))
    stop("compute_pev: supply exactly one of X (data) or S (covariance)")
  if (!is.null(X))
  {
    X = as.matrix(X)
    storage.mode(X) = "double"
    return(pevDataC(W, X))
  }
  S = as.matrix(S)
  storage.mode(S) = "double"
  pevCovC(W, S)
}
