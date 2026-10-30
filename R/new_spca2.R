#' Compute Cumulative Variances from Component Gram Matrices
#'
#' Cumulative explained variance of the first j components, for j = 1..k,
#' from the component Gram matrix G = A'A-type matrix and H, using
#' cv_j = trace(solve(G_j) H_j), where G_j and H_j are the leading j x j blocks.
#' @param G Symmetric positive definite k x k matrix.
#' @param H Symmetric k x k matrix with the same dimensions as G.
#' @return Numeric vector of length k, cumulative variances in raw units.
#' @noRd
tall_cumulative_variance = function(G, H) {
  k = ncol(G)
  vapply(seq_len(k), function(j) {
    ind = seq_len(j)
    spca:::trace_mat(spca:::ab(
      spca:::solve_spd(G[ind, ind, drop = FALSE]),
      H[ind, ind, drop = FALSE]))
  }, numeric(1))
}

#' Construct an SPCA Object Without a Large Covariance Matrix
#'
#' Uses the installed spca package's C++ wrappers. Source this file to use it
#' without rebuilding either package. X must use the same variable scaling as A.
#' @param A Numeric matrix of component weights (variables by components).
#' @param S Optional covariance matrix. When supplied, the existing constructor
#'   is used and S remains authoritative.
#' @param X Optional data matrix, centered internally when S is absent.
#' @param method_name Optional method label.
#' @param fat_matrix NULL selects observation-space calculations when X has
#'   more columns than rows and S is absent. TRUE forces that backend (requires
#'   X and no S); FALSE delegates to spca::new_spca.
#' @param pca_obj Optional object validated by spca::is.pca(), fitted to the
#'   same centered/scaled variables. Reuses its weights and variance references
#'   in either backend, without a PCA eigendecomposition. It must contain at
#'   least as many PCs as A has columns. Data are still needed as X or S.
#' @return An spca object with scores XA and least-squares variance summaries.
#' @details No p-by-p matrix is allocated by the fat backend. Dependent score
#'   columns and components beyond the numerical data rank are rejected.
#'   PC correlations can differ in sign from the covariance backend because
#'   eigenvector signs are arbitrary (and bases can differ for tied eigenvalues).
#'   The tall backend (fat_matrix = FALSE with pca_obj) avoids forming S when
#'   X is supplied: it works with the n x k scores and the p x k product X'B.
#'   When S is supplied it uses one product S [A V] and never forms S twice.
new_spca2 = function(A, S = NULL, X = NULL, method_name = NULL,
                     fat_matrix = NULL, pca_obj = NULL) {
  if (!requireNamespace("spca", quietly = TRUE))
    stop("The spca package is required", call. = FALSE)
  if (!is.null(pca_obj) && !isTRUE(spca::is.pca(pca_obj)))
    stop("pca_obj must be a valid pca object", call. = FALSE)
  if (!is.null(fat_matrix) &&
      (!is.logical(fat_matrix) || length(fat_matrix) != 1L ||
       is.na(fat_matrix)))
    stop("fat_matrix must be NULL, TRUE, or FALSE", call. = FALSE)
  if (isTRUE(fat_matrix) && (!is.null(S) || is.null(X)))
    stop("fat_matrix = TRUE requires X and S = NULL", call. = FALSE)
  if (is.data.frame(X)) X = as.matrix(X)
  if (is.null(fat_matrix))
    fat_matrix = is.null(S) && is.matrix(X) && nrow(X) < ncol(X)
  if (!fat_matrix && is.null(pca_obj))
    return(spca::new_spca(A = A, S = S, X = X,
                          method_name = method_name))

  if (is.data.frame(A)) A = as.matrix(A)
  if (!is.matrix(A) || !is.numeric(A) || any(!is.finite(A)) ||
      any(dim(A) == 0L))
    stop("A must be a nonempty finite numeric matrix", call. = FALSE)
  if (!is.null(X) && (!is.matrix(X) || !is.numeric(X) || any(!is.finite(X)) ||
      nrow(X) < 2L || ncol(X) != nrow(A))
      )
    stop("X must be finite numeric data with at least two rows and ncol(X) = nrow(A)",
         call. = FALSE)
  if (!is.null(method_name) &&
      (!is.character(method_name) || length(method_name) != 1L ||
       is.na(method_name)))
    stop("method_name must be NULL or a character scalar", call. = FALSE)
  norms = colSums(A^2)
  if (any(!is.finite(norms)) || any(norms <= 0))
    stop("Every weight column must have a finite positive norm", call. = FALSE)
  if (any(abs(norms - 1) > 1e-5))
    A = spca:::scale_columns(A, 2)
  if (is.null(X) && is.null(S))
    stop("S and X cannot both be NULL", call. = FALSE)
  k = ncol(A)
  if (!is.null(pca_obj)) {
    V = pca_obj$weights
    if (is.null(V)) V = pca_obj$loadings
    if (!is.matrix(V) || !is.numeric(V) || any(!is.finite(V)) ||
        nrow(V) != nrow(A) || ncol(V) < k)
      stop("pca_obj must contain compatible weights for at least ncol(A) PCs",
           call. = FALSE)
    V = V[, seq_len(k), drop = FALSE]
    pc_values = pca_obj$vexp_pc[seq_len(k)] * pca_obj$tot_var
    if (length(pca_obj$tot_var) != 1L || !is.finite(pca_obj$tot_var) ||
        pca_obj$tot_var <= 0 || length(pc_values) != k ||
        any(!is.finite(pc_values)) || any(pc_values <= 0))
      stop("pca_obj must contain positive finite PCA variances", call. = FALSE)
    if (!is.null(X) && !is.null(colnames(X)) && !is.null(rownames(V)) &&
        !identical(colnames(X), rownames(V)))
      stop("X and pca_obj variable names/order differ", call. = FALSE)
  }
  # Always center: scores, total variance and PCA references must agree.
  if (!is.null(X) && is.null(S))
    X = spca:::standardize_data(X, TRUE, FALSE)
  active = which(rowSums(A != 0) > 0L)
  Z = if (!is.null(X))
    spca:::ab(X[, active, drop = FALSE], A[active, , drop = FALSE]) else NULL
  if (!fat_matrix) {
    if (is.null(S)) {
      # Data route: S = X'X/(n-1) is never formed. Only the n x k scores
      # and the p x k product X'B are needed.
      totv = sum(X^2) / (nrow(X) - 1)
      if (!is.finite(totv) || totv <= 0)
        stop("X must have positive finite total variance", call. = FALSE)
      pc_values = pca_obj$vexp_pc[seq_len(k)] * totv
      XV = spca:::ab(X, V)
      if (!isTRUE(all.equal(unname(colSums(XV^2) / (nrow(X) - 1)),
                            unname(pc_values), tolerance = 1e-6)))
        stop("pca_obj variances are incompatible with X; check preprocessing",
             call. = FALSE)

      # Normalize scores for better conditioning; their span is unchanged.
      score_norms = colSums(Z^2)
      if (any(!is.finite(score_norms)) || any(score_norms <= 0))
        stop("Every component must have positive finite score variance",
             call. = FALSE)
      B = spca:::scale_columns(Z, 2)
      G = spca:::ata(B)
      if (rcond(G) <= max(dim(X)) * .Machine$double.eps)
        stop("Score columns are numerically linearly dependent", call. = FALSE)

      # B'SB = W'W/(n-1), with W = X'B (p x k)
      W = spca:::atb(X, B)
      H = spca:::ata(W) / (nrow(X) - 1)
      cv = tall_cumulative_variance(G, H)
      ve = diff(c(0, cv))
      # correlation of X a_j with X v_j: cov / (sd_scores * sd_pc)
      cor_pc = colSums(Z * XV) /
        sqrt((nrow(X) - 1) * colSums(Z^2) * pc_values)
      spc_cor = if (k == 1L) matrix(1, 1L, 1L) else cov2cor(G)
    } else {
      if (!is.matrix(S) || !is.numeric(S) || any(!is.finite(S)) ||
          !identical(dim(S), c(nrow(A), nrow(A))) ||
          !spca:::is_symmetric_fast(S))
        stop("S must be a compatible finite symmetric matrix", call. = FALSE)
      if (!is.null(X) && !isTRUE(all.equal(colSums(X^2) / (nrow(X) - 1),
                                          diag(S), check.attributes = FALSE,
                                          tolerance = 1e-6)))
        stop("diag(S) is not compatible with X", call. = FALSE)
      totv = spca:::trace_mat(S)
      pc_values = pca_obj$vexp_pc[seq_len(k)] * totv

      # one product with S gives both SA and SV
      SAV = spca:::ab(S, cbind(A, V))
      SA = SAV[, seq_len(k), drop = FALSE]
      SV = SAV[, k + seq_len(k), drop = FALSE]
      if (!isTRUE(all.equal(unname(SV),
                            unname(sweep(V, 2, pc_values, "*")),
                            tolerance = 1e-6)))
        stop("pca_obj eigenvectors/variances are incompatible with S",
             call. = FALSE)

      # G = A'SA, H = A'S^2A = (SA)'(SA)
      G = spca:::atb(A, SA)
      if (any(diag(G) <= 0))
        stop("Every component must have positive variance", call. = FALSE)
      H = spca:::ata(SA)
      cv = tall_cumulative_variance(G, H)
      ve = diff(c(0, cv))
      cor_pc = colSums(A * SV) / sqrt(diag(G) * pc_values)
      spc_cor = if (k == 1L) matrix(1, 1L, 1L) else cov2cor(G)
    }
  } else {
  K = spca:::aat(X) / (nrow(X) - 1)
  if (is.null(pca_obj)) {
  ee = spca:::eigen_sym(K)
  values = as.numeric(ee$values)
  rank_tol = max(dim(X)) * .Machine$double.eps * max(abs(values))
  if (k > sum(values > rank_tol))
    stop("Number of components exceeds the numerical rank of X", call. = FALSE)
  pc_values = values[seq_len(k)]
  pc_directions = ee$vectors[, seq_len(k), drop = FALSE]
  } else {
    pc_values = pca_obj$vexp_pc[seq_len(k)] * spca:::trace_mat(K)
    pc_scores = spca:::ab(X, V)
    if (!isTRUE(all.equal(unname(colSums(pc_scores^2)/(nrow(X)-1)),
                          unname(pc_values), tolerance = 1e-6)))
      stop("pca_obj variances are incompatible with X; check preprocessing",
           call. = FALSE)
    pc_directions = spca:::scale_columns(pc_scores, 2)
  }
  totv = spca:::trace_mat(K)
  if (!is.finite(totv) || totv <= 0)
    stop("X must have positive finite total variance", call. = FALSE)

  # Normalize scores for better conditioning; their span is unchanged.
  score_norms = colSums(Z^2)
  if (any(!is.finite(score_norms)) || any(score_norms <= 0))
    stop("Every component must have positive finite score variance", call. = FALSE)
  B = spca:::scale_columns(Z, 2)
  G = spca:::ata(B)
  if (rcond(G) <= max(dim(X)) * .Machine$double.eps)
    stop("Score columns are numerically linearly dependent", call. = FALSE)
  H = spca:::atb(B, spca:::ab(K, B))
  cv = vapply(seq_len(k), function(j) {
    ind = seq_len(j)
    spca:::trace_mat(spca:::ab(
      spca:::solve_spd(G[ind, ind, drop = FALSE]),
      H[ind, ind, drop = FALSE]))
  }, numeric(1))
  ve = diff(c(0, cv))
  cor_pc = diag(spca:::atb(B, pc_directions))
  spc_cor = if (k == 1L) matrix(1, 1L, 1L) else spca:::makeCorScoresC(Z)
  }
  if (!is.null(pca_obj)) {
    # pca(data) stores sums of squares; pca(covariance) stores variances.
    totals = as.numeric(pca_obj$tot_var)
    n_ref = if (!is.null(X)) nrow(X) else pca_obj$n_obs
    if (length(n_ref) == 1L && is.finite(n_ref) && n_ref > 1)
      totals = c(totals, pca_obj$tot_var / (n_ref - 1))
    if (!any(vapply(totals, function(t) isTRUE(all.equal(
        as.numeric(totv), as.numeric(t), tolerance = 1e-6)), logical(1))))
      stop("pca_obj total variance is incompatible with X or S", call. = FALSE)
  }
  indices = lapply(seq_len(k), function(j) which(A[, j] != 0))
  weights_list = lapply(seq_len(k), function(j) A[indices[[j]], j])
  contributions = spca:::make_contributions(A)
  dimnames(contributions) = dimnames(A)
  dimnames(spc_cor) = list(paste0("sPC", seq_len(k)),
                           paste0("sPC", seq_len(k)))
  structure(list(
    weights = A, contributions = contributions, n_comps = k,
    cardinality = colSums(A != 0), vexp = ve / totv,
    vexp_pc = pc_values / totv, cvexp = cv / totv,
    rvexp = ve / pc_values, rcvexp = cv / cumsum(pc_values),
    cor_with_pc = pmax(-1, pmin(1, cor_pc)), tot_var = totv,
    indices = indices, weights_list = weights_list, scores = Z,
    spc_cor = spc_cor, method_name = method_name),
    class = c("spca", "list"))
}
