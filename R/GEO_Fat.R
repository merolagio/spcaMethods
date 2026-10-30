


#GEO contains same elements as Original but
# 1. incorrect observations (3) removed
# 2. variables centred and scaled, keeping the zero columns (wtf)
# 3. 41675 columns as in Camacho instead of 43893  as in Guerra

GEO = readRDS("data/geo_CamachoIII.rds")
names(GEO)

geo = GEO$X
dim(geo)
ind_zv = (colSums(geo^2) > 0)
s = apply(geo, 2, stats::sd)
cat("zero-variance columns:", sum(s == 0), "\n")
# MEDA gives these a dummy scale; once centred they are all zeros either way
s[s == 0] = 1

geo = scale(geo, scale = s)
apply(geo[, 1:10], 2, mean)
apply(geo[, 1:10]^2, 2, mean)
anyNA(geo)

mm = apply(geo, 2, mean)
ss = apply(geo, 2, sd)
  
apply(geo^2, 2, sum)/(nrow(geo))
summary(ss)
sum(ss)
ss[1:3]
any(abs(mm) > 1e-6)
R = tcrossprod(geo)
dim(R)
ee = eigen(R)
sum(ee$values)

rownames(geo) = GEO$obs$obs

geo_classification = GEO$obs$class

saveRDS(geo, "data/geo.rds")
saveRDS(geo_classification, "data/geo_classification.rds")


#READ DATA==========
geo = readRDS("data/geo.rds")
geo_classification = readRDS("data/geo_classification.rds")

source("C:/Users/merol/Dropbox/Papers/Spca_Methods_comparison/cspca/R/new_spca2.R", echo = TRUE)
##PCA===============
library(spca)

#geo = spca:::scaleC(X, center = T, scale = F)
geo_pca = pca(geo, fat_matrix = T, qq_plot = F)
mp_qqplot(geo_pca, n_fitline = -3)

#spca================
geo_cspca95 = spca(geo, n_comps = 3)
summary(geo_cspca95)

geo_cspca90 = spca(geo, n_comps = 3, alpha = 0.90)
summary(geo_cspca90)

geo_uspca90 = spca(geo, n_comps = 3, alpha = 0.90, method = "u")
summary(geo_uspca90)

geo_uspca95 = spca(geo, n_comps = 3, alpha = 0.95, method = "u")
summary(geo_uspca95)

show_correlations(geo_cspca90)
show_correlations(geo_uspca90)

compare_spca(list(geo_cspca90, geo_uspca90))

compare_spca(list(geo_cspca95, geo_uspca95), print_weights = T)
show_correlations(geo_cspca95)
show_correlations(geo_uspca95)

#ZOU=====================

capture.output(
  Rcpp::sourceCpp("src/spca_zou.cpp"),
  type = "output",
  file = "Compile_output.txt"
)
source("R/spca_zou_wrapper.R")   # spca_zou(), for Camacho's sPCA model

# Camacho et al.'s model: Zou's sPCA, ridge 0, 4,4,4 non-zero weights (paper: 32.6%)

n_comps = 3L
geo_Zspca = spca_zou(geo, n_comps, para = c(4, 4, 4), lambda = 0, sparse = "varnum", center = FALSE)
#geo_Zspca = geo_zspca

geo_zspca = new_spca2(geo_Zspca$P, X = geo, fat_matrix = TRUE, 
                       method_name = "Zspca", pca_obj = geo_pca)

compare_spca(list(geo_cspca90, geo_zspca), methods_names = c("cSPCA90", "ZSPCA"))


summary(geo_zspca)
show_correlations(geo_zspca)
show_correlations(geo_cspca90)


runs$spca_camacho 
aa = list(W = sp$P, rho = rep(NA, n_comps))
names(aa)
aa$rho
rm(GEO)
gc()
## zou soft lambda = infty================
target = 4L
geo_SZspca = geo_szspca
geo_SZspca = spca_soft(geo, n_comps, stop = -rep(target, n_comps))
colSums(geo_SZspca$P != 0)
geo_SZspca$pev_pq


geo_szspca = new_spca2(geo_SZspca$P, X = geo, fat_matrix = TRUE, 
                      method_name = "SZspca", pca_obj = geo_pca)
summary(geo_szspca)
summary(geo_cspca90)
show_correlations(geo_zspca)
show_correlations(geo_szspca)
show_correlations(geo_cspca90)

compare_spca(list(geo_cspca90, geo_zspca, geo_szspca), 
             methods_names = c("cSPCA90", "ZSPCA", "SZSPCA"))


#Gpower====================
op = options(useFancyQuotes = FALSE)
capture.output(
  Rcpp::sourceCpp("src/gpower_byref.cpp", verbose = TRUE),
  type = "output",
  file = "Compile_output.txt"
)
capture.output(
  Rcpp::sourceCpp("src/spca_zou.cpp", verbose = TRUE),
  type = "output",
  file = "Compile_output.txt"
)

source("R/gpower_wrapper.R")     # gpower() and gpower_su()
source("R/spca_zou_wrapper.R")   # spca_zou(), for Camacho's sPCA model



#Gpower=================
rho = 0.825

##too sensitive 
# no warm start: C++ gpower(), starts at the largest-norm column of geo
st = Sys.time()
fit_cold = gpower(geo, rho = rho, n_comps = 3, penalty = "l1")
sum(fit_cold$Z != 0)
fn = Sys.time()
fn - st

apply(abs(fit_cold$Z), 2, function(x) sum(x > 1e-5 ))

# warm start: gpower_su() along rho = 0.5, ..., rho; each fit starts
# from the final iterate x of the previous fit
x0 = geo[, which.max(colSums(geo^2))]   # same start as gpower()
rho_path = seq(0.5, rho, length.out = 41)
fit_warm = gpower_su(geo, rho_path[1], x0, "l1")
for (r in rho_path[-1]) fit_warm = gpower_su(geo, r, fit_warm$x, "l1")
names(fit_warm)
length(fit_warm$z)
sum(fit_warm$z != 0)





