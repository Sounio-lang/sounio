`fit_pca` passes exclusive references into `standardize_data`, `center_data`, and `deflate_matrix`, so `stdlib/stats/multivariate/pca.sio` typechecks.
