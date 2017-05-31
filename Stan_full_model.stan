data {
  real L;                     // Lower fixed thresholds
  real<lower=L> U;            // Upper fixed threshold
  int<lower=1> K;             // Number of thresholds
  int<lower=2> J;             // Number of outcome levels
  int<lower=1> Nid;           // Number of dogs
  int<lower=0> N;             // Data length
  int<lower=0> P;             // Number of dog-level predictors in X
  int<lower=1,upper=J> y[N];  // Ordinal responses
  vector[N] day;              // Day (standardized)
  vector[N] day2;             // Day squared
  int ID[N];                  // ID number
  real meanY;                 // Arithmetic mean of ordinal responses
  row_vector[P] X[N];         // Matrix of dog-level predictors
}

transformed data {
  real<lower=0> diff;         // difference between upper and lower fixed thresholds
  diff = U - L;
}

parameters {
  simplex[K - 1] thresh_raw;   // raw thresholds
  matrix[4,Nid] z;
  vector<lower=0>[4] sigmaID;
  cholesky_factor_corr[4] L_Rho;  // Cholesky factor of the correlation matrix
  real alpha;
  vector[2] beta_day;
  real delta;
  vector[P] Beta1;
  vector[P] Beta2;
  vector[P] Beta3;
  vector[P] Beta4;
  real<lower=0> Beta_sigma;
}

transformed parameters {
  ordered[K] thresh;     // new thresholds with fixed first and last
  matrix[Nid,4] v;       // matrix of random effects on original scale
  matrix[4,4] Rho;       // Correlation matrix of random effects
  thresh[1] = L;
  thresh[2:K] = L + diff * cumulative_sum(thresh_raw);
  v = (diag_pre_multiply(sigmaID, L_Rho) * z)';
  Rho = L_Rho * L_Rho';           // multiply Cholesky factor by transpose of self to obtain correlation
}

model{
  vector[J] theta;                // local parameter for probability of ordinal categories
  to_vector(z) ~ normal(0, 1);    // vector of unit normals

  // prior distributions
  for(k in 2:K) thresh_raw[k] ~ normal(k+0.5,1);
  L_Rho ~ lkj_corr_cholesky(2.0);
  alpha ~ normal(meanY, 5);
  beta_day ~ normal(0, 1);
  delta ~ normal(0, 1);
  sigmaID ~ cauchy(0, 2);
  Beta_sigma ~ cauchy(0, 2);
  Beta1 ~ normal(0, Beta_sigma);
  Beta2 ~ normal(0, Beta_sigma);
  Beta3 ~ normal(0, Beta_sigma);
  Beta4 ~ normal(0, Beta_sigma);

  // likelihood
  for(n in 1:N) {
    real eta;
    real sigma;
    eta = (alpha + v[ID[n],1] + X[n] * Beta1)
          + (beta_day[1] + v[ID[n],2] + (X[n] * Beta2)) * day[n]
          + (beta_day[2] + v[ID[n],3] + (X[n] * Beta3)) * day2[n];
    sigma = exp(delta + v[ID[n],4] + X[n] * Beta4);

    theta[1] = normal_cdf( thresh[1] , eta, sigma );
    for (l in 2:K){
      theta[l] = fmax(0, normal_cdf(thresh[l], eta, sigma ) - normal_cdf(thresh[l-1], eta, sigma));
    }
    theta[J] = 1 - normal_cdf(thresh[K] , eta, sigma);

  y[n] ~ categorical(theta);
  }
}

generated quantities {   // for calculating the log likelihood
  vector[N] log_lik;
  vector[J] theta;
  for(n in 1:N) {
    real eta;
    real sigma;
    eta = (alpha + v[ID[n],1] + X[n] * Beta1)
          + (beta_day[1] + v[ID[n],2] + (X[n] * Beta2)) * day[n]
          + (beta_day[2] + v[ID[n],3] + (X[n] * Beta3)) * day2[n];
    sigma = exp(delta + v[ID[n],4] + X[n] * Beta4);

    theta[1] = normal_cdf( thresh[1] , eta, sigma );
    for (l in 2:K){
      theta[l] = fmax(0, normal_cdf(thresh[l], eta, sigma ) - normal_cdf(thresh[l-1], eta, sigma));
    }
    theta[J] = 1 - normal_cdf(thresh[K] , eta, sigma);

    log_lik[n] = categorical_lpmf(y[n] | theta );
  }
}
