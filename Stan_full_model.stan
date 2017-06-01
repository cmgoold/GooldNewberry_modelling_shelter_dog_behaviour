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
  simplex[K - 1] thresh_raw;      // raw thresholds
  matrix[4,Nid] z;                // scaled random effects (centered parameterisation)
  vector<lower=0>[4] sigmaID;     // standard deviations of random effects
  cholesky_factor_corr[4] L_Rho;  // Cholesky factor of the correlation matrix
  real alpha;                     // mean-model intercept
  vector[2] beta_day;             // day since arrival coefficients (linear/quadratic)
  real delta;                     // sd-model intercept
  vector[P] Beta1;                // BetaX = coefficients for dog-level predictors
  vector[P] Beta2;
  vector[P] Beta3;
  vector[P] Beta4;
  real<lower=0> Beta_sigma;       // shared sd for dog-level predictor coefficients
}

transformed parameters {
  ordered[K] thresh;     // new thresholds with fixed first and last
  matrix[Nid,4] v;       // unscaled random effects
  matrix[4,4] Rho;       // correlation matrix of random effects
  thresh[1] = L;
  thresh[2:K] = L + diff * cumulative_sum(thresh_raw);
  v = (diag_pre_multiply(sigmaID, L_Rho) * z)';
  Rho = L_Rho * L_Rho';           // correlation matrix = Cholesky factor * transpose(Choleksy factor)
}

model{
  vector[J] theta;                  // local parameter for ordinal categories

  //priors
  to_vector(z) ~ normal(0, 1);      // unit vector for scaled random effects
  L_Rho ~ lkj_corr_cholesky(2.0);   // LKJ prior on Cholesky factor of correlation matrix
  alpha ~ normal(meanY, 5);         // mean-model intercept prior
  beta_day ~ normal(0, 1);          // day since arrival coefficients priors
  delta ~ normal(0, 1);             // sd-model intercept prior
  sigmaID ~ cauchy(0, 2);           // priors on random effect SDs
  Beta_sigma ~ cauchy(0, 2);        // prior of shared sd among dog-level predictor coefficients
  Beta1 ~ normal(0, Beta_sigma);    // priors on dog-level predictor coefficients
  Beta2 ~ normal(0, Beta_sigma);
  Beta3 ~ normal(0, Beta_sigma);
  Beta4 ~ normal(0, Beta_sigma);

  // likelihood statement
  for(n in 1:N) {
    real eta;                                   // local parameter for mean-model
    real sigma;                                 // local parameter for sd-model
    eta = (alpha + v[ID[n],1] + X[n] * Beta1)
          + (beta_day[1] + v[ID[n],2] + (X[n] * Beta2)) * day[n]
          + (beta_day[2] + v[ID[n],3] + (X[n] * Beta3)) * day2[n];
    sigma = delta + v[ID[n],4] + X[n] * Beta4;

    // probability of ordinal category definitions
    theta[1] = normal_cdf( thresh[1] , eta, exp(sigma) );
    for (l in 2:K){
      theta[l] = fmax(0, normal_cdf(thresh[l], eta, exp(sigma) ) - normal_cdf(thresh[l-1], eta, exp(sigma)));
    }
    theta[J] = 1 - normal_cdf(thresh[K] , eta, exp(sigma));

  y[n] ~ categorical(theta);
  }
}

generated quantities {        // for calculating the log-likelihood
  vector[N] log_lik;
  vector[J] theta;

  for(n in 1:N) {
    real eta;
    real sigma;
    eta = (alpha + v[ID[n],1] + X[n] * Beta1)
          + (beta_day[1] + v[ID[n],2] + (X[n] * Beta2)) * day[n]
          + (beta_day[2] + v[ID[n],3] + (X[n] * Beta3)) * day2[n];
    sigma = exp(delta + v[ID[n],4] + X[n] * Beta4);

    theta[1] = normal_cdf( thresh[1] , eta, exp(sigma) );
    for (l in 2:K){
      theta[l] = fmax(0, normal_cdf(thresh[l], eta, exp(sigma) ) - normal_cdf(thresh[l-1], eta, exp(sigma)));
    }
    theta[J] = 1 - normal_cdf(thresh[K] , eta, exp(sigma));

    // log-likelihood calculation
    log_lik[n] = categorical_lpmf(y[n] | theta );
  }
}
