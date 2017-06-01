#****************************************************************************************
# R script for: Goold & Newberry (2017). "Modelling personality, plasticity and predictability in shelter dogs"
#****************************************************************************************

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(ggplot2)
library(rethinking)
library(data.table)
library(MASS)
library(reshape)

# Set the working directory!
# setwd("")

#=================================================================================
# Create figure 1
#=================================================================================

# simulate 100 individuals' reaction norms; correlation of 0.4 between 5 successive time points
Nid <- 100
set.seed(2017)
out <- mvrnorm(Nid, 
               mu = c(0,0,0,0,0), 
               Sigma = matrix(c(1,rep(0.4,4),
                                0.4,1,rep(0.4,3),
                                0.4,0.4,1,rep(0.4,2),
                                rep(0.4,3),1,0.4,
                                rep(0.4,4),1), ncol = 5), empirical = TRUE)

id = c(1:Nid)

out <- as.matrix(cbind(out, id))
colnames(out) <- c(1:5, "id")

outL = melt(out, id.vars=c("id"))   # long data format
outL = outL[-c((nrow(outL)-(Nid-1)):nrow(outL)), ]

df = outL[with(outL, order(X1)),]
colnames(df) <- c("ID","x","y")
rownames(df) <- 1:nrow(df)
df$ID = as.factor(df$ID)

df_1a <- df

ggplot(df_1a, aes(x = x, y = y, group=ID)) +
  geom_smooth(method="lm", se=FALSE, colour="black", lwd=0.5) +
  xlab("\nRepeated measurements") + 
  ylab("Behaviour\n") +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x=element_text(size=15, colour="black"),
        axis.text.y=element_text(size=15, colour="black"),
        axis.title.x=element_text(size=20, colour="black"),
        axis.title.y=element_text(size=20, colour="black"),
        legend.position = "none")

ggsave("Fig1a.png", last_plot(), width = 6, height = 5)

set.seed(123)
df_1b <- df[df$ID %in% sample(1:Nid, 4, replace=F),]
df_1b$ID <- rep(1:4, each=5)

ggplot(df_1b, aes(x = x, y = y, group=ID)) +
  geom_point() + 
  geom_smooth(method="lm", se=TRUE, colour="black", lwd=0.5) +
  facet_wrap(~ ID) + 
  xlab("\nRepeated measurements") + 
  ylab("Behaviour\n") +
  theme_bw() +
  theme(strip.text = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x=element_text(size=15, colour="black"),
        axis.text.y=element_text(size=15, colour="black"),
        axis.title.x=element_text(size=20, colour="black"),
        axis.title.y=element_text(size=20, colour="black"),
        legend.position = "none")

ggsave("Fig1b.png", last_plot(), width=6, height = 5)

#=================================================================================
# load the raw sample data on dog's longitudinal behaviour
#=================================================================================

my_data <- read.csv("Raw_sample_data.csv")

# COLUMN NAMES: // id (id number for each dog),
#               // day (day since arrival)
#               // aggCodes (aggregated behaviour codes used for analysis)
#               // nObs (number of observations on each dog)
#               // total_days (total number of days at the shelter)
#               // meanAge (average age in years while at the shelter)
#               // sex 
#               // weight (average weight in kg while at the shelter)
#               // neutered (neutered status - neutered on arrival, not neutered, on site, not examined)
#               // sourceType (how the dog arrived at the shelter: gift, stray, returned)
#               // site (which site the dogs were at: Battersea (London), Old Windsor, Brands Hatch)

#=================================================================================
# The number of observations and total number of days dogs were at the shelter 
# are both of interest, but they are correlated with one another.  
# -- Take the residuals of total_days instead with respect to nObs using a Gamma GLM.
#=================================================================================

my_data$resid_total_days <- residuals(glm(total_days+0.00001 ~ nObs , data = my_data , family = Gamma))

#=================================================================================
# Prepare the stan data. First, subset the data if running on smaller models
#=================================================================================

set.seed(12345)
sample_dogs <- sample(1:length(unique(my_data$id)), 
                      length(unique(my_data$id)) ,       # <--- change line to a smaller number if subsetting
                      replace=FALSE)

stan_data <- my_data[my_data$id %in% sample_dogs, ]

stan_data$id <- rep(1:length(sample_dogs),
                    unlist(lapply(split(stan_data, stan_data$id), FUN = nrow)))

#=================================================================================
# standardise metric variables, denoted with a Z at the end of the name 
#=================================================================================

stan_data$dayZ <- (stan_data$day - mean(stan_data$day)) / sd(stan_data$day)
stan_data$nObsZ <- (stan_data$nObs - mean(stan_data$nObs)) / sd(stan_data$nObs)
stan_data$resid_total_daysZ <- (stan_data$resid_total_days - mean(stan_data$resid_total_days)) / sd(stan_data$resid_total_days)
stan_data$mean_ageZ <- (stan_data$meanAge - mean(stan_data$meanAge)) / sd(stan_data$meanAge)
stan_data$weightZ <- (stan_data$weight - mean(stan_data$weight)) / sd(stan_data$weight) 

#=================================================================================
# build dog-level predictor matrix (sum-to-zero deflections)
#=================================================================================

stan_data[, c("sex","sourceType","site","neutered")] <- apply(stan_data[,c("sex","sourceType","site","neutered")], 
                                                                     2 , as.factor )

X <- model.matrix( ~ nObsZ + resid_total_daysZ + mean_ageZ + weightZ + sex + sourceType + site + neutered , 
                   data = stan_data , 
                   contrasts.arg = list(sex = "contr.sum", sourceType = "contr.sum", 
                                        site = "contr.sum", neutered = "contr.sum"))[,-1]

#=================================================================================
# Stan data list
#=================================================================================

stan_data_list = list(y = stan_data$aggCodes , N = nrow(stan_data), 
                      day = stan_data$dayZ , day2 = stan_data$dayZ^2,
                      X = X, P = ncol(X) ,
                      ID = stan_data$id, Nid = length(unique(stan_data$id)),
                      J = 6 , K = 5, L = 1.5, U = 5.5, 
                      meanY = mean(stan_data$aggCodes))

#=================================================================================
# set Stan run specifics (using the full model)  
#=================================================================================

stanModel <- "Stan_full_model.stan"
nIter = 5000; nWarmup = 2500; nChains = 4; nCores = 4
display_params <- c( "alpha", "beta_day", "sigmaID", 
                     "Rho", "L_Rho", 
                     "Beta1" , "Beta2" , "Beta3", "Beta4", 
                     "delta" , "Beta_sigma"  , 
                     "thresh_raw", "thresh"
)

#=================================================================================
# run Stan: depending on the data set, this could take a long time! 
# In the data set here, there were ~ 20,000 rows of data (3,263 dogs with varying numbers of data points) 
# and the model is estimating tens-of-thousands of parameters in the model block.

# It is best to run the model thoroughly on different sized subsets first to get an idea of the 
# running time, try to optimise the code as much as possible, and then go for a long run of the chains. 
#=================================================================================

startTime = proc.time()
stan_fit <- stan(file = stanModel, data = stan_data_list, init = 0,
                chains = nChains, warmup = nWarmup, iter = nIter, cores = nCores)
proc.time() - startTime

print(stan_fit , pars = display_params , digits_summary = 4 , probs = c(0.025, 0.975))

# Please get in touch (conor.goold@nmbu.no) if you would like the full MCMC output for the paper.
# It is around 7 GB in size, so putting it on Github is not really possible. 

#=================================================================================
######################## post-process Stan results ################################# 

# NB: Plots of figures are not in the same order as the paper, to allow a clearer workflow
#=================================================================================

# takes a couple of minutes
post_samples <- as.data.frame(stan_fit)

#=================================================================================
# Calculate WAIC (following McElreath (2015): Statistical Rethinking)
#=================================================================================

waic <- function( log_lik , data_length ) {
    lppd <- sapply(1:data_length, 
                   function(obs) log_sum_exp(log_lik[,obs]) - log(nrow(log_lik) ) 
                   )
    p_waic <- sapply(1:data_length, 
                    function(obs) var(log_lik[ , obs]) 
                    )
    waic <- (-2) * ( sum(lppd) - sum(p_waic) )
    se_waic <- sqrt( data_length * var( -2 * ( lppd - p_waic ) ) )
    list(WAIC = waic , 
         Standard.error = se_waic , 
         In_sample_dev = (-2) * sum( lppd )
        )
}

full_model_waic <- waic(log_lik = post_samples[,grep("log_lik",colnames(post_samples))],
                  data_length = nrow(my_data) 
                  )

#===== to save space, the MCMC matrices for other models are not provided.    ===============
#===== But the WAIC_results.csv file contains the WAIC results for all models ===============

waic_compare <- read.csv("WAIC_results.csv")

waic_compare$model <- factor(waic_compare$model, unique(waic_compare$model))

ggplot(waic_compare, aes(x = model, y = WAIC) ) +
  scale_y_continuous(limits = c(38000,46000)) +
  geom_point(size=2) + 
  geom_errorbar(aes(ymin = low_SE, ymax = high_SE), lwd=0.8, width=0) +
  labs(x = "", y = "\nWAIC") +
  theme_bw() +
  theme(axis.text = element_text(colour="black", size=15), 
        axis.title = element_text(colour="black", size=20)) +
  coord_flip()

ggsave(filename = "Fig2.png", last_plot(), width = 6, height = 3)

#=================================================================================
# Some functions for transforming log-normal parameters to original scales and a mode function
#=================================================================================

ModeLN <- function(mu, sigma) {
  exp(mu - sigma^2)
}

MeanLN <- function(mu, sigma) {
  exp(mu + (sigma^2/2) )
}

SdLN <- function( mu, sigma ) { 
  sqrt(  exp( (2*mu)+sigma^2 )*(exp(sigma^2)-1) )
}

# normal mode function
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#=================================================================================
# plot probabilities of different categories 
#=================================================================================

Nid <- length(unique(stan_data$id))
day_seq <- seq( min( stan_data$dayZ ), max(stan_data$dayZ), length = 31 )
chain_length = nrow(post_samples)

b0 = post_samples[,"alpha"]
b1 = post_samples[,"beta_day[1]"]
b2 = post_samples[,"beta_day[2]"]
sigma = exp(post_samples[,"delta"]) #MeanLN(mu = post_samples[, "delta"], sigma = post_samples[,"sigmaID[4]"])
thresh = post_samples[, grep("^thresh", colnames(post_samples))]
thresh = thresh[,5:9]
prob_cats <- rep(list(list()),6)

prob_cats[[1]] <- sapply(day_seq , 
                     function(x) { 
                       mu <- b0 + b1 * x + b2 * x^2 
                       prob <- pnorm( (as.numeric(thresh[ , 1]) - mu ) / sigma )
                       prob
                     }
                     )

for( j in 2:5 ) {  
  prob_cats[[j]] <-  sapply(day_seq , 
                            function(x) { 
                              mu <- b0 + b1 * x + b2 * x^2 
                              prob <- pnorm( (as.numeric(thresh[ , j]) - mu ) / sigma )  - 
                                                      pnorm( (as.numeric(thresh[ , j-1]) - mu ) / sigma )
                              prob
                            }
                    )
  }


prob_cats[[6]] <- sapply(day_seq , 
                         function(x) { 
                           mu <- b0 + b1 * x + b2 * x^2 
                           prob <- 1 - pnorm( (as.numeric(thresh[ , 5]) - mu ) / sigma )
                           prob
                         }
)

png(filename = "Fig3a.png", width=1500, height = 1200, res = 200)
par(mar=c(4.5,4.5,1,1))
plot( x = seq(0,30,1), y = seq(0,1,length.out = 31), type="n" , 
      xlab = "Day since arrival", ylab = "Probability of sociability code", 
      cex.axis=1 , cex.lab = 1.5)
legend(0,1, legend=c("Friendly", "Excited", "Independent", "Submissive", "Amber", "Red"),
       lty=1:6,  box.lty=0, cex = 1 )
for( i in 1:6 ){ 
  lines(seq(0,30,1) , apply(prob_cats[[i]], 2, mean ), lty = i)
  shade( apply(prob_cats[[i]],2,function(z) HPDI(z,0.95)), seq(0,30,1))
}
dev.off()


#=================================================================================
# plot average day ~ behaviour and individual curves 
#=================================================================================

pred <- sapply( day_seq , function(z) post_samples[, "alpha"] + post_samples[ , "beta_day[1]"] * z + 
                                                                post_samples[ , "beta_day[2]"] * z^2 )

pred_mu <- apply(pred, 2, mean )
pred_hdi <- apply(pred, 2 , function(z) HPDI(z, 0.95))

plot( jitter(aggCodes) ~ jitter(dayZ), data = stan_data ,
      col = col.alpha("black", 0.1), ylim = c(0,6.5),
      xlab = "Day since arrival", ylab = "Sociability score",
      cex.axis = 1, cex.lab = 1.5)
lines( day_seq , pred_mu, lwd = 3)
shade( pred_hdi, day_seq)

# arrange random effects to prepare for plotting
random_intercepts <- post_samples[,"alpha"] + 
                     post_samples[, grep("v", colnames(post_samples))][1:Nid] 
random_slopes_linear <- post_samples[,"beta_day[1]"] + 
                        post_samples[, grep("v", colnames(post_samples))][(Nid+1):(Nid*2)]
random_slopes_quad <- post_samples[,"beta_day[2]"] + 
                      post_samples[, grep("v", colnames(post_samples))][(Nid*2+1):(Nid*3)]
random_res_ints <- exp(post_samples[,"delta"] + 
                         post_samples[, grep("v", colnames(post_samples))][(Nid*3+1):(Nid*4)])

# split the raw data and make predictions
split_data <- split(stan_data$dayZ , stan_data$id)
individual_preds <- rep(list(list()), Nid)
for( i in 1:Nid ) { 
      individual_preds[[i]] <- sapply(split_data[[i]], 
                            function(z) random_intercepts[,i] + random_slopes_linear[,i]*z+random_slopes_quad[,i]*z^2)
}

#============ if simulated predictions are required, use this code =================
# simList <- rep( list(list()), Nid ) 
# for(i in 1:Nid){
#   simList[[i]] <- sapply(split_data[[i]], 
#                          function(z) 
#                            rnorm(n=chain_length,
#                                  mean = (random_intercepts[,i] + random_slopes_linear[,i]*z+random_slopes_quad[,i]*z^2),
#                                  sd = random_res_ints[,i])
#   )
# }

# create a data frame to hold predictions
individual_curves <- data.frame( id = stan_data$id, 
                                 day = stan_data$day , 
                                 dayZ = stan_data$dayZ ,
                                 code = stan_data$aggCodes, 
                                 pred_mu = unlist(lapply(individual_preds, function(z) apply(z, 2, mean ))), 
                                 pred_hdi_low = unlist(lapply(individual_preds, function(z) apply(z,2,function(x)HPDI(x,0.95))[1,])) ,
                                 pred_hdi_high = unlist(lapply(individual_preds, function(z) apply(z,2,function(x)HPDI(x,0.95))[2,]))
                                 #, sim_mu = unlist(lapply(simList, function(z) apply(z, 2, mean))),
                                 #sim_hdi_low = unlist(lapply(simList, function(z) apply(z,2,function(x)HPDI(x,0.95))[1,])) ,
                                 #sim_hdi_high = unlist(lapply(simList, function(z) apply(z,2,function(x)HPDI(x,0.95))[2,]))
                                 )

# plot all curves over the raw data
png(filename = "Fig3b.png", width=1500, height=1200, res=200)
par(mar=c(4,4.5,2,1))
plot( jitter(aggCodes) ~ dayZ, data = stan_data , 
      type="n", xaxt = "n" , 
      ylim = c(min(individual_curves$pred_mu),6),  
      xlab = "Day since arrival", ylab = "Sociability (latent scale)", 
      cex.axis = 1, cex.lab = 1.5)
at <- unique(day_seq)
newLabels = 0:30
axis( side = 1 , at=at , labels = newLabels, cex.axis =1)
for( i in 1:Nid ) { 
    lines(individual_curves[individual_curves$id %in% i, "dayZ"] , 
          individual_curves[individual_curves$id %in% i, "pred_mu"], 
          col = col.alpha("black", 0.2) )
}
lines( day_seq , pred_mu, lwd = 3, col = "blue")
dev.off()


# make counter factual predictions
# i.e. days 0 to 30 for each dog regardless of data
counterfac_preds <- rep(list(list()), Nid)
for( i in 1:Nid ) { 
  counterfac_preds[[i]] <- sapply(day_seq, 
                                  function(z) random_intercepts[,i] + random_slopes_linear[,i]*z+random_slopes_quad[,i]*z^2)
}

#======== if simulated predictions are required, use this data =====================
# counterfac_sims <- rep(list(list()), Nid)
# for(i in 1:Nid){
#   counterfac_sims[[i]] <- sapply(day_seq, 
#                          function(z) 
#                            rnorm(n=chain_length,
#                                  mean = (random_intercepts[,i] + random_slopes_linear[,i]*z+random_slopes_quad[,i]*z^2),
#                                  sd = random_res_ints[,i])
#   )
# }

# takes a long time if simulated predictions are included!
counterfac_curves <- data.frame( id = rep(1:Nid, each = length(day_seq)) ,  
                                 day = rep(0:30, Nid ) , 
                                 dayZ = rep(day_seq, Nid),
                                 pred_mu = unlist(lapply(counterfac_preds, function(z) apply(z, 2, mean ))), 
                                 pred_hdi_low = unlist(lapply(counterfac_preds, function(z) apply(z,2,function(x)HPDI(x,0.95))[1,])) ,
                                 pred_hdi_high = unlist(lapply(counterfac_preds, function(z) apply(z,2,function(x)HPDI(x,0.95))[2,]))
                                 # , sim_mu = unlist(lapply(counterfac_sims, function(z) apply(z, 2, mean ))), 
                                 # sim_hdi_low = unlist(lapply(counterfac_sims, function(z) apply(z,2,function(x)HPDI(x,0.95))[1,])) ,
                                 # sim_hdi_high = unlist(lapply(counterfac_sims, function(z) apply(z,2,function(x)HPDI(x,0.95))[2,])))
                                )

set.seed(2017)
sample_dogs <- sample(1:Nid, 20, replace = FALSE )
sample_data <- counterfac_curves[ counterfac_curves$id %in% sample_dogs, ]
sample_data$id <- rep(1:20, each = 31 )
raw_data <- stan_data[stan_data$id %in% sample_dogs, ]
raw_data$id <- rep(1:20, unlist(lapply(split(raw_data, raw_data$id), nrow)))

ggplot( raw_data , aes(x = day , y = aggCodes) ) + 
  geom_point( ) + 
  facet_wrap(~ id ) + 
  geom_line(data = sample_data, aes(x = day , y = pred_mu)) + 
  geom_line(data = sample_data, aes(x = day, y=pred_hdi_low ), linetype="dashed") +
  geom_line(data = sample_data, aes(x = day, y=pred_hdi_high ), linetype="dashed") +
  coord_cartesian(xlim=c(0,30), ylim=c(-5,6)) +
  scale_x_continuous(breaks=seq(0,30,10)) +
  ylab("Sociability (ordinal and latent scale)\n") + 
  xlab("\nDay since arrival") +
  theme_bw() + 
  theme( strip.text = element_text(colour="black", size=10),
         panel.grid.major = element_blank() , 
         panel.grid.minor = element_blank() ,
         axis.text.x = element_text(colour ="black", size=15), 
         axis.text.y = element_text(colour ="black", size=15),
         axis.title = element_text(colour = "black", size=20) )

ggsave("Fig6.png", last_plot() , width = 10, height = 8)

#=================================================================================
# plot random effects
#=================================================================================

random_effects <- data.frame(id = 1:Nid , 
                             intercept_mu = apply(random_intercepts , 2, median) , 
                             intercept_low95 = apply(random_intercepts , 2, function(z) HPDI(z, 0.95))[1,] ,
                             intercept_high95 = apply(random_intercepts , 2, function(z) HPDI(z, 0.95))[2,] ,
                             
                             lin_slope_mu = apply(random_slopes_linear , 2, median) , 
                             lin_slope_low95 = apply(random_slopes_linear , 2, function(z) HPDI(z, 0.95))[1,] ,
                             lin_slope_high95 = apply(random_slopes_linear , 2, function(z) HPDI(z, 0.95))[2,] ,
                             
                             quad_slope_mu = apply(random_slopes_quad , 2, median) , 
                             quad_slope_low95 = apply(random_slopes_quad , 2, function(z) HPDI(z, 0.95))[1,] ,
                             quad_slope_high95 = apply(random_slopes_quad , 2, function(z) HPDI(z, 0.95))[2,] ,
                            
                             scale_mu = apply(random_res_ints , 2, median) , 
                             scale_low95 = apply(random_res_ints , 2, function(z) HPDI(z, 0.95))[1,] ,
                             scale_high95 = apply(random_res_ints , 2, function(z) HPDI(z, 0.95))[2,] 
                             
                             )
  
  
ggplot( random_effects , aes( x = id , y = intercept_mu )) + 
  geom_errorbar(aes(x = id , ymin = intercept_low95 , ymax = intercept_high95), 
                lwd = 0.1, width = 0, alpha=0.5) + 
  geom_point(alpha=0.7, col = "black") + 
  scale_y_continuous(breaks = seq(-7,7,1)) +
  xlab("Individuals\n") + 
  ylab("\nIntercepts") +
  theme_bw() + 
  theme( panel.grid.major = element_blank() , 
         panel.grid.minor = element_blank() , 
         axis.text.x = element_text(colour = "black", size=15),
         axis.title.x = element_text(colour = "black", size=20),
         axis.text.y = element_text(colour = "black", size=15),
         axis.title.y = element_text(colour = "black", size=20)
         ) + 
  coord_flip()

ggsave("Fig4a.png", last_plot() , width=7, height=7)

ggplot( random_effects , aes( x = id , y = lin_slope_mu )) + 
  geom_errorbar(aes(x = id , ymin = lin_slope_low95 , ymax = lin_slope_high95), 
                lwd = 0.1, alpha=0.5,width = 0) + 
  geom_point(alpha=0.7, col = "black") + 
  xlab("Individuals\n") + 
  ylab("\nLinear slopes") +
  theme_bw() + 
  theme( panel.grid.major = element_blank() , 
         panel.grid.minor = element_blank() , 
         axis.text.x = element_text(colour = "black", size=15),
         axis.title.x = element_text(colour = "black", size=20),
         axis.text.y = element_text(colour = "black", size=15),
         axis.title.y = element_text(colour = "black", size=20)
  ) + 
  coord_flip()

ggsave("Fig4b.png", last_plot() , width=7, height=7)

ggplot( random_effects , aes( x = id , y = quad_slope_mu )) + 
  geom_errorbar(aes(x = id , ymin = quad_slope_low95 , ymax = quad_slope_high95), 
                lwd = 0.1, alpha=0.5, width = 0) + 
  geom_point(alpha=0.7, col = "black") + 
  xlab("Individuals\n") + 
  ylab("\nQuadratic slopes") +
  theme_bw() + 
  theme( panel.grid.major = element_blank() , 
         panel.grid.minor = element_blank() , 
         axis.text.x = element_text(colour = "black", size=15),
         axis.title.x = element_text(colour = "black", size=20),
         axis.text.y = element_text(colour = "black", size=15),
         axis.title.y = element_text(colour = "black", size=20)
  ) +
  coord_flip()

ggsave("Fig4c.png", last_plot() , width=7, height=7)

ggplot( random_effects , aes( x = id , y = scale_mu )) + 
  geom_errorbar(aes(x = id , ymin = scale_low95 , ymax = scale_high95), 
                lwd = 0.1,alpha=0.5, width = 0) + 
  scale_y_continuous(limits=c(0,18)) +
  geom_point(alpha=0.7, col = "black") + 
  xlab("Individuals\n") + 
  ylab("\nResidual SDs") +
  theme_bw() + 
  theme( panel.grid.major = element_blank() , 
         panel.grid.minor = element_blank() , 
         axis.text.x = element_text(colour = "black", size=15),
         axis.title.x = element_text(colour = "black", size=20),
         axis.text.y = element_text(colour = "black", size=15),
         axis.title.y = element_text(colour = "black", size=20)
  ) + 
  coord_flip()

ggsave("Fig4d.png", last_plot() , width=7, height=7)

#=================================================================================
# calculate ICC by day
#=================================================================================

# first, get relevant SDs and covariances
int_sd = post_samples[,"sigmaID[1]"] 
cov_int_lin_slope = post_samples[,"sigmaID[1]"]*post_samples[,"Rho[1,2]"]*post_samples[,"sigmaID[2]"] 
lin_slope_sd = post_samples[,"sigmaID[2]"] 
quad_slope_sd = post_samples[,"sigmaID[3]"] 
cov_int_quad_slope = post_samples[,"sigmaID[1]"]*post_samples[,"Rho[1,3]"]*post_samples[,"sigmaID[2]"]  
cov_lin_quad_slopes = post_samples[,"sigmaID[2]"]*post_samples[,"Rho[2,3]"]*post_samples[,"sigmaID[3]"]   
res_sd = exp(post_samples[,"delta"]) 
cov_res_int_slope = res_sd*post_samples[,"Rho[4,1]"]*post_samples[,"sigmaID[1]"]
cov_res_lin_slope = res_sd*post_samples[,"Rho[4,2]"]*post_samples[,"sigmaID[2]"]
cov_res_quad_slope = res_sd*post_samples[,"Rho[4,3]"]*post_samples[,"sigmaID[3]"]

day_values <- c(-1,0,1)

ICC_by_day <- sapply(day_values, 
                     function(x) { 
                       x <- x
                       x2 <- x^2
                       numerator <- int_sd^2 + (2*cov_int_lin_slope*x) + (lin_slope_sd^2)*(x^2) +
                         (2*cov_int_quad_slope*(x2)) + (quad_slope_sd^2)*(x2^2) 
                       denominator <- numerator + res_sd^2 
                       numerator/denominator
                     }
)

# create matrix
ICC_df <- data.frame(day_values = c(-1,0,1), 
                     ICC_mu = apply(ICC_by_day, 2 , mean ) , 
                     hdi_low = apply(ICC_by_day , 2 , function(z) HPDI(z,0.95) )[1,] ,
                     hdi_high = apply(ICC_by_day , 2 , function(z) HPDI(z,0.95) )[2,] )


#=================================================================================
# cross-environmental correlations
#=================================================================================

K <- rep(list(list()), chain_length )
P <- rep(list(list()), chain_length)
Corr_P <- rep(list(list()) , chain_length)

for( i in 1:chain_length ) { 
K[[i]] = matrix(c(int_sd[i]^2, cov_int_lin_slope[i], cov_int_quad_slope[i],
              cov_int_lin_slope[i], lin_slope_sd[i]^2, cov_lin_quad_slopes[i], 
              cov_int_quad_slope[i], cov_lin_quad_slopes[i], quad_slope_sd[i]^2 
            ), 
           ncol = 3 )

day_vals <- c( -1 , 0 , 1 )

Phi = matrix(c( rep(1,3), day_vals, day_vals^2), 
             ncol = 3, nrow = 3 )

P[[i]] = Phi%*%K[[i]]%*%t(Phi) 

Corr_P[[i]] <- cov2cor(P[[i]])
Corr_P
}

# find means and HDIs
mean(unlist(lapply(Corr_P, `[[`, 7)))
HPDI(unlist(lapply(Corr_P, `[[`, 7)), 0.95)

#=================================================================================
# plot predictions of dogs with 5 highest and lowest predictabilities (Figure 5)
#=================================================================================

order_scales = sort(apply(random_res_ints , 2 , mean ) , decreasing = FALSE)

bottom_five_ids <- c(75,2858,3143,2911,644)
top_five_ids <- c(759,1948,2719,79,2622)

pred_df <- cbind(stan_data[stan_data$id %in% c(bottom_five_ids,top_five_ids), ], 
                 individual_curves[individual_curves$id %in% c(bottom_five_ids,top_five_ids), ])
pred_df$IIV <- ifelse(pred_df$id %in% bottom_five_ids, "Low", "High")

ggplot(pred_df , aes(x = day, y = aggCodes)) + 
  geom_point(size=2) + 
  geom_line(aes(x = day, y = pred_mu)) + 
  geom_line(aes(x = day, y=pred_hdi_low ), linetype="dashed") +
  geom_line(aes(x = day, y=pred_hdi_high ), linetype="dashed") + 
  facet_wrap( ~ IIV + id, nrow=2) + 
  coord_cartesian(xlim=c(0,30), ylim=c(-3,7.5)) +
  scale_x_continuous(breaks=seq(0,30,10)) +
  scale_y_continuous(breaks=seq(1,6,1)) +
  ylab("Sociability (ordinal/latent scale)\n") + 
  xlab("\nDay since arrival") +
  theme_bw() + 
  theme( strip.text = element_blank(),
         panel.grid.major.x = element_blank() ,
         panel.grid.minor = element_blank(),
         panel.grid.major.y = element_line( size=.1, color="black" ), 
         axis.text.x = element_text(colour ="black", size=15), 
         axis.text.y = element_text(colour ="black", size=13),
         axis.title = element_text(colour = "black", size=20) )

ggsave("Fig5.png", last_plot() , width = 10, height=5)


#=========================== end =====================================
