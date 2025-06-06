# cva_risk_management_thesis
The training happens in trainModel.ipynb under PPO. 
The environment is handeled in dev_env.py under PPO, which generates environments based on simulated markets created by generate_paths_HW in Environment.
These path generations and path handling in turn relies on files in MarketGeneratingFunctions.

Demos contains a number of demo files explaining how the process is generated, and a file that tests unpickling (loading) of saved data files. The first files you should check if you are interested in the models are generathingHullWhitedemo and genereatingJCIRdemo to understand the basics of how we create interest rates and default intensities. After this you can look at coGeneratingdemo to understand how we insert correlation. To understand pricing check pricingdemo first, which prices under Vasicek, and then pricingdemoHW which extends pricing to the HullWhite model. delta_hedge simply helps examine how we perform delta hedging.

Autoencoder is an offshoot folder where we put together an autoencoder to test on. If you want to use an aute-encoder you first train it there hthen move it to PPO where the environment pulls it in to use.