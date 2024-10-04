# Quick and dirty place to put notes regarding testing.
**10/1/2024**
Regression_RegularizationChatGPT
In Arenas, CreditScoreRegressionNeedsBias and CreditScoreRegression it performs comparable, but when switching to SalaryExperienceRegressionNeedsBias it shits the bed
**10/3/2024**
For CreditScoreRegression
BlackBird (no bias) does best
Hayabusa_MSE_ close if bias starts at 0, if bias is 5 it's bad.
Regression_Bias_ChatGPT2 Good 

For CreditScoreRegressionNeedsBias
BB is bad
Busa good if bias init=.5 bad if init =0
Regression_Bias_ChatGPT2 good either way
