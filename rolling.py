import numpy as np 
import pandas as pd

np.random.seed(0)
test = pd.DataFrame(np.random.randint(100, size=20), columns = ["value"])
test.index = np.arange(100, 120)

lk_fwd = 5
test[f"-{lk_fwd}"] = test.shift(-lk_fwd)

# does not include self
# test["idx.min.1"] = test.shift(-lk_fwd).rolling(lk_fwd)["value"].apply(lambda x: pd.Series(x).idxmin() )

# include self
# test["idx.min.2"] = test.shift(-lk_fwd).rolling(lk_fwd+1)["value"].apply(lambda x: pd.Series(x).idxmin() )

# we are at row 105
# we look forward N rows, for eg, look foward 5 rows => 106 to 110 
# in this 5 rows we set a condition
# we find the index of the first element that match this condition 

to_roll = lk_fwd+1
lk_fwd_rolling = test.shift(-lk_fwd).rolling(to_roll)["value"]
func = lambda y: ( y.iloc[1:] - y.iloc[0] > 2)
test["idx.1st.occ"] = lk_fwd_rolling.apply(lambda x: func(x).idxmax() ) +lk_fwd
test["idx.1st.any"] = lk_fwd_rolling.apply(lambda x: func(x).sum() > 0 ) 

# just to check got "roll" properly
test["window.self"] = lk_fwd_rolling.apply(lambda x: x.iloc[0] ) 
test["window.first"] = lk_fwd_rolling.apply(lambda x: x.iloc[1] ) 
test["window.last"] = lk_fwd_rolling.apply(lambda x: x.iloc[-1] ) 
test
