import pandas as pd
import numpy as np

zero_share = {}

for col in df.columns:
    s = df[col]
    
    # сравниваем и с числовым 0, и со строкой "0"
    share = ((s == 0) | (s == "0")).mean()
    zero_share[col] = share

zero_share_df = (
    pd.Series(zero_share, name="zero_share")
    .sort_values(ascending=False)
    .to_frame()
)

print("Share of zeros (0 or '0') by column:")
print(zero_share_df)
