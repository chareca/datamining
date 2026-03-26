
print(df.head())
print(df.describe())

for c in df.columns:
    print(c)
    print(np.unique(df.loc[:,c]))