from sklearn.linear_model import LinearRegression


file = ''
weights = []
df = pd.read_excel(file)
window_size = 9
file_out = ''

for _, row in list(df.iterrows())[window_size:]:

    data = df.loc[df['Дата'] < row['Дата']]  # фильтруем данные до текущей даты
    data = data.iloc[-window_size:]  # берем последние 9 месяцев
    X = data[['y_ML', 'y_BaseModel']]
    y = data['y_actual']
    
    model = LinearRegression()
    model.fit(X, y)

    predict = model.predict(row[['y_ML', 'y_BaseModel']].to_frame().T)

    df.loc[row.name, 'y_ensemble'] = predict[0]
    weights.append((model.coef_[0], model.coef_[1], model.intercept_))

weights_df = pd.DataFrame(weights, columns=['w_ML', 'w_BaseModel', 'intercept'])

with pd.ExcelWriter(file_out, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="summary_with_ensemble", index=False)
    weights_df.to_excel(writer, sheet_name="ensemble_weights", index=False)
    
