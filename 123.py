rows_excel = [12, 20, 35]
rows_df = [r - 11 for r in rows_excel]

result = (
    df.loc[rows_df, ['Наименование показателя', 'Код'] + df.columns[df.columns.str.match(month_re, na=False)].tolist()]
      .melt(
          id_vars=['Наименование показателя', 'Код'],
          var_name='Период',
          value_name='Значение'
      )
      .assign(
          Месяц=lambda x: x['Период'].str.extract(r'(Январь|Февраль|Март|Апрель|Май|Июнь|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь)')[0],
          Год=lambda x: x['Период'].str.extract(r'(\d{4})')[0].astype(int),
          Дата=lambda x: pd.to_datetime({
              'year': x['Год'],
              'month': x['Месяц'].map(month_map),
              'day': 1
          }) + pd.offsets.MonthEnd(0)
      )
      .pivot_table(
          index='Дата',
          columns='Код',
          values='Значение',
          aggfunc='first'
      )
      .reset_index()
)

result.columns.name = None
