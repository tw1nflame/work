import pandas as pd

codes = [101, 102, 103]

month_map = {
    'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4,
    'Май': 5, 'Июнь': 6, 'Июль': 7, 'Август': 8,
    'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12
}

df = pd.read_excel(
    'прогноз для февраля.xlsx',
    sheet_name='ПАО ГМК',
    skiprows=9,
    header=[0, 1]
)

df.columns = [
    b if 'Unnamed' in str(a) else f'{b} {a}'
    for a, b in df.columns
]

result = (
    df.loc[:, ~df.columns.str.contains('НИТ|квартал', case=False, na=False)]
      .loc[lambda x: x['Код'].isin(codes)]
      .melt(
          id_vars=['Наименование показателя', 'Код'],
          var_name='Период',
          value_name='Значение'
      )
      .assign(
          Год=lambda x: x['Период'].str.extract(r'(\d{4})').astype(int),
          Месяц=lambda x: x['Период'].str.extract(r'(Январь|Февраль|Март|Апрель|Май|Июнь|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь)')[0],
          Дата=lambda x: pd.to_datetime({
              'year': x['Год'],
              'month': x['Месяц'].map(month_map),
              'day': 1
          }) + pd.offsets.MonthEnd(0)
      )
      .pivot_table(
          index='Дата',
          columns='Наименование показателя',
          values='Значение',
          aggfunc='first'
      )
      .reset_index()
)

result.columns.name = None
