import pandas as pd

codes = ['1.2', '1.6.6', '1.6.8.1', '1.6.8.2', '2', '1.7', '1.8.1', '2', '6', '10.1.2']

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

df = df.rename(columns={
    df.columns[0]: 'Наименование показателя',
    df.columns[1]: 'Код'
})

df['Код'] = df['Код'].astype(str).str.strip()

month_re = r'^(Январь|Февраль|Март|Апрель|Май|Июнь|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь)\s+\d{4}$'

result = (
    df.loc[df['Код'].isin(codes), ['Наименование показателя', 'Код'] + df.columns[df.columns.str.match(month_re, na=False)].tolist()]
      .melt(
          id_vars=['Наименование показателя', 'Код'],
          var_name='Период',
          value_name='Значение'
      )
      .assign(
          Месяц=lambda x: x['Период'].str.extract(month_re.replace('^', '').replace('$', ''))[0],
          Год=lambda x: x['Период'].str.extract(r'(\d{4})')[0].astype(int),
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
