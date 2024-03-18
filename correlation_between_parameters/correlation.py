import pandas as pd
import matplotlib.pyplot as plt

# читаем файл CSV и создаем датафрейм
df = pd.read_csv('results.csv')
# print(df.columns)
# выбираем два столбца для построения графика
df = df[[ '      train/box_loss',  '   metrics/precision','      metrics/recall', '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',]]

# строим график
df.plot()

# отображаем график
plt.show()