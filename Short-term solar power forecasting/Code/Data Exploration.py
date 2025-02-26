import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data_path = r'D:\My pythoin\Kaggle\Short-term solar power forecasting\Data\train (1).csv'
df = pd.read_csv(data_path)

# 检测缺失值
missing_values = df.isnull().sum()
print("缺失值检测：")
print(missing_values)

# 检测异常值（使用箱线图法）
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

columns_to_check = [
    'Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
    'Weather_Relative_Humidity', 'Global_Horizontal_Radiation',
    'Diffuse_Horizontal_Radiation', 'Wind_Direction',
    'Weather_Daily_Rainfall', 'Radiation_Global_Tilted',
    'Radiation_Diffuse_Tilted'
]

print("异常值检测：")
for column in columns_to_check:
    outliers = detect_outliers(df, column)
    print(f"{column} 异常值数量: {len(outliers)}")

# 简单可视化
def visualize_columns(df, columns):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns, 1):
        plt.subplot(5, 2, i)
        sns.boxplot(x=df[column])
        plt.title(f'{column} 箱线图')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns, 1):
        plt.subplot(5, 2, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'{column} 直方图')
    plt.tight_layout()
    plt.show()

    # 相关系数可视化
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix[['Active_Power']].drop('Active_Power').sort_values(by='Active_Power', ascending=False), annot=True, cmap='coolwarm')
    plt.title('各变量与Active_Power的相关系数')
    plt.show()

# 选择要可视化的列
columns_to_visualize = [
    'Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius'
]
visualize_columns(df, columns_to_visualize)

# 删除相关系数小于0.1的列
correlation_matrix = df.corr()
low_correlation_columns = correlation_matrix[correlation_matrix['Active_Power'].abs() < 0.1].index
df.drop(columns=low_correlation_columns, inplace=True)

# 定义绘制三条曲线的函数
def plot_three_curves(df, start=0, end=1000):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Global_Horizontal_Radiation'][start:end], label='Global_Horizontal_Radiation')
    plt.plot(df['Radiation_Global_Tilted'][start:end], label='Radiation_Global_Tilted')
    plt.plot(df['Active_Power'][start:end], label='Active_Power')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Global_Horizontal_Radiation, Radiation_Global_Tilted and Active_Power')
    plt.legend()
    plt.show()

# 调用绘制函数
plot_three_curves(df, start=0, end=1000)

