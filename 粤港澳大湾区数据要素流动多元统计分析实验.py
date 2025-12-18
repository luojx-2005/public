import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("\n1. 数据加载与预览")
df = pd.read_csv('main_data_advanced.csv', encoding='utf-8')
print(f"数据形状: {df.shape}")
print(f"城市数量: {len(df['城市'].unique())}")
print(f"时间范围: {df['年份'].min()}年 - {df['年份'].max()}年")

df_2023 = df[df['年份'] == 2023].copy()
if len(df_2023) == 0:
    df_2023 = df.iloc[-len(df['城市'].unique()):].copy()
key_indicators = ['GDP_亿元', '跨境数据传输总量_TB', '数字经济核心产业增加值_亿元',
                  '研发经费投入_亿元', '数据交易额_亿元', '5G基站数量']
available_indicators = [col for col in key_indicators if col in df_2023.columns]
df_analysis = df_2023[['城市'] + available_indicators].copy()
df_analysis = df_analysis.dropna()
print(f"\n分析指标 ({len(available_indicators)}个):")
for i, indicator in enumerate(available_indicators, 1):
    print(f"  {i}. {indicator}")

print("\n生成图片1：各城市GDP对比图...")
plt.figure(figsize=(12, 8))
df_sorted = df_analysis.sort_values('GDP_亿元')
bars = plt.barh(df_sorted['城市'], df_sorted['GDP_亿元'], color='steelblue', alpha=0.7)
if len(bars) > 0:
    bars[0].set_color('green')
    bars[-1].set_color('red')

plt.xlabel('GDP（亿元）', fontsize=12)
plt.title('粤港澳大湾区各城市GDP对比（2023年）', fontsize=14, pad=15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('图片1_各城市GDP对比.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图片2：各城市跨境数据传输对比图...")
plt.figure(figsize=(12, 8))
df_sorted = df_analysis.sort_values('跨境数据传输总量_TB')
bars = plt.barh(df_sorted['城市'], df_sorted['跨境数据传输总量_TB'],
                color='lightcoral', alpha=0.7)
if len(bars) > 0:
    bars[0].set_color('green')
    bars[-1].set_color('red')
plt.xlabel('跨境数据传输总量（TB）', fontsize=12)
plt.title('粤港澳大湾区各城市跨境数据传输对比（2023年）', fontsize=14, pad=15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('图片2_各城市跨境数据传输对比.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n生成图片3：相关系数矩阵热图...")
corr_matrix = df_analysis[available_indicators].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            fmt='.3f', annot_kws={"size": 10})
plt.title('关键指标相关系数矩阵热图', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('图片3_相关系数矩阵热图.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n关键相关性分析:")
print(f"1. GDP与数字经济核心产业增加值的相关性: {corr_matrix.loc['GDP_亿元', '数字经济核心产业增加值_亿元']:.3f}")
print(f"2. 跨境数据传输与研发投入的相关性: {corr_matrix.loc['跨境数据传输总量_TB', '研发经费投入_亿元']:.3f}")
print(f"3. 数据交易额与5G基站数量的相关性: {corr_matrix.loc['数据交易额_亿元', '5G基站数量']:.3f}")

print("\n4. 主成分分析（PCA）")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_analysis[available_indicators])
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
n_components = np.where(cumulative_variance >= 0.80)[0][0] + 1
print(f"提取 {n_components} 个主成分，累计解释方差 {cumulative_variance[n_components - 1]:.1%}")
print("\n主成分分析结果:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i + 1}: 方差解释率 = {var:.2%}, 累计解释率 = {cum_var:.2%}")

print("\n生成图片4：PCA碎石图...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-',
         linewidth=2, markersize=8, label='方差贡献率')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'rs--',
         linewidth=2, markersize=8, label='累计贡献率')
plt.axhline(y=0.80, color='g', linestyle='--', alpha=0.7, label='80%阈值')
plt.axvline(x=n_components, color='orange', linestyle='--', alpha=0.7,
            label=f'主成分数={n_components}')
plt.xlabel('主成分数量', fontsize=12)
plt.ylabel('方差解释率', fontsize=12)
plt.title('PCA碎石图（主成分数量选择）', fontsize=14, pad=15)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('图片4_PCA碎石图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图片5：PCA散点图...")
plt.figure(figsize=(12, 9))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=150, alpha=0.7, edgecolors='k', color='steelblue')

for i, city in enumerate(df_analysis['城市']):
    plt.annotate(city, (X_pca[i, 0], X_pca[i, 1]),
                 fontsize=10, alpha=0.8,
                 xytext=(5, 5), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.xlabel(f'第一主成分 PC1（解释方差：{explained_variance[0]:.1%}）', fontsize=12)
plt.ylabel(f'第二主成分 PC2（解释方差：{explained_variance[1]:.1%}）', fontsize=12)
plt.title('城市在前两个主成分上的分布', fontsize=14, pad=15)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('图片5_PCA散点图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5. 聚类分析")
def find_optimal_clusters(X, max_clusters=6):
    silhouette_scores = []
    for n in range(2, min(max_clusters + 1, len(X))):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    optimal_n = np.argmax(silhouette_scores) + 2  # +2因为从2开始
    return optimal_n, silhouette_scores
optimal_n, scores = find_optimal_clusters(X_scaled)
print(f"最佳聚类数: {optimal_n} (轮廓系数: {max(scores):.3f})")
kmeans = KMeans(n_clusters=optimal_n, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_analysis['聚类类别'] = clusters

print(f"\n聚类结果:")
for cluster_id in range(optimal_n):
    cities = df_analysis[df_analysis['聚类类别'] == cluster_id]['城市'].tolist()
    print(f"类别{cluster_id} ({len(cities)}个城市): {', '.join(cities)}")
print("\n生成图片6：聚类散点图...")
plt.figure(figsize=(12, 9))
colors = plt.cm.Set1(np.linspace(0, 1, optimal_n))

for cluster_id in range(optimal_n):
    mask = clusters == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                s=180, alpha=0.8, edgecolors='k', linewidth=1.5,
                color=colors[cluster_id],
                label=f'类别{cluster_id} ({sum(mask)}个城市)')

for i, city in enumerate(df_analysis['城市']):
    plt.annotate(city, (X_pca[i, 0], X_pca[i, 1]),
                 fontsize=9, alpha=0.8,
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel(f'第一主成分 PC1（解释方差：{explained_variance[0]:.1%}）', fontsize=12)
plt.ylabel(f'第二主成分 PC2（解释方差：{explained_variance[1]:.1%}）', fontsize=12)
plt.title(f'K-means聚类结果（聚类数={optimal_n}）', fontsize=14, pad=15)
plt.legend(fontsize=10, title='聚类类别', title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图片6_聚类散点图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图片7：聚类中心条形图...")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(14, 8))
x_pos = np.arange(len(available_indicators))
width = 0.8 / optimal_n

for i in range(optimal_n):
    offset = (i - optimal_n / 2 + 0.5) * width
    plt.bar(x_pos + offset, cluster_centers[i], width,
            color=plt.cm.Set1(i / optimal_n), alpha=0.7,
            label=f'类别{i}')

plt.xlabel('指标', fontsize=12)
plt.ylabel('指标值', fontsize=12)
plt.title('各类别中心指标值对比', fontsize=14, pad=15)
plt.xticks(x_pos, available_indicators, rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('图片7_聚类中心条形图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图片8：雷达图...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='polar')

angles = np.linspace(0, 2 * np.pi, len(available_indicators), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

colors = plt.cm.Set1(np.linspace(0, 1, optimal_n))

for i in range(optimal_n):
    values = cluster_centers[i].tolist()
    values += values[:1]
    values_norm = [(v - cluster_centers[:, j].min()) /
                   (cluster_centers[:, j].max() - cluster_centers[:, j].min() + 1e-8)
                   for j, v in enumerate(values[:-1])]
    values_norm += values_norm[:1]

    ax.plot(angles, values_norm, 'o-', linewidth=2,
            color=colors[i], label=f'类别{i}', markersize=6)
    ax.fill(angles, values_norm, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(available_indicators, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax.set_title('各类别特征雷达图（指标归一化）', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
plt.tight_layout()
plt.savefig('图片8_雷达图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n6. 综合评价")
# 计算综合得分
pca_scores = X_pca[:, :n_components]
weights = explained_variance[:n_components] / explained_variance[:n_components].sum()
comprehensive_score = np.dot(pca_scores, weights)
# 创建评价表
evaluation_df = pd.DataFrame({
    '城市': df_analysis['城市'],
    '综合得分': comprehensive_score,
    '聚类类别': clusters
})
evaluation_df = evaluation_df.sort_values('综合得分', ascending=False)
evaluation_df['排名'] = range(1, len(evaluation_df) + 1)
print("\n城市综合排名:")
print("-" * 50)
print(evaluation_df[['排名', '城市', '综合得分', '聚类类别']].to_string(index=False))

print("\n生成图片9：综合排名图...")
plt.figure(figsize=(14, 8))
colors = plt.cm.Set1(np.linspace(0, 1, optimal_n))
evaluation_sorted = evaluation_df.sort_values('排名', ascending=True)

for cluster_id in range(optimal_n):
    cluster_data = evaluation_sorted[evaluation_sorted['聚类类别'] == cluster_id]

    plt.barh(range(len(cluster_data)), cluster_data['综合得分'],
             color=colors[cluster_id], alpha=0.7,
             label=f'类别{cluster_id}')
y_offset = 0
for cluster_id in range(optimal_n):
    cluster_data = evaluation_sorted[evaluation_sorted['聚类类别'] == cluster_id]
    for i, (_, row) in enumerate(cluster_data.iterrows()):
        plt.text(row['综合得分'] + 0.02, y_offset + i,
                 f"{row['城市']} ({row['综合得分']:.3f})",
                 va='center', fontsize=9)
    y_offset += len(cluster_data)
plt.xlabel('综合得分', fontsize=12)
plt.ylabel('城市', fontsize=12)
plt.yticks([])  # 隐藏y轴刻度
plt.title('城市综合评价排名', fontsize=14, pad=15)
plt.legend(title='聚类类别', fontsize=10, title_fontsize=11)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('图片9_综合排名图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图片10：得分与排名散点图...")
plt.figure(figsize=(12, 8))
colors = plt.cm.Set1(np.linspace(0, 1, optimal_n))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

for cluster_id in range(optimal_n):
    cluster_data = evaluation_df[evaluation_df['聚类类别'] == cluster_id]
    marker = markers[cluster_id % len(markers)]

    plt.scatter(cluster_data['综合得分'], cluster_data['排名'],
                s=200, alpha=0.8, edgecolors='k', linewidth=1.5,
                color=colors[cluster_id], marker=marker,
                label=f'类别{cluster_id}')
for _, row in evaluation_df.iterrows():
    plt.annotate(row['城市'], (row['综合得分'], row['排名']),
                 fontsize=9, alpha=0.7,
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('综合得分', fontsize=12)
plt.ylabel('排名（数字越小越好）', fontsize=12)
plt.title('城市综合得分与排名关系', fontsize=14, pad=15)
plt.gca().invert_yaxis()  # 排名越小越靠上
plt.legend(title='聚类类别', fontsize=10, title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图片10_得分与排名散点图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n7. 时间趋势分析")
if '年份' in df.columns and len(df['年份'].unique()) > 1:
    years = sorted(df['年份'].unique())
    print(f"分析年份: {years}")
    trend_indicators = ['GDP_亿元', '跨境数据传输总量_TB', '数字经济核心产业增加值_亿元']
    trend_indicators = [ind for ind in trend_indicators if ind in df.columns]

    if 'GDP_亿元' in trend_indicators:
        print("\n生成图片11：GDP时间趋势图...")
        plt.figure(figsize=(12, 8))

        for city in df['城市'].unique():
            city_data = df[df['城市'] == city].sort_values('年份')
            if 'GDP_亿元' in city_data.columns:
                plt.plot(city_data['年份'], city_data['GDP_亿元'],
                         marker='o', linewidth=2, markersize=6, label=city, alpha=0.7)

        plt.xlabel('年份', fontsize=12)
        plt.ylabel('GDP（亿元）', fontsize=12)
        plt.title('粤港澳大湾区各城市GDP时间趋势', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig('图片11_GDP时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

    if '跨境数据传输总量_TB' in trend_indicators:
        print("\n生成图片12：跨境数据传输时间趋势图...")
        plt.figure(figsize=(12, 8))

        for city in df['城市'].unique():
            city_data = df[df['城市'] == city].sort_values('年份')
            if '跨境数据传输总量_TB' in city_data.columns:
                plt.plot(city_data['年份'], city_data['跨境数据传输总量_TB'],
                         marker='s', linewidth=2, markersize=6, label=city, alpha=0.7)

        plt.xlabel('年份', fontsize=12)
        plt.ylabel('跨境数据传输总量（TB）', fontsize=12)
        plt.title('粤港澳大湾区各城市跨境数据传输时间趋势', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig('图片12_跨境数据传输时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

    if '数字经济核心产业增加值_亿元' in trend_indicators:
        print("\n生成图片13：数字经济时间趋势图...")
        plt.figure(figsize=(12, 8))

        for city in df['城市'].unique():
            city_data = df[df['城市'] == city].sort_values('年份')
            if '数字经济核心产业增加值_亿元' in city_data.columns:
                plt.plot(city_data['年份'], city_data['数字经济核心产业增加值_亿元'],
                         marker='^', linewidth=2, markersize=6, label=city, alpha=0.7)

        plt.xlabel('年份', fontsize=12)
        plt.ylabel('数字经济核心产业增加值（亿元）', fontsize=12)
        plt.title('粤港澳大湾区各城市数字经济时间趋势', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig('图片13_数字经济时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

print("\n8. 生成分析报告")
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>粤港澳大湾区数据要素流动分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #3498db; color: white; }}
        .image {{ text-align: center; margin: 30px 0; }}
        img {{ max-width: 90%; height: auto; border: 1px solid #ddd; }}
        .ranking {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>粤港澳大湾区数据要素流动分析报告</h1>

    <div class="summary">
        <h2>📊 分析概要</h2>
        <p><strong>分析时间：</strong>2024年</p>
        <p><strong>分析城市：</strong>{len(df_analysis)}个城市</p>
        <p><strong>分析指标：</strong>{len(available_indicators)}个关键指标</p>
        <p><strong>主要方法：</strong>描述统计、相关性分析、主成分分析、聚类分析</p>
    </div>

    <h2>🏆 城市综合排名（前5名）</h2>
    <div class="ranking">
        <table>
            <tr><th>排名</th><th>城市</th><th>综合得分</th><th>聚类类别</th></tr>
"""

# 添加前5名排名
for i in range(min(5, len(evaluation_df))):
    row = evaluation_df.iloc[i]
    html_report += f"""
            <tr>
                <td>{int(row['排名'])}</td>
                <td>{row['城市']}</td>
                <td>{row['综合得分']:.3f}</td>
                <td>类别{int(row['聚类类别'])}</td>
            </tr>
"""

html_report += f"""
        </table>
    </div>

    <h2>🔍 聚类分析结果</h2>
    <p>通过K-means聚类将城市分为 <strong>{optimal_n}</strong> 个类别：</p>
    <ul>
"""

for cluster_id in range(optimal_n):
    cities = df_analysis[df_analysis['聚类类别'] == cluster_id]['城市'].tolist()
    html_report += f"""
        <li><strong>类别{cluster_id}</strong>：{', '.join(cities)}</li>
"""

html_report += """
    </ul>

    <h2>📈 分析图表展示</h2>

    <div class="image">
        <p><strong>图片1：各城市GDP对比</strong></p>
        <img src="图片1_各城市GDP对比.png">
    </div>

    <div class="image">
        <p><strong>图片2：各城市跨境数据传输对比</strong></p>
        <img src="图片2_各城市跨境数据传输对比.png">
    </div>

    <div class="image">
        <p><strong>图片3：相关系数矩阵热图</strong></p>
        <img src="图片3_相关系数矩阵热图.png">
    </div>

    <div class="image">
        <p><strong>图片4：PCA碎石图</strong></p>
        <img src="图片4_PCA碎石图.png">
    </div>

    <div class="image">
        <p><strong>图片5：PCA散点图</strong></p>
        <img src="图片5_PCA散点图.png">
    </div>

    <div class="image">
        <p><strong>图片6：聚类散点图</strong></p>
        <img src="图片6_聚类散点图.png">
    </div>

    <div class="image">
        <p><strong>图片7：聚类中心条形图</strong></p>
        <img src="图片7_聚类中心条形图.png">
    </div>

    <div class="image">
        <p><strong>图片8：雷达图</strong></p>
        <img src="图片8_雷达图.png">
    </div>

    <div class="image">
        <p><strong>图片9：综合排名图</strong></p>
        <img src="图片9_综合排名图.png">
    </div>

    <div class="image">
        <p><strong>图片10：得分与排名散点图</strong></p>
        <img src="图片10_得分与排名散点图.png">
    </div>
"""
if '年份' in df.columns and len(df['年份'].unique()) > 1:
    trend_indicators = ['GDP_亿元', '跨境数据传输总量_TB', '数字经济核心产业增加值_亿元']
    trend_indicators = [ind for ind in trend_indicators if ind in df.columns]

    if 'GDP_亿元' in trend_indicators:
        html_report += """
    <div class="image">
        <p><strong>图片11：GDP时间趋势图</strong></p>
        <img src="图片11_GDP时间趋势.png">
    </div>
"""

    if '跨境数据传输总量_TB' in trend_indicators:
        html_report += """
    <div class="image">
        <p><strong>图片12：跨境数据传输时间趋势图</strong></p>
        <img src="图片12_跨境数据传输时间趋势.png">
    </div>
"""

    if '数字经济核心产业增加值_亿元' in trend_indicators:
        html_report += """
    <div class="image">
        <p><strong>图片13：数字经济时间趋势图</strong></p>
        <img src="图片13_数字经济时间趋势.png">
    </div>
"""

html_report += """

    <h2>💡 主要结论</h2>
    <ol>
        <li>粤港澳大湾区城市在数据要素发展上存在显著差异</li>
        <li>经济发展水平与数据要素发展高度相关</li>
        <li>通过聚类分析可将城市分为不同类型，便于制定差异化政策</li>
        <li>需要加强区域协同，推动数据要素自由流动</li>
    </ol>

    <hr>
    <p style="text-align: center; color: #666; font-size: 0.9em;">
        报告生成时间：2024年 | 数据分析报告
    </p>
</body>
</html>
"""

with open('分析报告.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("\n✅ 所有图片已生成！")
print("📊 生成图片清单:")
print("   1. 图片1_各城市GDP对比.png")
print("   2. 图片2_各城市跨境数据传输对比.png")
print("   3. 图片3_相关系数矩阵热图.png")
print("   4. 图片4_PCA碎石图.png")
print("   5. 图片5_PCA散点图.png")
print("   6. 图片6_聚类散点图.png")
print("   7. 图片7_聚类中心条形图.png")
print("   8. 图片8_雷达图.png")
print("   9. 图片9_综合排名图.png")
print("   10. 图片10_得分与排名散点图.png")

if '年份' in df.columns and len(df['年份'].unique()) > 1:
    print("   11. 图片11_GDP时间趋势.png")
    print("   12. 图片12_跨境数据传输时间趋势.png")
    print("   13. 图片13_数字经济时间趋势.png")

print("\n📄 生成报告: 分析报告.html")
print("\n🎉 分析完成！所有图片都已单独保存，可直接用于论文。")