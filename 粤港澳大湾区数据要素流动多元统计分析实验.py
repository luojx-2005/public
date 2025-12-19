import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载
df = pd.read_csv('main_data_advanced.csv', encoding='utf-8-sig')
df_2023 = df[df['年份'] == 2023].copy()

print("数据加载完成！")
print(f"数据形状: {df.shape}")
print(f"2023年数据: {df_2023.shape[0]}个城市")

print("\n生成图1: 跨境数据传输总量趋势")

plt.figure(figsize=(12, 8))
yearly_avg = df.groupby('年份')['跨境数据传输总量_TB'].mean()
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, markersize=10, color='royalblue')
plt.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.2, color='royalblue')

plt.title('粤港澳大湾区跨境数据传输总量年均变化趋势 (2019-2023)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('年份', fontsize=14)
plt.ylabel('跨境数据传输总量 (TB)', fontsize=14)
plt.grid(True, alpha=0.3)

# 添加增长率标注
for i, (x, y) in enumerate(zip(yearly_avg.index, yearly_avg.values)):
    if i > 0:
        growth = (y - yearly_avg.values[i - 1]) / yearly_avg.values[i - 1] * 100
        plt.annotate(f'+{growth:.1f}%', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('图1_跨境数据传输趋势.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n生成图2: 各城市研发投入对比")

plt.figure(figsize=(14, 8))
sorted_rd = df_2023.sort_values('研发经费投入_亿元', ascending=True)

bars = plt.barh(sorted_rd['城市'], sorted_rd['研发经费投入_亿元'],
                color=plt.cm.viridis(np.linspace(0, 1, len(sorted_rd))))

plt.title('2023年粤港澳大湾区各城市研发经费投入对比', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('研发经费投入 (亿元)', fontsize=14)
plt.ylabel('城市', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height() / 2,
             f'{width:.1f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('图2_各城市研发投入对比.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n1. 生成图3: 相关性热力图")
plt.figure(figsize=(12, 10))

key_vars = ['跨境数据传输总量_TB', '数据中心机架数', 'GDP_亿元',
            '研发经费投入_亿元', '5G基站数量', '数字经济核心产业增加值_亿元']

corr_matrix = df[key_vars].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8, "label": "相关系数"})

plt.title('关键变量相关性矩阵热力图', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('图3_相关性热力图.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存相关性矩阵到CSV
corr_matrix.to_csv('相关性矩阵.csv', encoding='utf-8-sig')
print("相关性矩阵已保存至: 相关性矩阵.csv")

print("\n2. 偏相关分析（控制变量：GDP_亿元）")

import pingouin as pg

# 选择变量进行偏相关分析
partial_vars = [
    '跨境数据传输总量_TB', '研发经费投入_亿元', '数据中心机架数',
    '5G基站数量', '数字经济核心产业增加值_亿元', 'GDP_亿元'
]
df_partial = df_2023[partial_vars].dropna()

print("偏相关分析结果（控制变量：GDP_亿元）")
print("=" * 80)

partial_results = []
for var in ['研发经费投入_亿元', '数据中心机架数', '5G基站数量', '数字经济核心产业增加值_亿元']:
    try:
        # 计算偏相关
        pc = pg.partial_corr(data=df_partial,
                             x='跨境数据传输总量_TB',
                             y=var,
                             covar='GDP_亿元',
                             method='pearson')

        # 计算简单相关
        simple_r = df_partial['跨境数据传输总量_TB'].corr(df_partial[var])

        # 保存结果
        result = {
            '变量': var,
            '简单相关系数': simple_r,
            '偏相关系数': pc['r'].values[0],
            'p值': pc['p-val'].values[0],
            '样本量': pc['n'].values[0],
            '变化': pc['r'].values[0] - simple_r
        }
        partial_results.append(result)

        # 打印结果
        print(f"目标变量：跨境数据传输总量_TB 与 {var}")
        print(f"  简单相关系数: {simple_r:.3f}")
        print(f"  偏相关系数:   {pc['r'].values[0]:.3f}")
        print(f"  p值:         {pc['p-val'].values[0]:.4f}")
        print(f"  样本量:       {pc['n'].values[0]}")
        print(f"  变化差异:     {pc['r'].values[0] - simple_r:+.3f}")
        print("-" * 60)

    except Exception as e:
        print(f"计算{var}偏相关时出错: {e}")

try:
    pc_extra = pg.partial_corr(data=df_partial,
                               x='研发经费投入_亿元',
                               y='数字经济核心产业增加值_亿元',
                               covar='GDP_亿元',
                               method='pearson')

    result = {
        '变量': '研发投入~数字经济增加值',
        '简单相关系数': df_partial['研发经费投入_亿元'].corr(df_partial['数字经济核心产业增加值_亿元']),
        '偏相关系数': pc_extra['r'].values[0],
        'p值': pc_extra['p-val'].values[0],
        '样本量': pc_extra['n'].values[0],
        '变化': pc_extra['r'].values[0] - df_partial['研发经费投入_亿元'].corr(
            df_partial['数字经济核心产业增加值_亿元'])
    }
    partial_results.append(result)

    print(f"\n额外分析：研发投入 ~ 数字经济增加值（控制GDP）")
    print(f"  简单相关系数: {result['简单相关系数']:.3f}")
    print(f"  偏相关系数:   {result['偏相关系数']:.3f}")
    print(f"  p值:         {result['p值']:.4f}")

except Exception as e:
    print(f"计算额外偏相关时出错: {e}")

# 保存偏相关结果
partial_df = pd.DataFrame(partial_results)
partial_df.to_csv('偏相关分析结果.csv', index=False, encoding='utf-8-sig')
print("\n偏相关分析结果已保存至: 偏相关分析结果.csv")

print("\n1. KMO与Bartlett球形检验")
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# 选择因子分析变量（与主成分分析一致）
factor_vars = ['跨境数据传输总量_TB', '数据中心机架数', 'GDP_亿元',
               '数字经济核心产业增加值_亿元', '研发经费投入_亿元', '5G基站数量']

factor_data = df_2023[factor_vars].dropna()

# 计算KMO值
kmo_all, kmo_model = calculate_kmo(factor_data)
print(f"KMO检验值: {kmo_model:.3f}")

# 判断KMO值
if kmo_model >= 0.9:
    kmo_judge = "非常适合"
elif kmo_model >= 0.8:
    kmo_judge = "很适合"
elif kmo_model >= 0.7:
    kmo_judge = "适合"
elif kmo_model >= 0.6:
    kmo_judge = "勉强适合"
elif kmo_model >= 0.5:
    kmo_judge = "不太适合"
else:
    kmo_judge = "完全不适合"

print(f"KMO判断: {kmo_judge} (≥0.7为适合因子分析)")

# Bartlett球形检验
chi_square_value, p_value = calculate_bartlett_sphericity(factor_data)
print(f"\nBartlett球形检验:")
print(f"  近似卡方值: {chi_square_value:.2f}")
print(f"  自由度: {len(factor_vars) * (len(factor_vars) - 1) // 2:.0f}")
print(f"  显著性p值: {p_value:.6f}")
if p_value < 0.001:
    print(f"  检验结论: 极其显著 (p<0.001)，强烈拒绝变量独立假设")
elif p_value < 0.05:
    print(f"  检验结论: 显著 (p<0.05)，拒绝变量独立假设")
else:
    print(f"  检验结论: 不显著，不能拒绝变量独立假设")

# 保存检验结果
kmo_bartlett_result = pd.DataFrame({
    '检验指标': ['KMO值', 'Bartlett卡方值', '自由度', 'p值'],
    '数值': [kmo_model, chi_square_value, len(factor_vars) * (len(factor_vars) - 1) // 2, p_value],
    '判断标准': ['≥0.7为适合', 'p<0.05为显著', '', 'p<0.05拒绝原假设'],
    '结论': [kmo_judge, '极其显著' if p_value < 0.001 else ('显著' if p_value < 0.05 else '不显著'), '',
             '数据存在相关性' if p_value < 0.05 else '数据独立']
})
kmo_bartlett_result.to_csv('KMO_Bartlett检验结果.csv', index=False, encoding='utf-8-sig')
print("\nKMO与Bartlett检验结果已保存至: KMO_Bartlett检验结果.csv")

print("\n2. 生成图4: PCA方差解释率图")
plt.figure(figsize=(10, 8))

pca_vars = factor_vars  # 使用相同的变量

pca_data = df_2023[pca_vars].copy()
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

pca = PCA()
pca.fit(pca_data_scaled)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

components = range(1, len(explained_var) + 1)

# 创建双Y轴
fig, ax1 = plt.subplots(figsize=(10, 8))

# 条形图：单个主成分解释率
bars = ax1.bar(components, explained_var, alpha=0.6, color='skyblue', label='单个成分解释率')
ax1.set_xlabel('主成分', fontsize=14)
ax1.set_ylabel('单个成分解释率', fontsize=14, color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_xticks(components)

# 折线图：累计解释率
ax2 = ax1.twinx()
line = ax2.plot(components, cumulative_var, 'r-', marker='o', linewidth=3,
                markersize=8, label='累计解释率')
ax2.set_ylabel('累计解释率', fontsize=14, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim([0, 1.1])

# 添加阈值线
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, linewidth=2, label='80%阈值')

plt.title('PCA方差解释率分析（碎石图）', fontsize=16, fontweight='bold', pad=20)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('图4_PCA方差解释率.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出主成分详细信息
print("\n主成分详细信息:")
for i, (ev, evr, cv) in enumerate(zip(pca.explained_variance_, explained_var, cumulative_var), 1):
    print(f"  主成分{i}: 特征值={ev:.3f}, 解释方差={evr * 100:.1f}%, 累计解释方差={cv * 100:.1f}%")

# 确定提取的主成分数量（特征值>1且累计贡献率>80%）
n_components = sum(pca.explained_variance_ > 1)
print(f"\n根据特征值>1原则，提取主成分数量: {n_components}")
print(f"前{n_components}个主成分累计解释方差: {cumulative_var[n_components - 1] * 100:.1f}%")

print("\n【精确PCA输出】主成分特征值与贡献率表")
print("=" * 50)

# 获取精确的特征值（解释方差）
explained_variance = pca.explained_variance_
# 获取精确的方差贡献率
explained_variance_ratio = pca.explained_variance_ratio_
# 计算累计贡献率
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 创建并显示精确的表格
pca_details_df = pd.DataFrame({
    '主成分': [f'PC{i + 1}' for i in range(len(explained_variance))],
    '特征值': explained_variance,
    '方差贡献率(%)': explained_variance_ratio * 100,
    '累计贡献率(%)': cumulative_variance_ratio * 100
})

print(pca_details_df.round(3).to_string(index=False))
pca_details_df.to_csv('PCA_特征值与贡献率_精确表.csv', index=False, encoding='utf-8-sig')


n_components_to_show = 3
print(f"\n前{n_components_to_show}个主成分的详细情况:")
for i in range(min(n_components_to_show, len(explained_variance))):
    print(f"  PC{i + 1}: 特征值={explained_variance[i]:.3f}, ",
          f"贡献率={explained_variance_ratio[i] * 100:.1f}%, ",
          f"累计={cumulative_variance_ratio[i] * 100:.1f}%")

print("\n3. 主成分载荷矩阵")

# 获取主成分载荷矩阵（特征向量乘以特征值平方根）
pca_components = pca.components_.T * np.sqrt(pca.explained_variance_)

# 创建载荷表
loadings_df = pd.DataFrame(
    pca_components[:, :n_components],
    index=pca_vars,
    columns=[f'PC{i + 1}' for i in range(n_components)]
)

# 简化的变量名用于显示
var_names_simple = [v.split('_')[0] if '_' in v else v for v in pca_vars]
loadings_display = pd.DataFrame(
    pca_components[:, :n_components],
    index=var_names_simple,
    columns=[f'PC{i + 1}' for i in range(n_components)]
)

print("\n主成分载荷矩阵（前3个主成分）:")
print(loadings_display.round(3))

# 保存到CSV
loadings_df.to_csv('主成分载荷矩阵.csv', encoding='utf-8-sig')
print("\n主成分载荷矩阵已保存至: 主成分载荷矩阵.csv")

# 对主成分进行命名解释（基于载荷绝对值>0.7）
print("\n主成分命名解释（基于载荷绝对值>0.7）:")
for i in range(min(3, n_components)):
    pc_num = i + 1
    high_loadings = loadings_display.iloc[:, i].abs().nlargest(3)
    print(f"\n  PC{pc_num} (解释方差 {explained_var[i] * 100:.1f}%):")
    for var_name, loading in high_loadings.items():
        original_var = pca_vars[var_names_simple.index(var_name)]
        print(f"    • {var_name}: {loadings_display.loc[var_name, f'PC{pc_num}']:.3f}")

print("\n4. 生成图5: PCA散点图")

plt.figure(figsize=(12, 10))

pca_result = pca.transform(pca_data_scaled)

scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                      s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                      c=range(len(df_2023)), cmap='viridis')

# 添加城市标签
for i, city in enumerate(df_2023['城市']):
    plt.annotate(city, (pca_result[i, 0], pca_result[i, 1]),
                 fontsize=11, alpha=0.8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.xlabel(f'第一主成分 PC1 ({explained_var[0] * 100:.1f}%)', fontsize=14)
plt.ylabel(f'第二主成分 PC2 ({explained_var[1] * 100:.1f}%)', fontsize=14)
plt.title('城市在PCA主成分空间中的分布', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('城市序号', fontsize=12)

plt.tight_layout()
plt.savefig('图5_PCA散点图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5. 因子分析尝试（KMO=0.455，不适合）")

# 执行因子分析
fa_vars = factor_vars + ['数据交易额_亿元', '金融科技交易规模_亿元']
fa_data = df_2023[fa_vars].copy().dropna()

# 检查数据是否适合因子分析
try:
    fa_kmo_all, fa_kmo_model = calculate_kmo(fa_data)
    print(f"因子分析KMO值: {fa_kmo_model:.3f}")

    if fa_kmo_model >= 0.7:
        print("数据适合进行因子分析")
        # ... (原有因子分析代码，但KMO=0.455不会执行到这里)
    else:
        print(f"KMO值{fa_kmo_model:.3f}低于0.7，数据不适合因子分析，跳过正式分析")

except Exception as e:
    print(f"因子分析出错: {e}")


print("\n1. 生成图7.1: 轮廓系数确定最优聚类数")
plt.figure(figsize=(10, 6))
# 使用PCA结果进行聚类
X_cluster = pca_result[:, :2]  # 使用前两个主成分

# 确定最优聚类数
sil_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)
    sil_score = silhouette_score(X_cluster, labels)
    sil_scores.append(sil_score)

optimal_k = k_range[np.argmax(sil_scores)]

# 绘制轮廓系数图
plt.plot(k_range, sil_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('聚类数量 (K)', fontsize=14)
plt.ylabel('轮廓系数', fontsize=14)
plt.title('轮廓系数法确定最优聚类数', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'最优K={optimal_k}')
plt.legend()

plt.tight_layout()
plt.savefig('图7.1_轮廓系数确定最优聚类数.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"最优聚类数: {optimal_k} (轮廓系数: {max(sil_scores):.3f})")

print("\n2. 生成图6: K-means聚类结果")

plt.figure(figsize=(12, 10))

# 执行K-means聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_2023['聚类标签'] = kmeans.fit_predict(X_cluster)


def get_cluster_name(cluster_id, cities):
    if cluster_id == 0:
        return "外围发展型"
    elif cluster_id == 1:
        if cities == ['深圳']:
            return "创新引领型"
        else:
            return "核心引领型"
    elif cluster_id == 2:
        if '广州' in cities and '香港' in cities:
            return "枢纽支撑型"
        else:
            return f"类别{cluster_id}"
    else:
        return f"类别{cluster_id}"
# 绘制聚类结果
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
# 获取每个聚类的城市列表
cluster_cities_map = {}
for cluster_id in range(optimal_k):
    cluster_cities = df_2023[df_2023['聚类标签'] == cluster_id]['城市'].tolist()
    cluster_cities_map[cluster_id] = cluster_cities

# 按类别名称排序绘制
for cluster_id in range(optimal_k):
    cluster_data = pca_result[df_2023['聚类标签'] == cluster_id]
    cluster_name = get_cluster_name(cluster_id, cluster_cities_map[cluster_id])
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                color=colors[cluster_id], label=cluster_name)

# 标记聚类中心
centers = kmeans.cluster_centers_[:, :2]
plt.scatter(centers[:, 0], centers[:, 1],
            c='red', marker='X', s=300, alpha=0.9, linewidth=3, label='聚类中心')

# 添加城市标签
for i, city in enumerate(df_2023['城市']):
    plt.annotate(city, (pca_result[i, 0], pca_result[i, 1]),
                 fontsize=10, alpha=0.8)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('第一主成分 PC1', fontsize=14)
plt.ylabel('第二主成分 PC2', fontsize=14)
plt.title(f'K-means聚类结果 (K={optimal_k})', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('图6_Kmeans聚类结果.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出聚类结果
print("\n聚类分组结果:")
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_cities = df_2023[df_2023['聚类标签'] == cluster_id]['城市'].tolist()
    cluster_size = len(cluster_cities)

    # 为每个聚类命名
    cluster_name = get_cluster_name(cluster_id, cluster_cities)

    cluster_summary.append({
        '类别标签': cluster_id,
        '类别名称': cluster_name,
        '城市数量': cluster_size,
        '城市列表': ', '.join(cluster_cities)
    })
    print(f"  {cluster_name} (类别{cluster_id}, {cluster_size}个城市): {', '.join(cluster_cities)}")

# 保存聚类结果
cluster_df = pd.DataFrame(cluster_summary)
cluster_df.to_csv('聚类分析结果.csv', index=False, encoding='utf-8-sig')
print("\n聚类分析结果已保存至: 聚类分析结果.csv")

print("\n3. 生成表7.1: 聚类类别关键指标对比分析")

# 计算各类别均值对比
key_indicators = ['GDP_亿元', '研发经费投入_亿元', '跨境数据传输总量_TB',
                  '数字经济占GDP比重_%', '数据中心机架数', '5G基站数量']

# 按聚类标签分组计算
group_stats = df_2023.groupby('聚类标签')[key_indicators].mean().round(1)

# 创建对比表格
comparison_data = []
for cluster_id in sorted(group_stats.index):
    cluster_name = get_cluster_name(cluster_id, cluster_cities_map.get(cluster_id, []))
    cluster_values = group_stats.loc[cluster_id]

    # 计算与所有城市均值的差异倍数
    overall_mean = df_2023[key_indicators].mean()
    diff_ratio = (cluster_values / overall_mean).round(2)

    comparison_data.append({
        '类别': cluster_name,
        'GDP（亿元）': f"{cluster_values['GDP_亿元']:.1f} ({diff_ratio['GDP_亿元']}倍)",
        '研发投入（亿元）': f"{cluster_values['研发经费投入_亿元']:.1f} ({diff_ratio['研发经费投入_亿元']}倍)",
        '数据流量（TB）': f"{cluster_values['跨境数据传输总量_TB']:.1f} ({diff_ratio['跨境数据传输总量_TB']}倍)",
        '数字经济占比（%）': f"{cluster_values['数字经济占GDP比重_%']:.1f} ({diff_ratio['数字经济占GDP比重_%']}倍)",
        '数据中心机架数': f"{cluster_values['数据中心机架数']:.1f} ({diff_ratio['数据中心机架数']}倍)",
        '5G基站数量': f"{cluster_values['5G基站数量']:.1f} ({diff_ratio['5G基站数量']}倍)"
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n表7.1 聚类类别关键指标对比（2023年）")
print("=" * 100)
print(comparison_df.to_string(index=False))

# 保存到CSV（只保存数值部分）
numeric_comparison_df = group_stats.copy()
numeric_comparison_df['类别名称'] = [get_cluster_name(i, cluster_cities_map.get(i, [])) for i in
                                     numeric_comparison_df.index]
numeric_comparison_df.reset_index(inplace=True)
numeric_comparison_df.to_csv('聚类类别特征对比表.csv', index=False, encoding='utf-8-sig')
print("\n聚类类别特征对比表已保存至: 聚类类别特征对比表.csv")

print("\n4. 生成图7.3: 多类城市群动态演化分析")

# 为每年计算PCA得分
pca_vars_for_history = ['跨境数据传输总量_TB', '数据中心机架数', 'GDP_亿元',
                        '数字经济核心产业增加值_亿元', '研发经费投入_亿元', '5G基站数量']

yearly_pc1_scores = []

for year in sorted(df['年份'].unique()):
    df_year = df[df['年份'] == year].copy()

    # 标准化
    scaler_year = StandardScaler()
    pca_data_year_scaled = scaler_year.fit_transform(df_year[pca_vars_for_history])

    # 使用与2023年相同的PCA对象（基于2023年数据训练的）
    pca_year = PCA(n_components=1)
    pca_year.fit(pca_data_year_scaled)
    pc1_scores_year = pca_year.transform(pca_data_year_scaled)

    # 为每个城市记录PC1得分和年份
    for idx, city in enumerate(df_year['城市']):
        yearly_pc1_scores.append({
            '年份': year,
            '城市': city,
            'PC1_得分': pc1_scores_year[idx][0]
        })

# 转换为DataFrame
df_pc1_history = pd.DataFrame(yearly_pc1_scores)

# 将2023年的聚类标签映射到历史数据
df_2023_labels = df_2023[['城市', '聚类标签']].copy()
df_pc1_history = pd.merge(df_pc1_history, df_2023_labels, on='城市', how='left')

# 为每个聚类添加类别名称
df_pc1_history['类别名称'] = df_pc1_history.apply(
    lambda row: get_cluster_name(row['聚类标签'],
                                 df_2023[df_2023['聚类标签'] == row['聚类标签']]['城市'].tolist()
                                 if row['聚类标签'] in df_2023['聚类标签'].values else []),
    axis=1
)

# 计算每年每类的平均PC1得分
class_yearly_avg = df_pc1_history.groupby(['年份', '类别名称'])['PC1_得分'].mean().reset_index()

# 绘制动态演化图
plt.figure(figsize=(12, 8))

# 为每个类别绘制趋势线
colors_evolve = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为三类城市设置不同颜色
for idx, class_name in enumerate(class_yearly_avg['类别名称'].unique()):
    class_data = class_yearly_avg[class_yearly_avg['类别名称'] == class_name].sort_values('年份')
    if len(class_data) > 0:
        plt.plot(class_data['年份'], class_data['PC1_得分'],
                 marker='o', linewidth=2, markersize=8,
                 color=colors_evolve[idx % len(colors_evolve)], label=class_name)

plt.xlabel('年份', fontsize=14)
plt.ylabel('PC1平均得分', fontsize=14)
plt.title('不同类别城市群PC1得分动态演化 (2019-2023)',
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(title='城市类别', fontsize=11)
plt.tight_layout()
plt.savefig('图7.3_多类城市群动态演化.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 图7.3 '多类城市群动态演化' 已生成")

# 计算演化趋势
print("\n动态演化趋势分析:")
for class_name in class_yearly_avg['类别名称'].unique():
    class_data = class_yearly_avg[class_yearly_avg['类别名称'] == class_name].sort_values('年份')
    if len(class_data) >= 2:
        initial_score = class_data.iloc[0]['PC1_得分']
        final_score = class_data.iloc[-1]['PC1_得分']
        change = final_score - initial_score
        change_percent = (change / abs(initial_score) * 100) if initial_score != 0 else 0
        print(f"  {class_name}: {initial_score:.3f} → {final_score:.3f}, 变化: {change:+.3f} ({change_percent:+.1f}%)")

print("\n5. 判别分析验证")

# 准备数据：使用PCA得分作为特征，聚类标签作为目标
X = pca_result[:, :2]  # 前两个主成分
y = df_2023['聚类标签']

# 线性判别分析
try:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # 预测并评估
    y_pred = lda.predict(X)

    # 计算准确率
    accuracy = np.mean(y_pred == y)
    print(f"判别分析准确率: {accuracy:.1%}")

    # 创建混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y, y_pred)

    print("\n混淆矩阵:")
    print(cm)

    print("\n分类报告:")
    # 获取类别名称列表
    target_names = [get_cluster_name(i, cluster_cities_map.get(i, [])) for i in sorted(y.unique())]
    print(classification_report(y, y_pred, target_names=target_names))

    # 特征重要性（判别函数的系数）
    print("\n判别函数系数 (特征重要性):")
    n_classes = len(lda.coef_)
    for i in range(n_classes):
        class_name = get_cluster_name(i, cluster_cities_map.get(i, []))
        feature_importance = pd.DataFrame({
            '特征': ['PC1', 'PC2'],
            '系数': lda.coef_[i],
            '系数绝对值': np.abs(lda.coef_[i]),
            '相对重要性(%)': (np.abs(lda.coef_[i]) / np.abs(lda.coef_[i]).sum() * 100).round(1)
        })
        print(f"\n判别函数{chr(65 + i)} (对应{class_name}):")
        print(feature_importance.to_string(index=False))

    # 保存判别分析结果
    discriminant_results = {
        '准确率': accuracy,
        '类别数量': optimal_k,
    }
    # 添加每个判别函数的系数
    for i in range(n_classes):
        discriminant_results[f'判别函数{i + 1}_系数_PC1'] = lda.coef_[i][0]
        discriminant_results[f'判别函数{i + 1}_系数_PC2'] = lda.coef_[i][1]

    pd.DataFrame([discriminant_results]).to_csv('判别分析结果.csv', index=False, encoding='utf-8-sig')
    print("\n判别分析结果已保存至: 判别分析结果.csv")

except Exception as e:
    print(f"判别分析出错: {e}")
    print("跳过判别分析部分...")

# 继续生成其他图表...
print("\n生成图9: 复合年增长率分析")
plt.figure(figsize=(14, 10))

# 计算各城市CAGR
cagr_results = []
for city in df['城市'].unique():
    city_df = df[df['城市'] == city].sort_values('年份')
    if len(city_df) >= 2:
        initial = city_df['跨境数据传输总量_TB'].iloc[0]
        final = city_df['跨境数据传输总量_TB'].iloc[-1]
        years = len(city_df) - 1
        if initial > 0:
            cagr = (final / initial) ** (1 / years) - 1
        else:
            cagr = 0
        cagr_results.append({'城市': city, 'CAGR': cagr * 100})

cagr_df = pd.DataFrame(cagr_results).sort_values('CAGR', ascending=True)  # 升序

# 根据聚类类别着色
cagr_df = pd.merge(cagr_df, df_2023[['城市', '聚类标签']], on='城市', how='left')
cagr_df['类别名称'] = cagr_df.apply(
    lambda row: get_cluster_name(row['聚类标签'],
                                 df_2023[df_2023['聚类标签'] == row['聚类标签']]['城市'].tolist()
                                 if row['聚类标签'] in df_2023['聚类标签'].values else []),
    axis=1
)

# 绘制CAGR图
fig, ax = plt.subplots(figsize=(14, 10))
colors_map = {'外围发展型': '#1f77b4', '创新引领型': '#ff7f0e', '枢纽支撑型': '#2ca02c'}
bars = []
for i, (city, cagr, class_name) in enumerate(zip(cagr_df['城市'], cagr_df['CAGR'], cagr_df['类别名称'])):
    color = colors_map.get(class_name, 'gray')
    bar = ax.barh(i, cagr, color=color, edgecolor='black', linewidth=1)
    bars.append(bar[0])

    # 添加数值标签
    color_text = 'red' if cagr < 0 else 'green'
    ax.text(cagr + (0.5 if cagr >= 0 else -3), i,
            f'{cagr:.1f}%', va='center', fontsize=10, color=color_text, fontweight='bold')

ax.set_yticks(range(len(cagr_df)))
ax.set_yticklabels(cagr_df['城市'])
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('复合年增长率 (%)', fontsize=14)
ax.set_ylabel('城市', fontsize=14)
ax.set_title('各城市跨境数据传输总量复合年增长率(CAGR) 2019-2023 (按聚类类别着色)',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

# 添加图例
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=color, label=label) for label, color in colors_map.items()]
ax.legend(handles=legend_elements, loc='upper right', title='城市类别')

plt.tight_layout()
plt.savefig('图9_复合年增长率分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存CAGR结果
cagr_df.sort_values('CAGR', ascending=False, inplace=True)
cagr_df.to_csv('复合年增长率分析.csv', index=False, encoding='utf-8-sig')
print("复合年增长率分析结果已保存至: 复合年增长率分析.csv")

print("\n生成图10: 发展路径图")
plt.figure(figsize=(12, 10))

# 选择代表性城市（每个类别选1-2个）
representative_cities = []
for cluster_id in range(optimal_k):
    cities_in_cluster = df_2023[df_2023['聚类标签'] == cluster_id]['城市'].tolist()
    # 每个类别选择1-2个代表性城市
    if len(cities_in_cluster) > 0:
        representative_cities.append(cities_in_cluster[0])
        if len(cities_in_cluster) > 1 and len(representative_cities) < 6:
            representative_cities.append(cities_in_cluster[1])

colors_path = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, city in enumerate(representative_cities):
    city_df = df[df['城市'] == city].sort_values('年份')
    if len(city_df) > 0:
        # 计算数据密度（数据流量/GDP）
        data_density = city_df['跨境数据传输总量_TB'] / city_df['GDP_亿元'] * 1000
        digital_share = city_df['数字经济占GDP比重_%']

        # 获取该城市所属类别
        city_class = df_2023[df_2023['城市'] == city]['聚类标签'].values[0]
        class_name = get_cluster_name(city_class,
                                      df_2023[df_2023['聚类标签'] == city_class]['城市'].tolist())

        # 绘制路径
        plt.plot(data_density, digital_share, marker='o', linewidth=2.5,
                 markersize=8, color=colors_path[idx % len(colors_path)],
                 label=f'{city} ({class_name})')

        # 标记起点和终点
        plt.scatter(data_density.iloc[0], digital_share.iloc[0],
                    s=100, color='red', zorder=5, marker='s',
                    label=f'{city}起点' if idx == 0 else "")
        plt.scatter(data_density.iloc[-1], digital_share.iloc[-1],
                    s=100, color='green', zorder=5, marker='^',
                    label=f'{city}终点' if idx == 0 else "")

        # 添加年份标签
        for idx_year, year in enumerate(city_df['年份']):
            plt.annotate(str(year), (data_density.iloc[idx_year], digital_share.iloc[idx_year]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)

plt.xlabel('数据密度 (TB/十亿GDP)', fontsize=14)
plt.ylabel('数字经济占GDP比重 (%)', fontsize=14)
plt.title('代表性城市数据要素发展路径演变 (2019-2023)', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 添加图例说明
plt.text(0.02, 0.98, '起点 (2019年)', transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
plt.text(0.02, 0.94, '终点 (2023年)', transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

plt.tight_layout()
plt.savefig('图10_发展路径图.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存详细分析结果
df_2023.to_csv('大湾区数据要素分析结果_2023.csv', index=False, encoding='utf-8-sig')
print(f"\n详细分析结果已保存至: 大湾区数据要素分析结果_2023.csv")
print(f"📊 最优聚类数: {optimal_k}类 (轮廓系数: {max(sil_scores):.3f})")
print(f"🏙️ 聚类结果:")
for cluster in cluster_summary:
    print(f"  • {cluster['类别名称']}: {cluster['城市数量']}个城市")
print(f"📊 判别分析准确率: {accuracy:.1%}") if 'accuracy' in locals() else None
