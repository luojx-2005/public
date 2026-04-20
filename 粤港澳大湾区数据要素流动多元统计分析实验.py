import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import pingouin as pg
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def check_data_file(file_path):
    """检查数据文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"数据文件 {file_path} 不存在。\n"
            "请将老师提供的数据文件放在项目根目录下。\n"
            "数据格式要求：包含字段 城市、年份、GDP_亿元、跨境数据传输总量_TB、"
            "研发经费投入_亿元、数字经济占GDP比重_%、数据中心机架数、5G基站数量 等"
        )


def load_and_prepare_data(file_path):
    """加载和预处理数据"""
    check_data_file(file_path)
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df_2023 = df[df['年份'] == 2023].copy()
    return df, df_2023


def plot_trend_analysis(df):
    """图1: 跨境数据传输总量趋势"""
    plt.figure(figsize=(12, 8))
    yearly_avg = df.groupby('年份')['跨境数据传输总量_TB'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, 
             markersize=10, color='royalblue')
    plt.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.2, color='royalblue')
    plt.title('粤港澳大湾区跨境数据传输总量年均变化趋势 (2019-2023)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('年份', fontsize=14)
    plt.ylabel('跨境数据传输总量 (TB)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for i, (x, y) in enumerate(zip(yearly_avg.index, yearly_avg.values)):
        if i > 0:
            growth = (y - yearly_avg.values[i - 1]) / yearly_avg.values[i - 1] * 100
            plt.annotate(f'+{growth:.1f}%', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('图1_跨境数据传输趋势.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_rd_comparison(df_2023):
    """图2: 各城市研发投入对比"""
    plt.figure(figsize=(14, 8))
    sorted_rd = df_2023.sort_values('研发经费投入_亿元', ascending=True)
    bars = plt.barh(sorted_rd['城市'], sorted_rd['研发经费投入_亿元'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(sorted_rd))))
    plt.title('2023年粤港澳大湾区各城市研发经费投入对比', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('研发经费投入 (亿元)', fontsize=14)
    plt.ylabel('城市', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('图2_各城市研发投入对比.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df):
    """图3: 相关性热力图"""
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
    
    # 保存相关性矩阵
    corr_matrix.to_csv('相关性矩阵.csv', encoding='utf-8-sig')


def partial_correlation_analysis(df_2023):
    """偏相关分析"""
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
            pc = pg.partial_corr(data=df_partial,
                                 x='跨境数据传输总量_TB',
                                 y=var,
                                 covar='GDP_亿元',
                                 method='pearson')
            simple_r = df_partial['跨境数据传输总量_TB'].corr(df_partial[var])
            
            result = {
                '变量': var,
                '简单相关系数': simple_r,
                '偏相关系数': pc['r'].values[0],
                'p值': pc['p-val'].values[0],
                '样本量': pc['n'].values[0],
                '变化': pc['r'].values[0] - simple_r
            }
            partial_results.append(result)
            
            print(f"目标变量：跨境数据传输总量_TB 与 {var}")
            print(f"  简单相关系数: {simple_r:.3f}")
            print(f"  偏相关系数:   {pc['r'].values[0]:.3f}")
            print(f"  p值:         {pc['p-val'].values[0]:.4f}")
            print("-" * 60)
        except Exception as e:
            print(f"计算{var}偏相关时出错: {e}")
    
    partial_df = pd.DataFrame(partial_results)
    partial_df.to_csv('偏相关分析结果.csv', index=False, encoding='utf-8-sig')


def kmo_bartlett_test(df_2023):
    """KMO与Bartlett球形检验"""
    factor_vars = ['跨境数据传输总量_TB', '数据中心机架数', 'GDP_亿元',
                   '数字经济核心产业增加值_亿元', '研发经费投入_亿元', '5G基站数量']
    factor_data = df_2023[factor_vars].dropna()
    
    kmo_all, kmo_model = calculate_kmo(factor_data)
    print(f"KMO检验值: {kmo_model:.3f}")
    
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
    print(f"KMO判断: {kmo_judge}")
    
    chi_square_value, p_value = calculate_bartlett_sphericity(factor_data)
    print(f"\nBartlett球形检验:")
    print(f"  近似卡方值: {chi_square_value:.2f}")
    print(f"  自由度: {len(factor_vars) * (len(factor_vars) - 1) // 2:.0f}")
    print(f"  显著性p值: {p_value:.6f}")
    
    # 保存结果
    kmo_bartlett_result = pd.DataFrame({
        '检验指标': ['KMO值', 'Bartlett卡方值', '自由度', 'p值'],
        '数值': [kmo_model, chi_square_value, len(factor_vars) * (len(factor_vars) - 1) // 2, p_value],
        '判断标准': ['≥0.7为适合', 'p<0.05为显著', '', 'p<0.05拒绝原假设'],
        '结论': [kmo_judge, '极其显著' if p_value < 0.001 else ('显著' if p_value < 0.05 else '不显著'), '',
                 '数据存在相关性' if p_value < 0.05 else '数据独立']
    })
    kmo_bartlett_result.to_csv('KMO_Bartlett检验结果.csv', index=False, encoding='utf-8-sig')
    
    return factor_vars, factor_data


def pca_analysis(df_2023, factor_vars):
    """PCA主成分分析"""
    pca_data = df_2023[factor_vars].copy()
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    pca = PCA()
    pca.fit(pca_data_scaled)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # 图4: PCA方差解释率
    plt.figure(figsize=(10, 8))
    components = range(1, len(explained_var) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    bars = ax1.bar(components, explained_var, alpha=0.6, color='skyblue', label='单个成分解释率')
    ax1.set_xlabel('主成分', fontsize=14)
    ax1.set_ylabel('单个成分解释率', fontsize=14, color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(components)
    
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative_var, 'r-', marker='o', linewidth=3,
             markersize=8, label='累计解释率')
    ax2.set_ylabel('累计解释率', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 1.1])
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, linewidth=2, label='80%阈值')
    
    plt.title('PCA方差解释率分析（碎石图）', fontsize=16, fontweight='bold', pad=20)
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
    
    # 保存PCA结果
    pca_details_df = pd.DataFrame({
        '主成分': [f'PC{i + 1}' for i in range(len(explained_var))],
        '特征值': pca.explained_variance_,
        '方差贡献率(%)': explained_var * 100,
        '累计贡献率(%)': cumulative_var * 100
    })
    pca_details_df.to_csv('PCA_特征值与贡献率_精确表.csv', index=False, encoding='utf-8-sig')
    
    # 载荷矩阵
    n_components = sum(pca.explained_variance_ > 1)
    pca_components = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        pca_components[:, :min(3, n_components)],
        index=factor_vars,
        columns=[f'PC{i + 1}' for i in range(min(3, n_components))]
    )
    loadings_df.to_csv('主成分载荷矩阵.csv', encoding='utf-8-sig')
    
    # PCA散点图
    pca_result = pca.transform(pca_data_scaled)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                          c=range(len(df_2023)), cmap='viridis')
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
    plt.colorbar(scatter, label='城市序号')
    plt.tight_layout()
    plt.savefig('图5_PCA散点图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, pca_data_scaled, pca_result, explained_var


def clustering_analysis(df_2023, pca_result, explained_var):
    """K-means聚类分析"""
    X_cluster = pca_result[:, :2]
    
    # 轮廓系数确定最优K
    plt.figure(figsize=(10, 6))
    sil_scores = []
    k_range = range(2, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        sil_score = silhouette_score(X_cluster, labels)
        sil_scores.append(sil_score)
    
    optimal_k = k_range[np.argmax(sil_scores)]
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
    
    # 执行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_2023['聚类标签'] = kmeans.fit_predict(X_cluster)
    
    def get_cluster_name(cluster_id, cities):
        if cluster_id == 0:
            return "外围发展型"
        elif cluster_id == 1:
            return "创新引领型"
        elif cluster_id == 2:
            return "枢纽支撑型"
        else:
            return f"类别{cluster_id}"
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    cluster_cities_map = {}
    for cluster_id in range(optimal_k):
        cluster_cities = df_2023[df_2023['聚类标签'] == cluster_id]['城市'].tolist()
        cluster_cities_map[cluster_id] = cluster_cities
        cluster_data = pca_result[df_2023['聚类标签'] == cluster_id]
        cluster_name = get_cluster_name(cluster_id, cluster_cities_map[cluster_id])
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                    color=colors[cluster_id], label=cluster_name)
    
    centers = kmeans.cluster_centers_[:, :2]
    plt.scatter(centers[:, 0], centers[:, 1],
                c='red', marker='X', s=300, alpha=0.9, linewidth=3, label='聚类中心')
    
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
        cluster_name = get_cluster_name(cluster_id, cluster_cities)
        cluster_summary.append({
            '类别标签': cluster_id,
            '类别名称': cluster_name,
            '城市数量': len(cluster_cities),
            '城市列表': ', '.join(cluster_cities)
        })
        print(f"  {cluster_name}: {', '.join(cluster_cities)}")
    
    cluster_df = pd.DataFrame(cluster_summary)
    cluster_df.to_csv('聚类分析结果.csv', index=False, encoding='utf-8-sig')
    
    return optimal_k, kmeans, cluster_cities_map, get_cluster_name


def lda_verification(df_2023, pca_result, optimal_k, get_cluster_name, cluster_cities_map):
    """LDA判别分析验证"""
    X = pca_result[:, :2]
    y = df_2023['聚类标签']
    
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        y_pred = lda.predict(X)
        accuracy = np.mean(y_pred == y)
        print(f"判别分析准确率: {accuracy:.1%}")
        
        cm = confusion_matrix(y, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        
        target_names = [get_cluster_name(i, cluster_cities_map.get(i, [])) for i in sorted(y.unique())]
        print("\n分类报告:")
        print(classification_report(y, y_pred, target_names=target_names))
        
        # 保存判别分析结果
        discriminant_results = {'准确率': accuracy, '类别数量': optimal_k}
        pd.DataFrame([discriminant_results]).to_csv('判别分析结果.csv', index=False, encoding='utf-8-sig')
        
        return accuracy
    except Exception as e:
        print(f"判别分析出错: {e}")
        return None


def dynamic_evolution_analysis(df, pca_result, df_2023, optimal_k, get_cluster_name, cluster_cities_map):
    """动态演化分析"""
    pca_vars_for_history = ['跨境数据传输总量_TB', '数据中心机架数', 'GDP_亿元',
                            '数字经济核心产业增加值_亿元', '研发经费投入_亿元', '5G基站数量']
    
    yearly_pc1_scores = []
    for year in sorted(df['年份'].unique()):
        df_year = df[df['年份'] == year].copy()
        scaler_year = StandardScaler()
        pca_data_year_scaled = scaler_year.fit_transform(df_year[pca_vars_for_history])
        pca_year = PCA(n_components=1)
        pca_year.fit(pca_data_year_scaled)
        pc1_scores_year = pca_year.transform(pca_data_year_scaled)
        
        for idx, city in enumerate(df_year['城市']):
            yearly_pc1_scores.append({
                '年份': year,
                '城市': city,
                'PC1_得分': pc1_scores_year[idx][0]
            })
    
    df_pc1_history = pd.DataFrame(yearly_pc1_scores)
    df_2023_labels = df_2023[['城市', '聚类标签']].copy()
    df_pc1_history = pd.merge(df_pc1_history, df_2023_labels, on='城市', how='left')
    df_pc1_history['类别名称'] = df_pc1_history.apply(
        lambda row: get_cluster_name(row['聚类标签'],
                                     df_2023[df_2023['聚类标签'] == row['聚类标签']]['城市'].tolist()
                                     if row['聚类标签'] in df_2023['聚类标签'].values else []),
        axis=1
    )
    
    class_yearly_avg = df_pc1_history.groupby(['年份', '类别名称'])['PC1_得分'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    colors_evolve = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, class_name in enumerate(class_yearly_avg['类别名称'].unique()):
        class_data = class_yearly_avg[class_yearly_avg['类别名称'] == class_name].sort_values('年份')
        if len(class_data) > 0:
            plt.plot(class_data['年份'], class_data['PC1_得分'],
                     marker='o', linewidth=2, markersize=8,
                     color=colors_evolve[idx % len(colors_evolve)], label=class_name)
    
    plt.xlabel('年份', fontsize=14)
    plt.ylabel('PC1平均得分', fontsize=14)
    plt.title('不同类别城市群PC1得分动态演化 (2019-2023)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(title='城市类别', fontsize=11)
    plt.tight_layout()
    plt.savefig('图7.3_多类城市群动态演化.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n动态演化趋势分析:")
    for class_name in class_yearly_avg['类别名称'].unique():
        class_data = class_yearly_avg[class_yearly_avg['类别名称'] == class_name].sort_values('年份')
        if len(class_data) >= 2:
            initial_score = class_data.iloc[0]['PC1_得分']
            final_score = class_data.iloc[-1]['PC1_得分']
            change = final_score - initial_score
            print(f"  {class_name}: {initial_score:.3f} → {final_score:.3f}, 变化: {change:+.3f}")


def cagr_analysis(df, df_2023, get_cluster_name, cluster_cities_map):
    """复合年增长率分析"""
    plt.figure(figsize=(14, 10))
    
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
    
    cagr_df = pd.DataFrame(cagr_results).sort_values('CAGR', ascending=True)
    cagr_df = pd.merge(cagr_df, df_2023[['城市', '聚类标签']], on='城市', how='left')
    cagr_df['类别名称'] = cagr_df.apply(
        lambda row: get_cluster_name(row['聚类标签'],
                                     df_2023[df_2023['聚类标签'] == row['聚类标签']]['城市'].tolist()
                                     if row['聚类标签'] in df_2023['聚类标签'].values else []),
        axis=1
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    colors_map = {'外围发展型': '#1f77b4', '创新引领型': '#ff7f0e', '枢纽支撑型': '#2ca02c'}
    
    for i, (city, cagr, class_name) in enumerate(zip(cagr_df['城市'], cagr_df['CAGR'], cagr_df['类别名称'])):
        color = colors_map.get(class_name, 'gray')
        bar = ax.barh(i, cagr, color=color, edgecolor='black', linewidth=1)
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
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) for label, color in colors_map.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='城市类别')
    
    plt.tight_layout()
    plt.savefig('图9_复合年增长率分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    cagr_df.sort_values('CAGR', ascending=False, inplace=True)
    cagr_df.to_csv('复合年增长率分析.csv', index=False, encoding='utf-8-sig')


def development_path_analysis(df, df_2023, optimal_k, get_cluster_name, cluster_cities_map):
    """发展路径图"""
    plt.figure(figsize=(12, 10))
    
    representative_cities = []
    for cluster_id in range(optimal_k):
        cities_in_cluster = df_2023[df_2023['聚类标签'] == cluster_id]['城市'].tolist()
        if len(cities_in_cluster) > 0:
            representative_cities.append(cities_in_cluster[0])
            if len(cities_in_cluster) > 1 and len(representative_cities) < 6:
                representative_cities.append(cities_in_cluster[1])
    
    colors_path = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, city in enumerate(representative_cities):
        city_df = df[df['城市'] == city].sort_values('年份')
        if len(city_df) > 0:
            data_density = city_df['跨境数据传输总量_TB'] / city_df['GDP_亿元'] * 1000
            digital_share = city_df['数字经济占GDP比重_%']
            
            city_class = df_2023[df_2023['城市'] == city]['聚类标签'].values[0]
            class_name = get_cluster_name(city_class, cluster_cities_map.get(city_class, []))
            
            plt.plot(data_density, digital_share, marker='o', linewidth=2.5,
                     markersize=8, color=colors_path[idx % len(colors_path)],
                     label=f'{city} ({class_name})')
            
            plt.scatter(data_density.iloc[0], digital_share.iloc[0],
                        s=100, color='red', zorder=5, marker='s')
            plt.scatter(data_density.iloc[-1], digital_share.iloc[-1],
                        s=100, color='green', zorder=5, marker='^')
            
            for idx_year, year in enumerate(city_df['年份']):
                plt.annotate(str(year), (data_density.iloc[idx_year], digital_share.iloc[idx_year]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    plt.xlabel('数据密度 (TB/十亿GDP)', fontsize=14)
    plt.ylabel('数字经济占GDP比重 (%)', fontsize=14)
    plt.title('代表性城市数据要素发展路径演变 (2019-2023)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, '起点 (2019年)', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    plt.text(0.02, 0.94, '终点 (2023年)', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('图10_发展路径图.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_final_results(df_2023):
    """保存最终结果"""
    df_2023.to_csv('大湾区数据要素分析结果_2023.csv', index=False, encoding='utf-8-sig')
    print("\n详细分析结果已保存至: 大湾区数据要素分析结果_2023.csv")


def main():
    """主函数"""
    print("=" * 60)
    print("粤港澳大湾区数据要素流动多元统计分析")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n1. 加载数据...")
    df, df_2023 = load_and_prepare_data('main_data_advanced.csv')
    print(f"数据加载完成！总数据: {df.shape[0]}条, 2023年数据: {df_2023.shape[0]}个城市")
    
    # 2. 趋势分析
    print("\n2. 生成图1: 跨境数据传输总量趋势...")
    plot_trend_analysis(df)
    
    # 3. 研发投入对比
    print("\n3. 生成图2: 各城市研发投入对比...")
    plot_rd_comparison(df_2023)
    
    # 4. 相关性分析
    print("\n4. 生成图3: 相关性热力图...")
    plot_correlation_heatmap(df)
    
    # 5. 偏相关分析
    print("\n5. 偏相关分析...")
    partial_correlation_analysis(df_2023)
    
    # 6. KMO与Bartlett检验
    print("\n6. KMO与Bartlett球形检验...")
    factor_vars, factor_data = kmo_bartlett_test(df_2023)
    
    # 7. PCA分析
    print("\n7. PCA主成分分析...")
    pca, pca_data_scaled, pca_result, explained_var = pca_analysis(df_2023, factor_vars)
    
    # 8. 聚类分析
    print("\n8. K-means聚类分析...")
    optimal_k, kmeans, cluster_cities_map, get_cluster_name = clustering_analysis(df_2023, pca_result, explained_var)
    
    # 9. LDA判别验证
    print("\n9. LDA判别分析验证...")
    accuracy = lda_verification(df_2023, pca_result, optimal_k, get_cluster_name, cluster_cities_map)
    
    # 10. 动态演化分析
    print("\n10. 动态演化分析...")
    dynamic_evolution_analysis(df, pca_result, df_2023, optimal_k, get_cluster_name, cluster_cities_map)
    
    # 11. CAGR分析
    print("\n11. 复合年增长率分析...")
    cagr_analysis(df, df_2023, get_cluster_name, cluster_cities_map)
    
    # 12. 发展路径图
    print("\n12. 发展路径图...")
    development_path_analysis(df, df_2023, optimal_k, get_cluster_name, cluster_cities_map)
    
    # 13. 保存结果
    print("\n13. 保存最终结果...")
    save_final_results(df_2023)
    
    # 输出总结
    print("\n" + "=" * 60)
    print("分析完成！")
    if accuracy:
        print(f"📊 最优聚类数: {optimal_k}类")
        print(f"🏙️ 聚类结果: 详见聚类分析结果.csv")
        print(f"📊 判别分析准确率: {accuracy:.1%}")

if __name__ == "__main__":
    main()
