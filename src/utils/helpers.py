
"""
Kaggle竞赛工具库 - 完整版
包含实验管理、数据处理、可视化等核心功能
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ExperimentLogger:
    """
    实验日志记录器 - 用于系统化跟踪所有实验
    INTJ风格的完整实验管理系统
    """
    
    def __init__(self, log_dir: str = "logs/experiments", project_name: str = "kaggle-project"):
        """
        初始化实验日志记录器
        
        参数:
            log_dir: 日志目录
            project_name: 项目名称
        """
        self.log_dir = Path(log_dir)
        self.project_name = project_name
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment_dir = self.log_dir / self.experiment_id
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志数据结构
        self.log_data = {
            "experiment_id": self.experiment_id,
            "project_name": project_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "parameters": {},
            "metrics": {},
            "files": [],
            "status": "running",
            "notes": ""
        }
        
        print(f"🔬 实验 {self.experiment_id} 已启动")
        print(f"📁 日志目录: {self.current_experiment_dir}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """记录实验参数"""
        self.log_data["parameters"].update(params)
        print(f"📝 记录参数: {len(params)} 个")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录评估指标"""
        if step is not None:
            if "step_metrics" not in self.log_data:
                self.log_data["step_metrics"] = {}
            self.log_data["step_metrics"][step] = metrics
        else:
            self.log_data["metrics"].update(metrics)
        print(f"📊 记录指标: {metrics}")
    
    def log_file(self, file_path: str, description: str = "") -> None:
        """记录生成的文件"""
        file_info = {
            "path": file_path,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.log_data["files"].append(file_info)
    
    def log_note(self, note: str) -> None:
        """记录实验笔记"""
        if "notes" not in self.log_data:
            self.log_data["notes"] = ""
        self.log_data["notes"] += f"[{datetime.now().strftime('%H:%M:%S')}] {note}\n"
    
    def save(self, status: str = "completed") -> None:
        """保存实验日志"""
        self.log_data["end_time"] = datetime.now().isoformat()
        self.log_data["status"] = status
        
        # 保存JSON日志
        log_file = self.current_experiment_dir / "experiment_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        
        # 保存人类可读版本
        txt_file = self.current_experiment_dir / "experiment_summary.txt"
        with open(txt_file, 'w') as f:
            f.write(self._generate_summary())
        
        print(f"💾 实验日志已保存: {log_file}")
        print(f"📋 实验状态: {status}")
    
    def _generate_summary(self) -> str:
        """生成实验摘要"""
        summary = f"""实验摘要报告
{'='*60}
实验ID: {self.log_data['experiment_id']}
项目: {self.log_data['project_name']}
开始时间: {self.log_data['start_time']}
结束时间: {self.log_data['end_time']}
状态: {self.log_data['status']}
{'='*60}

📊 关键指标:
{self._format_metrics()}

⚙️ 实验参数:
{self._format_parameters()}

📁 生成文件 ({len(self.log_data.get('files', []))}个):
{self._format_files()}

📝 实验笔记:
{self.log_data.get('notes', '无')}
"""
        return summary
    
    def _format_metrics(self) -> str:
        """格式化指标输出"""
        if not self.log_data.get("metrics"):
            return "  无指标记录"
        
        metrics = self.log_data["metrics"]
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _format_parameters(self) -> str:
        """格式化参数输出"""
        if not self.log_data.get("parameters"):
            return "  无参数记录"
        
        params = self.log_data["parameters"]
        lines = []
        for key, value in params.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _format_files(self) -> str:
        """格式化文件列表"""
        if not self.log_data.get("files"):
            return "  无文件记录"
        
        files = self.log_data["files"]
        lines = []
        for i, file_info in enumerate(files, 1):
            lines.append(f"  {i}. {file_info['path']}")
            if file_info['description']:
                lines.append(f"     描述: {file_info['description']}")
        return "\n".join(lines)

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    
    参数:
        df: 输入DataFrame
        verbose: 是否打印优化信息
    
    返回:
        优化后的DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # 对象类型转换为分类
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f"📉 内存优化: {start_mem:.2f} MB → {end_mem:.2f} MB (减少 {(start_mem-end_mem)/start_mem*100:.1f}%)")
    
    return df

def plot_feature_importance(model, feature_names, top_n: int = 20, figsize=(10, 8)):
    """
    绘制特征重要性图
    
    参数:
        model: 训练好的模型（支持LightGBM、XGBoost、RandomForest等）
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    # 根据模型类型获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance()
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("模型不支持特征重要性分析")
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # 创建水平条形图
    bars = plt.barh(range(len(importance_df)), importance_df['importance'], align='center')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'Top {top_n} 特征重要性')
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'])):
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return importance_df

def analyze_prediction_distribution(predictions: np.ndarray, 
                                   true_labels: Optional[np.ndarray] = None,
                                   thresholds: List[float] = None) -> Dict[str, Any]:
    """
    分析预测概率分布
    
    参数:
        predictions: 预测概率数组（0-1之间）
        true_labels: 真实标签（可选）
        thresholds: 要分析的阈值列表
    
    返回:
        分布分析字典
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    analysis = {
        "统计量": {
            "平均值": float(predictions.mean()),
            "标准差": float(predictions.std()),
            "最小值": float(predictions.min()),
            "最大值": float(predictions.max()),
            "中位数": float(np.median(predictions)),
            "偏度": float(pd.Series(predictions).skew())
        },
        "分布分位数": {
            f"{p}分位": float(np.percentile(predictions, p)) 
            for p in [10, 25, 50, 75, 90]
        },
        "阈值分析": {},
        "预测分类": {}
    }
    
    # 阈值分析
    for threshold in thresholds:
        binary_preds = (predictions > threshold).astype(int)
        analysis["阈值分析"][f"阈值={threshold:.2f}"] = {
            "正类比例": float(binary_preds.mean()),
            "正类数量": int(binary_preds.sum()),
            "负类数量": int(len(binary_preds) - binary_preds.sum())
        }
    
    # 预测分类（基于自然阈值0.5）
    binary_preds = (predictions > 0.5).astype(int)
    analysis["预测分类"]["阈值=0.50"] = {
        "正类比例": float(binary_preds.mean()),
        "正类数量": int(binary_preds.sum()),
        "负类数量": int(len(binary_preds) - binary_preds.sum())
    }
    
    # 如果有真实标签，计算更多指标
    if true_labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        analysis["性能指标"] = {}
        for threshold in thresholds:
            binary_preds = (predictions > threshold).astype(int)
            analysis["性能指标"][f"阈值={threshold:.2f}"] = {
                "准确率": float(accuracy_score(true_labels, binary_preds)),
                "精确率": float(precision_score(true_labels, binary_preds, zero_division=0)),
                "召回率": float(recall_score(true_labels, binary_preds, zero_division=0)),
                "F1分数": float(f1_score(true_labels, binary_preds, zero_division=0))
            }
    
    return analysis

def save_submission(predictions: np.ndarray, 
                   sample_submission_path: str,
                   output_path: str,
                   threshold: float = 0.5,
                   competition_format: str = "binary") -> str:
    """
    生成Kaggle提交文件
    
    参数:
        predictions: 预测概率或标签
        sample_submission_path: 示例提交文件路径
        output_path: 输出文件路径
        threshold: 二分类阈值
        competition_format: 竞赛格式 ('binary', 'probability', 'regression')
    
    返回:
        保存的文件路径
    """
    # 读取示例提交文件
    sample_df = pd.read_csv(sample_submission_path)
    
    # 根据格式处理预测
    if competition_format == "binary":
        # 二分类：应用阈值
        binary_predictions = (predictions > threshold).astype(int)
        sample_df.iloc[:, 1] = binary_predictions
    elif competition_format == "probability":
        # 概率：直接使用
        sample_df.iloc[:, 1] = predictions
    elif competition_format == "regression":
        # 回归：直接使用
        sample_df.iloc[:, 1] = predictions
    else:
        raise ValueError(f"不支持的竞赛格式: {competition_format}")
    
    # 保存文件
    sample_df.to_csv(output_path, index=False)
    print(f"💾 提交文件已保存: {output_path}")
    print(f"📊 预测统计: 形状={sample_df.shape}, 正类比例={sample_df.iloc[:, 1].mean():.3f}")
    
    return output_path

def create_cv_folds(df: pd.DataFrame, 
                   target: str,
                   n_splits: int = 5,
                   stratified: bool = True,
                   shuffle: bool = True,
                   random_state: int = 42) -> pd.DataFrame:
    """
    创建交叉验证折叠
    
    参数:
        df: 输入DataFrame
        target: 目标列名
        n_splits: 折叠数量
        stratified: 是否分层
        shuffle: 是否打乱
        random_state: 随机种子
    
    返回:
        包含fold列的DataFrame
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    df_folds = df.copy()
    
    if stratified and target in df.columns:
        # 分层K折
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        df_folds['fold'] = -1
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target])):
            df_folds.loc[val_idx, 'fold'] = fold
    else:
        # 普通K折
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        df_folds['fold'] = -1
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df_folds.loc[val_idx, 'fold'] = fold
    
    print(f"✅ 创建了 {n_splits} 折交叉验证")
    print(f"📊 每折样本数: {df_folds['fold'].value_counts().sort_index().to_dict()}")
    
    return df_folds

def compare_feature_distributions(train_df: pd.DataFrame, 
                                 test_df: pd.DataFrame,
                                 features: List[str] = None,
                                 max_features: int = 20) -> pd.DataFrame:
    """
    比较训练集和测试集特征分布
    
    参数:
        train_df: 训练集DataFrame
        test_df: 测试集DataFrame
        features: 要比较的特征列表（None表示所有共同特征）
        max_features: 最多显示的特征数量
    
    返回:
        分布比较的DataFrame
    """
    if features is None:
        # 获取共同特征
        common_features = list(set(train_df.columns) & set(test_df.columns))
    else:
        common_features = [f for f in features if f in train_df.columns and f in test_df.columns]
    
    # 限制特征数量
    if len(common_features) > max_features:
        print(f"⚠️  特征过多 ({len(common_features)})，只显示前{max_features}个")
        common_features = common_features[:max_features]
    
    comparison_data = []
    
    for feature in common_features:
        train_vals = train_df[feature]
        test_vals = test_df[feature]
        
        # 数值特征
        if pd.api.types.is_numeric_dtype(train_vals):
            comparison = {
                '特征': feature,
                '类型': '数值',
                '训练集均值': train_vals.mean(),
                '测试集均值': test_vals.mean(),
                '均值差异%': abs((train_vals.mean() - test_vals.mean()) / train_vals.mean() * 100) if train_vals.mean() != 0 else float('inf'),
                '训练集缺失%': train_vals.isna().mean() * 100,
                '测试集缺失%': test_vals.isna().mean() * 100
            }
        else:
            # 类别特征
            train_top = train_vals.mode().iloc[0] if not train_vals.mode().empty else None
            test_top = test_vals.mode().iloc[0] if not test_vals.mode().empty else None
            
            comparison = {
                '特征': feature,
                '类型': '类别',
                '训练集众数': train_top,
                '测试集众数': test_top,
                '众数是否一致': train_top == test_top,
                '训练集缺失%': train_vals.isna().mean() * 100,
                '测试集缺失%': test_vals.isna().mean() * 100
            }
        
        comparison_data.append(comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        print(f"🔍 特征分布比较完成，共比较 {len(comparison_df)} 个特征")
        
        # 识别潜在问题
        numeric_df = comparison_df[comparison_df['类型'] == '数值']
        if not numeric_df.empty:
            problematic = numeric_df[numeric_df['均值差异%'] > 20]
            if len(problematic) > 0:
                print(f"⚠️  发现 {len(problematic)} 个数值特征分布差异 > 20%")
                print(problematic[['特征', '均值差异%']].to_string())
    
    return comparison_df

def visualize_prediction_distribution(predictions: np.ndarray, 
                                     true_labels: Optional[np.ndarray] = None,
                                     figsize: tuple = (12, 8)):
    """
    可视化预测分布
    
    参数:
        predictions: 预测概率
        true_labels: 真实标签（可选）
        figsize: 图形大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 预测概率直方图
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='阈值=0.5')
    axes[0, 0].set_xlabel('预测概率')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('预测概率分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 累积分布函数
    sorted_probs = np.sort(predictions)
    cum_probs = np.arange(1, len(sorted_probs)+1) / len(sorted_probs)
    axes[0, 1].plot(sorted_probs, cum_probs, color='green', linewidth=2)
    axes[0, 1].set_xlabel('预测概率')
    axes[0, 1].set_ylabel('累积比例')
    axes[0, 1].set_title('累积分布函数 (CDF)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 箱线图
    axes[1, 0].boxplot(predictions, vert=False)
    axes[1, 0].set_xlabel('预测概率')
    axes[1, 0].set_title('预测概率箱线图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 如果有真实标签，显示正负类分布
    if true_labels is not None:
        pos_probs = predictions[true_labels == 1]
        neg_probs = predictions[true_labels == 0]
        
        axes[1, 1].hist(pos_probs, bins=30, alpha=0.5, color='green', label='正类', density=True)
        axes[1, 1].hist(neg_probs, bins=30, alpha=0.5, color='red', label='负类', density=True)
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='阈值=0.5')
        axes[1, 1].set_xlabel('预测概率')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title('正负类预测分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # 如果没有真实标签，显示QQ图
        from scipy import stats
        stats.probplot(predictions, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('QQ图（正态性检验）')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 导出所有函数
__all__ = [
    'ExperimentLogger',
    'reduce_memory_usage',
    'plot_feature_importance',
    'analyze_prediction_distribution',
    'save_submission',
    'create_cv_folds',
    'compare_feature_distributions',
    'visualize_prediction_distribution'
]
