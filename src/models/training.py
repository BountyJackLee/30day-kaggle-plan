
"""
INTJ模型训练系统 - 稳健、可复现、自动化的模型训练框架
基于30天Kaggle竞赛经验的最佳实践
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.base import BaseEstimator, clone
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class INTJModelFactory:
    """模型工厂 - 创建和管理各种机器学习模型"""
    
    @staticmethod
    def create_lightgbm(params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """创建LightGBM模型"""
        default_params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 1.3,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        return lgb.LGBMClassifier(**default_params)
    
    @staticmethod
    def create_xgboost(params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """创建XGBoost模型"""
        default_params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
        
        return xgb.XGBClassifier(**default_params)
    
    @staticmethod
    def create_catboost(params: Optional[Dict] = None) -> CatBoostClassifier:
        """创建CatBoost模型"""
        default_params = {
            'iterations': 150,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        if params:
            default_params.update(params)
        
        return CatBoostClassifier(**default_params)
    
    @staticmethod
    def create_random_forest(params: Optional[Dict] = None):
        """创建随机森林模型"""
        from sklearn.ensemble import RandomForestClassifier
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        return RandomForestClassifier(**default_params)
    
    @staticmethod
    def create_model(model_type: str, params: Optional[Dict] = None):
        """通用模型创建函数"""
        model_creators = {
            'lightgbm': INTJModelFactory.create_lightgbm,
            'xgboost': INTJModelFactory.create_xgboost,
            'catboost': INTJModelFactory.create_catboost,
            'random_forest': INTJModelFactory.create_random_forest
        }
        
        if model_type not in model_creators:
            raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: {list(model_creators.keys())}")
        
        return model_creators[model_type](params)

class INTJCrossValidator:
    """交叉验证器 - 稳健的模型评估"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 stratified: bool = True,
                 shuffle: bool = True,
                 random_state: int = 42):
        """
        初始化交叉验证器
        
        参数:
            n_splits: 交叉验证折数
            stratified: 是否使用分层交叉验证
            shuffle: 是否打乱数据
            random_state: 随机种子
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        
        # 创建交叉验证器
        if stratified:
            self.cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            self.cv = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
    
    def cross_validate(self,
                      model,
                      X: pd.DataFrame,
                      y: pd.Series,
                      metrics: List[str] = ['accuracy', 'roc_auc'],
                      return_models: bool = False,
                      verbose: bool = True) -> Dict:
        """
        执行交叉验证
        
        返回:
            包含验证结果的字典
        """
        if verbose:
            print(f"🔍 开始{self.n_splits}折交叉验证...")
        
        # 初始化存储
        fold_results = []
        oof_predictions = np.zeros(len(X))
        oof_probas = np.zeros(len(X))
        
        # 如果return_models为True，存储模型
        trained_models = [] if return_models else None
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            if verbose:
                print(f"  折叠 {fold+1}/{self.n_splits}")
            
            # 分割数据
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # 预测
            if hasattr(fold_model, 'predict_proba'):
                val_probas = fold_model.predict_proba(X_val)[:, 1]
                val_preds = (val_probas > 0.5).astype(int)
                oof_probas[val_idx] = val_probas
            else:
                val_preds = fold_model.predict(X_val)
                val_probas = None
            
            oof_predictions[val_idx] = val_preds
            
            # 计算指标
            fold_metrics = self._calculate_metrics(y_val, val_preds, val_probas, metrics)
            fold_results.append(fold_metrics)
            
            # 存储模型
            if return_models:
                trained_models.append(fold_model)
            
            if verbose:
                metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in fold_metrics.items()])
                print(f"    结果: {metric_str}")
        
        # 汇总结果
        summary = self._summarize_results(fold_results, metrics)
        summary['oof_predictions'] = oof_predictions
        summary['oof_probas'] = oof_probas
        
        # 计算OOF指标
        oof_metrics = self._calculate_metrics(y, oof_predictions, oof_probas, metrics)
        summary['oof_metrics'] = oof_metrics
        
        if verbose:
            print(f"✅ 交叉验证完成")
            print(f"📊 平均结果: {summary['mean']}")
            print(f"📈 OOF结果: {oof_metrics}")
        
        if return_models:
            summary['models'] = trained_models
        
        return summary
    
    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray],
                          metrics: List[str]) -> Dict:
        """计算指标"""
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == 'roc_auc' and y_proba is not None:
                results[metric] = roc_auc_score(y_true, y_proba)
            elif metric == 'f1':
                results[metric] = f1_score(y_true, y_pred)
            elif metric == 'log_loss' and y_proba is not None:
                results[metric] = log_loss(y_true, y_proba)
            else:
                raise ValueError(f"不支持的指标: {metric}")
        
        return results
    
    def _summarize_results(self, fold_results: List[Dict], metrics: List[str]) -> Dict:
        """汇总结果"""
        summary = {
            'fold_results': fold_results,
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            summary['mean'][metric] = np.mean(values)
            summary['std'][metric] = np.std(values)
            summary['min'][metric] = np.min(values)
            summary['max'][metric] = np.max(values)
        
        return summary

class INTJThresholdOptimizer:
    """阈值优化器 - 基于预测概率优化分类阈值"""
    
    def __init__(self, 
                 metric: str = 'f1',
                 threshold_range: Tuple[float, float] = (0.3, 0.7),
                 num_points: int = 50):
        """
        初始化阈值优化器
        
        参数:
            metric: 优化指标 ('f1', 'accuracy', 'custom')
            threshold_range: 阈值搜索范围
            num_points: 搜索点数
        """
        self.metric = metric
        self.threshold_range = threshold_range
        self.num_points = num_points
        
        # 存储结果
        self.results_ = None
        self.best_threshold_ = None
        self.best_score_ = None
    
    def optimize(self,
                 y_true: np.ndarray,
                 y_proba: np.ndarray,
                 custom_metric: Optional[Callable] = None) -> Dict:
        """
        优化阈值
        
        返回:
            优化结果字典
        """
        thresholds = np.linspace(self.threshold_range[0], 
                                self.threshold_range[1], 
                                self.num_points)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            if self.metric == 'custom' and custom_metric:
                score = custom_metric(y_true, y_pred, y_proba)
            elif self.metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif self.metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"不支持的指标: {self.metric}")
            
            results.append({
                'threshold': threshold,
                'score': score,
                'positive_rate': y_pred.mean()
            })
        
        # 找到最佳阈值
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        self.results_ = results_df
        self.best_threshold_ = best_result['threshold']
        self.best_score_ = best_result['score']
        
        return {
            'best_threshold': self.best_threshold_,
            'best_score': self.best_score_,
            'positive_rate_at_best': best_result['positive_rate'],
            'all_results': results_df
        }
    
    def plot_optimization(self, figsize: Tuple[int, int] = (10, 6)):
        """绘制优化曲线"""
        import matplotlib.pyplot as plt
        
        if self.results_ is None:
            raise ValueError("请先运行optimize()方法")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 指标 vs 阈值
        ax1.plot(self.results_['threshold'], self.results_['score'], 
                'b-', linewidth=2, label='Score')
        ax1.axvline(self.best_threshold_, color='r', linestyle='--', 
                   label=f'Best: {self.best_threshold_:.3f}')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Threshold Optimization ({self.metric})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 正类比例 vs 阈值
        ax2.plot(self.results_['threshold'], self.results_['positive_rate'],
                'g-', linewidth=2, label='Positive Rate')
        ax2.axvline(self.best_threshold_, color='r', linestyle='--')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Positive Rate')
        ax2.set_title('Positive Rate vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class INTJModelTrainer:
    """模型训练器 - 完整的训练流程管理"""
    
    def __init__(self,
                 model_type: str = 'lightgbm',
                 model_params: Optional[Dict] = None,
                 cv_strategy: str = 'stratified',
                 n_folds: int = 5):
        """
        初始化模型训练器
        
        参数:
            model_type: 模型类型
            model_params: 模型参数
            cv_strategy: 交叉验证策略
            n_folds: 交叉验证折数
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        
        # 创建模型
        self.model = INTJModelFactory.create_model(model_type, model_params)
        
        # 创建交叉验证器
        self.cv = INTJCrossValidator(
            n_splits=n_folds,
            stratified=(cv_strategy == 'stratified')
        )
        
        # 存储结果
        self.cv_results_ = None
        self.final_model_ = None
        self.feature_importance_ = None
    
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              optimize_threshold: bool = False,
              verbose: bool = True) -> Dict:
        """
        训练模型
        
        返回:
            训练结果字典
        """
        if verbose:
            print(f"🚀 开始训练 {self.model_type} 模型")
            print(f"📊 数据形状: {X_train.shape}")
        
        # 1. 交叉验证
        if verbose:
            print("📈 执行交叉验证...")
        
        self.cv_results_ = self.cv.cross_validate(
            model=self.model,
            X=X_train,
            y=y_train,
            return_models=True,
            verbose=verbose
        )
        
        # 2. 训练最终模型
        if verbose:
            print("🔧 训练最终模型...")
        
        self.final_model_ = clone(self.model)
        self.final_model_.fit(X_train, y_train)
        
        # 3. 特征重要性
        if hasattr(self.final_model_, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.final_model_.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # 4. 阈值优化
        threshold_result = None
        if optimize_threshold and self.cv_results_['oof_probas'] is not None:
            if verbose:
                print("🎯 优化分类阈值...")
            
            optimizer = INTJThresholdOptimizer()
            threshold_result = optimizer.optimize(y_train, self.cv_results_['oof_probas'])
        
        # 5. 验证集评估（如果有）
        val_metrics = None
        if X_val is not None and y_val is not None:
            if verbose:
                print("📋 验证集评估...")
            
            if hasattr(self.final_model_, 'predict_proba'):
                val_probas = self.final_model_.predict_proba(X_val)[:, 1]
                
                # 使用最佳阈值或默认阈值
                if threshold_result:
                    best_threshold = threshold_result['best_threshold']
                    val_preds = (val_probas > best_threshold).astype(int)
                else:
                    val_preds = (val_probas > 0.5).astype(int)
            else:
                val_preds = self.final_model_.predict(X_val)
                val_probas = None
            
            # 计算指标
            val_metrics = {
                'accuracy': accuracy_score(y_val, val_preds)
            }
            
            if val_probas is not None:
                val_metrics['roc_auc'] = roc_auc_score(y_val, val_probas)
                val_metrics['log_loss'] = log_loss(y_val, val_probas)
        
        # 汇总结果
        results = {
            'model_type': self.model_type,
            'cv_summary': self.cv_results_['mean'],
            'cv_std': self.cv_results_['std'],
            'oof_metrics': self.cv_results_['oof_metrics'],
            'final_model': self.final_model_,
            'feature_importance': self.feature_importance_,
            'threshold_optimization': threshold_result,
            'validation_metrics': val_metrics,
            'training_complete': True
        }
        
        if verbose:
            print("✅ 训练完成")
            print(f"📊 CV平均准确率: {results['cv_summary'].get('accuracy', 0):.4f}")
            print(f"📊 OOF准确率: {results['oof_metrics'].get('accuracy', 0):.4f}")
            
            if val_metrics:
                print(f"📊 验证集准确率: {val_metrics.get('accuracy', 0):.4f}")
        
        return results
    
    def predict(self, 
                X: pd.DataFrame, 
                threshold: Optional[float] = None,
                return_proba: bool = False):
        """
        使用最终模型进行预测
        
        参数:
            X: 特征数据
            threshold: 分类阈值（None表示使用0.5）
            return_proba: 是否返回概率
        
        返回:
            预测结果
        """
        if self.final_model_ is None:
            raise ValueError("请先训练模型")
        
        if hasattr(self.final_model_, 'predict_proba'):
            probas = self.final_model_.predict_proba(X)
            
            if return_proba:
                return probas
            
            # 应用阈值
            if threshold is None:
                # 使用训练时优化的阈值
                if (self.cv_results_ and 
                    'threshold_optimization' in self.cv_results_ and 
                    self.cv_results_['threshold_optimization']):
                    threshold = self.cv_results_['threshold_optimization']['best_threshold']
                else:
                    threshold = 0.5
            
            return (probas[:, 1] > threshold).astype(int)
        else:
            return self.final_model_.predict(X)
    
    def save_model(self, filepath: str):
        """保存模型"""
        import pickle
        
        if self.final_model_ is None:
            raise ValueError("没有训练好的模型可以保存")
        
        model_data = {
            'model': self.final_model_,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance_,
            'cv_results': self.cv_results_,
            'feature_names': list(self.final_model_.feature_names_in_) if hasattr(self.final_model_, 'feature_names_in_') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建训练器实例
        trainer = cls(
            model_type=model_data['model_type']
        )
        
        # 恢复状态
        trainer.final_model_ = model_data['model']
        trainer.feature_importance_ = model_data['feature_importance']
        trainer.cv_results_ = model_data['cv_results']
        
        return trainer

# 模型评估工具
class ModelEvaluationUtils:
    """模型评估工具类"""
    
    @staticmethod
    def create_model_comparison_report(models_results: List[Dict]) -> pd.DataFrame:
        """创建模型比较报告"""
        comparison_data = []
        
        for result in models_results:
            comparison_data.append({
                'model_type': result.get('model_type', 'Unknown'),
                'cv_accuracy_mean': result.get('cv_summary', {}).get('accuracy', 0),
                'cv_accuracy_std': result.get('cv_std', {}).get('accuracy', 0),
                'oof_accuracy': result.get('oof_metrics', {}).get('accuracy', 0),
                'cv_roc_auc_mean': result.get('cv_summary', {}).get('roc_auc', 0),
                'cv_roc_auc_std': result.get('cv_std', {}).get('roc_auc', 0),
                'oof_roc_auc': result.get('oof_metrics', {}).get('roc_auc', 0),
                'training_time': result.get('training_time', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 排序
        df = df.sort_values('oof_accuracy', ascending=False).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame, 
                             metric: str = 'oof_accuracy',
                             figsize: Tuple[int, int] = (10, 6)):
        """绘制模型比较图"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 排序数据
        comparison_df = comparison_df.sort_values(metric, ascending=True)
        
        # 创建条形图
        y_pos = np.arange(len(comparison_df))
        ax.barh(y_pos, comparison_df[metric], color='steelblue', alpha=0.8)
        
        # 添加误差条（如果有）
        if f'cv_{metric}_std' in comparison_df.columns:
            std_col = f'cv_{metric}_std'
            ax.errorbar(comparison_df[metric], y_pos, 
                       xerr=comparison_df[std_col], 
                       fmt='none', color='black', capsize=3)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comparison_df['model_type'])
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison by {metric.replace("_", " ").title()}')
        
        # 添加数值标签
        for i, v in enumerate(comparison_df[metric]):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        return fig

# 导出主要类
__all__ = [
    'INTJModelFactory',
    'INTJCrossValidator',
    'INTJThresholdOptimizer',
    'INTJModelTrainer',
    'ModelEvaluationUtils'
]
