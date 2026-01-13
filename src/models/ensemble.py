
"""
INTJ集成学习系统 - 稳健、多样化的模型集成框架
基于30天Kaggle竞赛经验的最佳实践
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class INTJEnsembleBuilder:
    """集成模型构建器 - 创建多样化的模型集合"""
    
    def __init__(self, 
                 base_models: Optional[List[Tuple[str, Dict]]] = None,
                 diversity_metric: str = 'correlation'):
        """
        初始化集成构建器
        
        参数:
            base_models: 基础模型列表 [(model_type, params), ...]
            diversity_metric: 多样性度量指标
        """
        if base_models is None:
            # 默认模型集合（从30天经验中总结）
            base_models = [
                ('lightgbm', {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'scale_pos_weight': 1.3,
                    'random_state': 42
                }),
                ('xgboost', {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'random_state': 42
                }),
                ('catboost', {
                    'iterations': 150,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'random_seed': 42
                })
            ]
        
        self.base_models_config = base_models
        self.diversity_metric = diversity_metric
        self.models_ = []
        self.model_names_ = []
        
        # 从训练模块导入模型工厂
        try:
            from src.models.training import INTJModelFactory
            self.model_factory = INTJModelFactory
        except ImportError:
            # 如果在单独使用，定义简单版本
            self.model_factory = None
    
    def build_models(self) -> List:
        """构建基础模型集合"""
        self.models_ = []
        self.model_names_ = []
        
        for i, (model_type, params) in enumerate(self.base_models_config):
            if self.model_factory:
                model = self.model_factory.create_model(model_type, params)
            else:
                # 简单实现
                if model_type == 'lightgbm':
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**params)
                elif model_type == 'xgboost':
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**params)
                elif model_type == 'catboost':
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(**params)
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.models_.append(model)
            self.model_names_.append(f"{model_type}_{i}")
        
        return self.models_
    
    def calculate_diversity(self, 
                           predictions: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> Dict:
        """
        计算模型预测的多样性
        
        参数:
            predictions: 模型预测矩阵 (n_models, n_samples)
            labels: 真实标签（可选，用于计算成对差异）
        
        返回:
            多样性指标字典
        """
        n_models = predictions.shape[0]
        
        if self.diversity_metric == 'correlation':
            # 计算模型预测之间的相关性
            corr_matrix = np.corrcoef(predictions)
            
            # 平均绝对相关性（越低表示多样性越高）
            mask = ~np.eye(n_models, dtype=bool)
            avg_correlation = np.mean(np.abs(corr_matrix[mask]))
            
            diversity_score = 1 - avg_correlation
            
            return {
                'diversity_metric': 'correlation',
                'diversity_score': diversity_score,
                'avg_correlation': avg_correlation,
                'correlation_matrix': corr_matrix
            }
        
        elif self.diversity_metric == 'disagreement':
            # 计算模型之间的不一致率
            disagreements = []
            
            for i in range(n_models):
                for j in range(i+1, n_models):
                    disagreement_rate = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagreement_rate)
            
            avg_disagreement = np.mean(disagreements)
            
            return {
                'diversity_metric': 'disagreement',
                'diversity_score': avg_disagreement,
                'avg_disagreement': avg_disagreement,
                'disagreement_matrix': self._create_disagreement_matrix(predictions)
            }
        
        else:
            raise ValueError(f"不支持的多样性指标: {self.diversity_metric}")
    
    def _create_disagreement_matrix(self, predictions: np.ndarray) -> np.ndarray:
        """创建不一致矩阵"""
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    matrix[i, j] = np.mean(predictions[i] != predictions[j])
        
        return matrix

class INTJWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """加权集成模型"""
    
    def __init__(self, 
                 base_models: List,
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        初始化加权集成
        
        参数:
            base_models: 基础模型列表
            weights: 模型权重列表（None表示等权重）
            voting: 投票方式 ('hard' 或 'soft')
        """
        self.base_models = base_models
        self.weights = weights
        self.voting = voting
        
        # 如果未提供权重，使用等权重
        if self.weights is None:
            self.weights = [1.0 / len(base_models)] * len(base_models)
        
        # 验证权重
        if len(self.weights) != len(base_models):
            raise ValueError("权重数量必须与模型数量相同")
        
        # 归一化权重
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        # 存储训练后的模型
        self.trained_models_ = []
        self.n_classes_ = None
    
    def fit(self, X, y):
        """训练所有基础模型"""
        self.trained_models_ = []
        
        for model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.trained_models_.append(model_clone)
        
        self.n_classes_ = len(np.unique(y))
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        else:
            return self._soft_voting_predict(X)
    
    def predict_proba(self, X):
        """预测概率（仅适用于soft voting）"""
        if self.voting != 'soft':
            raise ValueError("predict_proba仅适用于soft voting")
        
        return self._soft_voting_predict_proba(X)
    
    def _hard_voting_predict(self, X):
        """硬投票预测"""
        predictions = np.array([model.predict(X) for model in self.trained_models_])
        
        # 加权投票
        weighted_votes = np.zeros((X.shape[0], self.n_classes_))
        
        for i, model_preds in enumerate(predictions):
            for sample_idx, pred_class in enumerate(model_preds):
                weighted_votes[sample_idx, pred_class] += self.weights[i]
        
        return np.argmax(weighted_votes, axis=1)
    
    def _soft_voting_predict(self, X):
        """软投票预测（基于概率）"""
        probas = self._soft_voting_predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def _soft_voting_predict_proba(self, X):
        """软投票概率预测"""
        all_probas = []
        
        for i, model in enumerate(self.trained_models_):
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
            else:
                # 如果模型没有predict_proba，使用one-hot编码的预测
                preds = model.predict(X)
                probas = np.zeros((len(X), self.n_classes_))
                for j, pred in enumerate(preds):
                    probas[j, pred] = 1.0
            
            # 应用权重
            weighted_probas = probas * self.weights[i]
            all_probas.append(weighted_probas)
        
        # 求和得到最终概率
        final_probas = np.sum(all_probas, axis=0)
        
        return final_probas
    
    def get_model_contributions(self, X) -> pd.DataFrame:
        """获取每个模型的贡献度"""
        contributions = []
        
        for i, model in enumerate(self.trained_models_):
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                # 计算平均置信度
                avg_confidence = np.mean(np.max(probas, axis=1))
            else:
                avg_confidence = np.nan
            
            contributions.append({
                'model_index': i,
                'model_type': type(model).__name__,
                'weight': self.weights[i],
                'avg_confidence': avg_confidence,
                'contribution_score': self.weights[i] * (avg_confidence if not np.isnan(avg_confidence) else 1.0)
            })
        
        return pd.DataFrame(contributions).sort_values('contribution_score', ascending=False)

class INTJStackingEnsemble:
    """堆叠集成（二级学习器）"""
    
    def __init__(self,
                 base_models: List,
                 meta_model = None,
                 n_folds: int = 5,
                 use_probas: bool = True,
                 random_state: int = 42):
        """
        初始化堆叠集成
        
        参数:
            base_models: 基础模型列表
            meta_model: 元模型（二级学习器）
            n_folds: 用于生成训练元特征的折数
            use_probas: 是否使用概率作为特征
            random_state: 随机种子
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_probas = use_probas
        self.random_state = random_state
        
        # 从训练模块导入模型工厂（用于创建默认元模型）
        try:
            from src.models.training import INTJModelFactory
            self.model_factory = INTJModelFactory
        except ImportError:
            self.model_factory = None
        
        # 设置默认元模型
        if self.meta_model is None and self.model_factory:
            self.meta_model = self.model_factory.create_lightgbm({
                'n_estimators': 100,
                'learning_rate': 0.05,
                'random_state': random_state
            })
        
        # 存储训练后的模型
        self.base_models_trained_ = []
        self.meta_model_trained_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """训练堆叠集成模型"""
        self.n_classes_ = len(np.unique(y))
        
        # 1. 训练基础模型
        self.base_models_trained_ = []
        for model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.base_models_trained_.append(model_clone)
        
        # 2. 生成元特征
        X_meta = self._generate_meta_features(X, y, training=True)
        
        # 3. 训练元模型
        self.meta_model_trained_ = clone(self.meta_model)
        self.meta_model_trained_.fit(X_meta, y)
        
        return self
    
    def predict(self, X):
        """预测"""
        X_meta = self._generate_meta_features(X, training=False)
        return self.meta_model_trained_.predict(X_meta)
    
    def predict_proba(self, X):
        """预测概率"""
        X_meta = self._generate_meta_features(X, training=False)
        
        if hasattr(self.meta_model_trained_, 'predict_proba'):
            return self.meta_model_trained_.predict_proba(X_meta)
        else:
            raise ValueError("元模型不支持概率预测")
    
    def _generate_meta_features(self, X, y=None, training=True):
        """生成元特征"""
        if training:
            # 使用交叉验证生成元特征
            return self._generate_meta_features_cv(X, y)
        else:
            # 使用训练好的基础模型生成元特征
            return self._generate_meta_features_direct(X)
    
    def _generate_meta_features_cv(self, X, y):
        """使用交叉验证生成元特征（防止数据泄漏）"""
        from sklearn.model_selection import StratifiedKFold
        
        kf = StratifiedKFold(n_splits=self.n_folds, 
                            shuffle=True, 
                            random_state=self.random_state)
        
        n_samples = X.shape[0]
        
        if self.use_probas:
            # 使用概率作为特征
            n_meta_features = len(self.base_models) * self.n_classes_
            X_meta = np.zeros((n_samples, n_meta_features))
        else:
            # 使用预测类别作为特征（one-hot编码）
            n_meta_features = len(self.base_models) * self.n_classes_
            X_meta = np.zeros((n_samples, n_meta_features))
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # 在训练折上训练基础模型
            fold_models = []
            for model in self.base_models:
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                fold_models.append(model_clone)
            
            # 为验证折生成元特征
            meta_start_idx = 0
            for i, model in enumerate(fold_models):
                if self.use_probas and hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_val)
                    n_class_probas = probas.shape[1]
                    X_meta[val_idx, meta_start_idx:meta_start_idx + n_class_probas] = probas
                    meta_start_idx += n_class_probas
                else:
                    preds = model.predict(X_val)
                    # One-hot编码
                    for j, pred in enumerate(preds):
                        X_meta[val_idx[j], meta_start_idx + pred] = 1
                    meta_start_idx += self.n_classes_
        
        return X_meta
    
    def _generate_meta_features_direct(self, X):
        """直接使用训练好的基础模型生成元特征"""
        n_samples = X.shape[0]
        
        if self.use_probas:
            n_meta_features = len(self.base_models_trained_) * self.n_classes_
            X_meta = np.zeros((n_samples, n_meta_features))
            
            meta_start_idx = 0
            for i, model in enumerate(self.base_models_trained_):
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    n_class_probas = probas.shape[1]
                    X_meta[:, meta_start_idx:meta_start_idx + n_class_probas] = probas
                    meta_start_idx += n_class_probas
                else:
                    # 如果不支持概率，使用预测类别
                    preds = model.predict(X)
                    for j, pred in enumerate(preds):
                        X_meta[j, meta_start_idx + pred] = 1
                    meta_start_idx += self.n_classes_
        else:
            n_meta_features = len(self.base_models_trained_) * self.n_classes_
            X_meta = np.zeros((n_samples, n_meta_features))
            
            meta_start_idx = 0
            for i, model in enumerate(self.base_models_trained_):
                preds = model.predict(X)
                for j, pred in enumerate(preds):
                    X_meta[j, meta_start_idx + pred] = 1
                meta_start_idx += self.n_classes_
        
        return X_meta
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性（如果元模型支持）"""
        if (self.meta_model_trained_ is not None and 
            hasattr(self.meta_model_trained_, 'feature_importances_')):
            
            # 创建特征名称
            feature_names = []
            for i, model in enumerate(self.base_models_trained_):
                model_name = type(model).__name__
                if self.use_probas:
                    for class_idx in range(self.n_classes_):
                        feature_names.append(f"{model_name}_prob_class{class_idx}")
                else:
                    for class_idx in range(self.n_classes_):
                        feature_names.append(f"{model_name}_pred_class{class_idx}")
            
            importances = self.meta_model_trained_.feature_importances_
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            raise ValueError("元模型不支持特征重要性分析")

class INTJEnsembleOptimizer:
    """集成优化器 - 优化集成权重和配置"""
    
    def __init__(self,
                 ensemble_method: str = 'weighted',
                 optimization_metric: str = 'accuracy',
                 n_trials: int = 50):
        """
        初始化集成优化器
        
        参数:
            ensemble_method: 集成方法 ('weighted', 'stacking')
            optimization_metric: 优化指标
            n_trials: 优化试验次数
        """
        self.ensemble_method = ensemble_method
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
    
    def optimize(self, 
                 base_predictions: np.ndarray,
                 labels: np.ndarray,
                 base_model_names: Optional[List[str]] = None) -> Dict:
        """
        优化集成参数
        
        参数:
            base_predictions: 基础模型预测 (n_models, n_samples)
            labels: 真实标签
            base_model_names: 基础模型名称列表
        
        返回:
            优化结果字典
        """
        n_models = base_predictions.shape[0]
        
        if self.ensemble_method == 'weighted':
            return self._optimize_weighted_ensemble(base_predictions, labels, n_models)
        elif self.ensemble_method == 'stacking':
            return self._optimize_stacking_ensemble(base_predictions, labels, base_model_names)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
    
    def _optimize_weighted_ensemble(self, predictions, labels, n_models):
        """优化加权集成权重"""
        import optuna
        
        def objective(trial):
            # 生成权重
            weights = []
            for i in range(n_models):
                weights.append(trial.suggest_float(f'weight_{i}', 0, 1))
            
            # 归一化权重
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            
            # 加权预测
            weighted_preds = np.zeros_like(predictions[0], dtype=float)
            for i in range(n_models):
                weighted_preds += predictions[i] * weights[i]
            
            # 应用阈值（对于概率预测）
            final_preds = (weighted_preds > 0.5).astype(int)
            
            # 计算指标
            from sklearn.metrics import accuracy_score
            score = accuracy_score(labels, final_preds)
            
            # 记录试验
            self.optimization_history_.append({
                'weights': weights,
                'score': score
            })
            
            return score
        
        # 创建研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # 提取最佳结果
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # 提取最佳权重
        best_weights = []
        for i in range(n_models):
            best_weights.append(self.best_params_[f'weight_{i}'])
        
        # 归一化
        weight_sum = sum(best_weights)
        best_weights = [w / weight_sum for w in best_weights]
        
        return {
            'best_weights': best_weights,
            'best_score': self.best_score_,
            'optimization_history': self.optimization_history_,
            'study': study
        }
    
    def _optimize_stacking_ensemble(self, predictions, labels, model_names):
        """优化堆叠集成参数"""
        # 这里可以扩展为优化元模型参数
        # 当前版本返回简单结果
        return {
            'method': 'stacking',
            'note': 'Stacking optimization requires more sophisticated implementation',
            'recommendation': 'Use default LightGBM or XGBoost as meta-model'
        }
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (10, 6)):
        """绘制优化历史"""
        import matplotlib.pyplot as plt
        
        if not self.optimization_history_:
            raise ValueError("没有优化历史数据")
        
        scores = [entry['score'] for entry in self.optimization_history_]
        trials = range(1, len(scores) + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(trials, scores, 'b-', alpha=0.6, label='Score')
        ax.axhline(y=self.best_score_, color='r', linestyle='--', 
                  label=f'Best: {self.best_score_:.4f}')
        ax.fill_between(trials, np.min(scores), scores, alpha=0.1)
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        ax.set_title('Ensemble Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# 集成策略选择器
class INTJEnsembleStrategySelector:
    """集成策略选择器 - 根据数据特征选择最佳集成方法"""
    
    @staticmethod
    def select_strategy(X: pd.DataFrame,
                       y: pd.Series,
                       base_models: List,
                       cv_folds: int = 3) -> Dict:
        """
        选择最佳集成策略
        
        返回:
            策略推荐字典
        """
        from sklearn.model_selection import cross_val_score
        
        # 1. 评估基础模型性能
        base_scores = []
        for model in base_models:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            base_scores.append({
                'model': type(model).__name__,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })
        
        # 2. 计算模型多样性
        # 使用OOF预测计算多样性
        from sklearn.model_selection import StratifiedKFold
        
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_predictions = []
        
        for model in base_models:
            model_oof_preds = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                model_oof_preds[val_idx] = model_clone.predict(X_val)
            
            oof_predictions.append(model_oof_preds)
        
        oof_predictions = np.array(oof_predictions)
        
        # 计算多样性
        diversity_metrics = []
        n_models = len(base_models)
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(oof_predictions[i] != oof_predictions[j])
                diversity_metrics.append(disagreement)
        
        avg_diversity = np.mean(diversity_metrics) if diversity_metrics else 0
        
        # 3. 选择策略
        base_score_mean = np.mean([score['mean_score'] for score in base_scores])
        base_score_std = np.mean([score['std_score'] for score in base_scores])
        
        strategy_recommendation = {
            'base_model_performance': {
                'mean_accuracy': base_score_mean,
                'std_accuracy': base_score_std,
                'individual_scores': base_scores
            },
            'diversity_analysis': {
                'avg_disagreement': avg_diversity,
                'interpretation': '越高表示模型多样性越好'
            },
            'recommended_strategy': '',
            'reasoning': '',
            'expected_improvement': ''
        }
        
        # 决策逻辑
        if avg_diversity > 0.2:  # 高多样性
            if base_score_std < 0.02:  # 稳定性能
                strategy_recommendation['recommended_strategy'] = '加权平均集成'
                strategy_recommendation['reasoning'] = '模型多样性高且性能稳定，适合加权平均'
                strategy_recommendation['expected_improvement'] = '中等提升 (1-3%)'
            else:
                strategy_recommendation['recommended_strategy'] = '堆叠集成'
                strategy_recommendation['reasoning'] = '模型多样性高但性能不稳定，堆叠集成可以学习最佳组合'
                strategy_recommendation['expected_improvement'] = '较高提升 (3-5%)'
        else:  # 低多样性
            if base_score_mean > 0.8:  # 高性能
                strategy_recommendation['recommended_strategy'] = '简单投票'
                strategy_recommendation['reasoning'] = '模型性能高但多样性低，简单投票足够'
                strategy_recommendation['expected_improvement'] = '小幅提升 (0-1%)'
            else:
                strategy_recommendation['recommended_strategy'] = '模型选择 + 调优'
                strategy_recommendation['reasoning'] = '模型多样性低且性能一般，建议选择最佳模型并调优'
                strategy_recommendation['expected_improvement'] = '取决于调优效果'
        
        return strategy_recommendation

# 导出主要类
__all__ = [
    'INTJEnsembleBuilder',
    'INTJWeightedEnsemble',
    'INTJStackingEnsemble',
    'INTJEnsembleOptimizer',
    'INTJEnsembleStrategySelector'
]
