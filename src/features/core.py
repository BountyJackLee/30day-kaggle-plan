
"""
INTJ特征工程系统 - 稳健、可复用、防泄漏的特征工程框架
基于30天Kaggle竞赛经验总结的最佳实践
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class INTJFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    INTJ风格特征工程器
    核心原则：一致性、稳健性、可追溯性
    """
    
    def __init__(self, 
                 strategy: str = 'robust',
                 handle_categorical: str = 'frequency',
                 create_interactions: bool = True,
                 debug_mode: bool = False):
        """
        初始化特征工程器
        
        参数:
            strategy: 'minimal' | 'robust' | 'advanced'
            handle_categorical: 'frequency' | 'target' | 'onehot'
            create_interactions: 是否创建交互特征
            debug_mode: 调试模式，记录更多信息
        """
        self.strategy = strategy
        self.handle_categorical = handle_categorical
        self.create_interactions = create_interactions
        self.debug_mode = debug_mode
        
        # 存储训练集统计信息
        self.feature_stats_ = {}
        self.encoders_ = {}
        self.scalers_ = {}
        
        # 特征列表
        self.feature_names_ = []
        self.categorical_cols_ = []
        self.numerical_cols_ = []
        
        # 调试信息
        self.debug_log_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        拟合特征工程器
        
        参数:
            X: 训练数据
            y: 目标变量（可选，用于目标编码）
        """
        X_ = X.copy()
        
        # 1. 基础特征工程
        X_processed = self._create_basic_features(X_)
        
        # 2. 分析特征类型
        self._analyze_feature_types(X_processed)
        
        # 3. 处理类别特征
        if self.handle_categorical == 'frequency' and self.categorical_cols_:
            self._fit_frequency_encoding(X_processed)
        elif self.handle_categorical == 'target' and y is not None and self.categorical_cols_:
            self._fit_target_encoding(X_processed, y)
        
        # 4. 处理数值特征
        if self.numerical_cols_:
            self._fit_numerical_scaling(X_processed)
        
        # 记录特征列表
        self.feature_names_ = list(X_processed.columns)
        
        if self.debug_mode:
            self.debug_log_.append({
                'action': 'fit',
                'feature_count': len(self.feature_names_),
                'categorical_cols': self.categorical_cols_,
                'numerical_cols': self.numerical_cols_
            })
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据
        
        参数:
            X: 要转换的数据
        
        返回:
            转换后的数据
        """
        X_ = X.copy()
        
        # 1. 基础特征工程
        X_processed = self._create_basic_features(X_)
        
        # 2. 应用类别编码
        if self.handle_categorical == 'frequency' and self.categorical_cols_:
            X_processed = self._apply_frequency_encoding(X_processed)
        elif self.handle_categorical == 'target' and self.categorical_cols_:
            X_processed = self._apply_target_encoding(X_processed)
        
        # 3. 应用数值缩放
        if self.numerical_cols_:
            X_processed = self._apply_numerical_scaling(X_processed)
        
        # 4. 创建交互特征
        if self.create_interactions:
            X_processed = self._create_interaction_features(X_processed)
        
        # 5. 确保特征顺序一致
        X_processed = X_processed.reindex(columns=self.feature_names_, fill_value=0)
        
        if self.debug_mode:
            self.debug_log_.append({
                'action': 'transform',
                'input_shape': X.shape,
                'output_shape': X_processed.shape,
                'missing_features': set(self.feature_names_) - set(X_processed.columns)
            })
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X)
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础特征"""
        df_proc = df.copy()
        
        # 记录原始列
        original_cols = set(df_proc.columns)
        
        # 1. 乘客ID处理
        if 'PassengerId' in df_proc.columns:
            passenger_split = df_proc['PassengerId'].str.split('_', expand=True)
            if passenger_split.shape[1] == 2:
                df_proc['GroupId'] = passenger_split[0]
                df_proc['PersonId'] = passenger_split[1].astype(int)
                
                # 计算组大小
                group_counts = df_proc['GroupId'].value_counts()
                df_proc['GroupSize'] = df_proc['GroupId'].map(group_counts)
                df_proc['Alone'] = (df_proc['GroupSize'] == 1).astype(int)
        
        # 2. Cabin处理
        if 'Cabin' in df_proc.columns:
            cabin_split = df_proc['Cabin'].str.split('/', expand=True)
            if cabin_split.shape[1] == 3:
                df_proc['Deck'] = cabin_split[0]
                df_proc['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
                df_proc['Side'] = cabin_split[2]
                
                # Deck编码
                deck_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
                deck_mapping = {deck: i+1 for i, deck in enumerate(deck_order)}
                df_proc['Deck_encoded'] = df_proc['Deck'].map(deck_mapping).fillna(0)
        
        # 3. 姓名处理
        if 'Name' in df_proc.columns:
            df_proc['LastName'] = df_proc['Name'].str.split().str[-1]
            df_proc['NameLength'] = df_proc['Name'].str.len().fillna(0)
        
        # 4. 消费特征处理
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        available_expense = [col for col in expense_cols if col in df_proc.columns]
        
        if available_expense:
            # 总消费
            df_proc['TotalSpent'] = df_proc[available_expense].sum(axis=1)
            
            # 是否有消费
            df_proc['HasSpent'] = (df_proc['TotalSpent'] > 0).astype(int)
            
            # 消费种类数
            df_proc['ExpenseTypes'] = (df_proc[available_expense] > 0).sum(axis=1)
            
            # 对数变换（处理偏态）
            for col in available_expense:
                df_proc[f'{col}_log'] = np.log1p(df_proc[col] + 1)
            
            # 消费比例特征
            if 'TotalSpent' in df_proc.columns:
                for col in available_expense:
                    df_proc[f'{col}_Ratio'] = df_proc[col] / (df_proc['TotalSpent'] + 1)
        
        # 5. 年龄处理
        if 'Age' in df_proc.columns:
            df_proc['Age'] = df_proc['Age'].fillna(df_proc['Age'].median())
            
            # 年龄分组
            bins = [0, 12, 18, 30, 50, 100]
            labels = ['Child', 'Teen', 'Young', 'Adult', 'Senior']
            df_proc['AgeGroup'] = pd.cut(df_proc['Age'], bins=bins, labels=labels, include_lowest=True)
            
            # 是否为儿童
            df_proc['IsChild'] = (df_proc['Age'] < 13).astype(int)
        
        # 6. 布尔特征转换
        bool_cols = ['CryoSleep', 'VIP']
        for col in bool_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna(False).astype(int)
        
        # 记录新增特征
        new_cols = set(df_proc.columns) - original_cols
        if self.debug_mode and new_cols:
            self.debug_log_.append({
                'action': 'create_basic_features',
                'new_features': list(new_cols),
                'count': len(new_cols)
            })
        
        return df_proc
    
    def _analyze_feature_types(self, df: pd.DataFrame):
        """分析特征类型"""
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            if dtype in ['object', 'category', 'bool']:
                self.categorical_cols_.append(col)
            elif dtype.startswith('int') or dtype.startswith('float'):
                self.numerical_cols_.append(col)
    
    def _fit_frequency_encoding(self, df: pd.DataFrame):
        """拟合频率编码"""
        for col in self.categorical_cols_:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                self.feature_stats_[f'{col}_freq'] = freq.to_dict()
    
    def _fit_target_encoding(self, df: pd.DataFrame, y: pd.Series):
        """拟合目标编码（需要交叉验证防泄漏）"""
        # 这里实现安全的目标编码
        # 实际应用中应该使用交叉验证
        pass
    
    def _fit_numerical_scaling(self, df: pd.DataFrame):
        """拟合数值特征缩放"""
        for col in self.numerical_cols_:
            if col in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[[col]].fillna(0))
                self.scalers_[col] = scaler
    
    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用频率编码"""
        df_proc = df.copy()
        
        for col in self.categorical_cols_:
            if col in df_proc.columns and f'{col}_freq' in self.feature_stats_:
                freq_map = self.feature_stats_[f'{col}_freq']
                df_proc[f'{col}_freq'] = df_proc[col].map(freq_map).fillna(0)
        
        return df_proc
    
    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用目标编码"""
        # 需要实现安全的目标编码应用
        return df
    
    def _apply_numerical_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数值缩放"""
        df_proc = df.copy()
        
        for col in self.numerical_cols_:
            if col in df_proc.columns and col in self.scalers_:
                df_proc[col] = self.scalers_[col].transform(df_proc[[col]].fillna(0))
        
        return df_proc
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        if self.strategy == 'minimal':
            return df
        
        df_proc = df.copy()
        
        # 1. 年龄与消费的交互
        if 'Age' in df_proc.columns and 'TotalSpent' in df_proc.columns:
            df_proc['Age_Spending_Ratio'] = df_proc['TotalSpent'] / (df_proc['Age'] + 1)
        
        # 2. 冷冻睡眠与VIP的交互
        if 'CryoSleep' in df_proc.columns and 'VIP' in df_proc.columns:
            df_proc['CryoSleep_VIP'] = df_proc['CryoSleep'] * df_proc['VIP']
        
        # 3. 组大小与年龄的交互
        if 'GroupSize' in df_proc.columns and 'Age' in df_proc.columns:
            df_proc['Group_Age_Interaction'] = df_proc['GroupSize'] * df_proc['Age']
        
        # 4. 高级策略的特征
        if self.strategy == 'advanced':
            # 更多的交互特征
            if all(col in df_proc.columns for col in ['Deck_encoded', 'TotalSpent']):
                df_proc['Deck_Spending'] = df_proc['Deck_encoded'] * df_proc['TotalSpent']
            
            # 多项式特征
            if 'Age' in df_proc.columns:
                df_proc['Age_squared'] = df_proc['Age'] ** 2
                df_proc['Age_cubed'] = df_proc['Age'] ** 3
        
        return df_proc
    
    def get_feature_summary(self) -> pd.DataFrame:
        """获取特征摘要"""
        summary = []
        
        for feature in self.feature_names_:
            summary.append({
                'feature': feature,
                'type': 'categorical' if feature in self.categorical_cols_ else 'numerical',
                'origin': 'original' if feature in self.original_cols_ else 'engineered'
            })
        
        return pd.DataFrame(summary)
    
    def save_config(self, filepath: str):
        """保存配置"""
        import pickle
        
        config = {
            'strategy': self.strategy,
            'handle_categorical': self.handle_categorical,
            'feature_stats': self.feature_stats_,
            'scalers': self.scalers_,
            'feature_names': self.feature_names_,
            'categorical_cols': self.categorical_cols_,
            'numerical_cols': self.numerical_cols_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
    
    @classmethod
    def load_config(cls, filepath: str):
        """加载配置"""
        import pickle
        
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        instance = cls(
            strategy=config['strategy'],
            handle_categorical=config['handle_categorical']
        )
        
        instance.feature_stats_ = config['feature_stats']
        instance.scalers_ = config['scalers']
        instance.feature_names_ = config['feature_names']
        instance.categorical_cols_ = config['categorical_cols']
        instance.numerical_cols_ = config['numerical_cols']
        
        return instance

# 特征工程工具函数
class FeatureEngineeringUtils:
    """特征工程工具类"""
    
    @staticmethod
    def check_feature_consistency(train_features: pd.DataFrame, 
                                 test_features: pd.DataFrame,
                                 tolerance: float = 1e-5) -> Dict:
        """
        检查特征一致性
        
        返回:
            一致性报告字典
        """
        report = {
            'status': 'PASS',
            'issues': [],
            'metrics': {}
        }
        
        # 1. 检查列名
        train_cols = set(train_features.columns)
        test_cols = set(test_features.columns)
        
        if train_cols != test_cols:
            report['status'] = 'FAIL'
            report['issues'].append({
                'type': 'column_mismatch',
                'train_only': list(train_cols - test_cols),
                'test_only': list(test_cols - train_cols)
            })
        
        # 2. 检查数据类型
        common_cols = train_cols & test_cols
        for col in common_cols:
            if train_features[col].dtype != test_features[col].dtype:
                report['status'] = 'FAIL'
                report['issues'].append({
                    'type': 'dtype_mismatch',
                    'column': col,
                    'train_dtype': str(train_features[col].dtype),
                    'test_dtype': str(test_features[col].dtype)
                })
        
        # 3. 检查缺失值比例
        for col in common_cols:
            train_missing = train_features[col].isna().mean()
            test_missing = test_features[col].isna().mean()
            
            if abs(train_missing - test_missing) > 0.1:  # 10%差异阈值
                report['issues'].append({
                    'type': 'missing_value_discrepancy',
                    'column': col,
                    'train_missing': train_missing,
                    'test_missing': test_missing,
                    'difference': abs(train_missing - test_missing)
                })
        
        # 4. 检查数值范围
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in test_features.columns:
                train_min, train_max = train_features[col].min(), train_features[col].max()
                test_min, test_max = test_features[col].min(), test_features[col].max()
                
                if abs(train_min - test_min) > tolerance or abs(train_max - test_max) > tolerance:
                    report['issues'].append({
                        'type': 'value_range_discrepancy',
                        'column': col,
                        'train_range': [train_min, train_max],
                        'test_range': [test_min, test_max]
                    })
        
        # 计算指标
        report['metrics'] = {
            'total_features': len(common_cols),
            'matching_features': len(common_cols) - len(report['issues']),
            'issue_count': len(report['issues']),
            'consistency_score': (len(common_cols) - len(report['issues'])) / len(common_cols) if common_cols else 0
        }
        
        return report
    
    @staticmethod
    def create_feature_importance_report(model, 
                                        feature_names: List[str],
                                        top_n: int = 20) -> pd.DataFrame:
        """创建特征重要性报告"""
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_df['importance_pct'] = 100 * importance_df['importance'] / importance_df['importance'].sum()
        importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
        
        # 标记关键特征（累计达到特定阈值的特征）
        thresholds = [50, 80, 95]
        for threshold in thresholds:
            col_name = f'top_{threshold}pct'
            importance_df[col_name] = importance_df['cumulative_pct'] <= threshold
        
        return importance_df.head(top_n)
    
    @staticmethod
    def analyze_feature_correlation(df: pd.DataFrame, 
                                   threshold: float = 0.8) -> pd.DataFrame:
        """分析特征相关性"""
        corr_matrix = df.corr().abs()
        
        # 获取高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return pd.DataFrame(high_corr_pairs)

# 导出主要类
__all__ = [
    'INTJFeatureEngineer',
    'FeatureEngineeringUtils'
]
