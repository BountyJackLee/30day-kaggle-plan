# 30天Kaggle学习项目

![GitHub License](https://img.shields.io/github/license/BountyJackLee/30day-kaggle-plan)
![GitHub repo size](https://img.shields.io/github/repo-size/BountyJackLee/30day-kaggle-plan)
![GitHub last commit](https://img.shields.io/github/last-commit/BountyJackLee/30day-kaggle-plan)
![GitHub stars](https://img.shields.io/github/stars/BountyJackLee/30day-kaggle-plan?style=social)


## GitHub Topics

本项目的GitHub Topics:

- machine-learning
- kaggle
- python
- data-science
- lightgbm
- beginner-friendly
- tutorial

这些标签帮助项目被正确分类和发现。## 项目简介
这是一个完整的30天Kaggle竞赛学习项目，记录从机器学习入门到进阶的完整过程。

## 项目成果
- 最佳竞赛分数: 0.80897 (Spaceship Titanic)
- 技术文档: 50+份实验日志
- 代码库: 完整ML工具链
- 学习方法: 系统化学习体系

## 项目结构
```
30day-kaggle-plan/
├── src/           # 源代码
├── notebooks/     # 实验笔记本
├── docs/          # 文档
├── logs/          # 实验日志
└── tests/         # 测试
```

## 快速开始
```python
# 导入项目模块
import sys
sys.path.append('src')

from utils.helpers import ExperimentLogger
logger = ExperimentLogger()
```

## 许可证
MIT License

## ✨ 项目特点


### 🛠️ 技术栈
- **特征工程**: 防泄漏特征工程框架
- **模型训练**: LightGBM, XGBoost, CatBoost
- **集成学习**: 加权平均, Stacking, 投票法
- **工具库**: 实验跟踪, 内存优化, 特征分析


### 📈 学习成果
1. **竞赛成绩**: Spaceship Titanic 0.80897 (前15%)
2. **技术文档**: 50+份详细实验日志
3. **代码质量**: 模块化, 可复用的代码结构
4. **学习方法**: 系统化的学习与实验流程


### 🚀 快速开始
```bash
# 克隆项目
git clone https://github.com/BountyJackLee/30day-kaggle-plan.git
cd 30day-kaggle-plan
```

```python
# 导入项目模块
import sys
sys.path.append('src')

# 使用特征工程
from features.core import FeatureEngineering
fe = FeatureEngineering()

# 使用模型训练
from models.training import ModelFactory
model = ModelFactory.create_lightgbm()

# 使用实验日志
from utils.helpers import ExperimentLogger
logger = ExperimentLogger()
```


### 📁 项目结构
```
30day-kaggle-plan/
├── src/                    # 源代码
│   ├── features/          # 特征工程
│   ├── models/            # 模型训练与集成
│   └── utils/             # 工具函数
├── docs/                  # 文档
│   └── learnings/         # 学习笔记
├── notebooks/             # Jupyter笔记本
├── logs/                  # 实验日志
├── tests/                 # 测试用例
└── config/                # 配置文件
```


### 🤝 贡献指南
欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。


### 📄 许可证
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。


### 📞 联系
- GitHub Issues: [报告问题或提出建议](https://github.com/BountyJackLee/30day-kaggle-plan/issues)
- 学习笔记: [30天学习总结](docs/learnings/30day_summary.md)