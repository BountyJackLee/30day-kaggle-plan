#!/bin/bash
# GitHub部署脚本
echo "🚀 开始部署到GitHub..."

# 初始化Git仓库
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit: 30-day Kaggle learning project"
    echo "✅ Git仓库已初始化"
    echo "请运行: git remote add origin <your-repo-url>"
    echo "然后运行: git push -u origin main"
else
    git add .
    git commit -m "Update: $(date +"%Y-%m-%d %H:%M")"
    git push
    echo "✅ 更新已提交到GitHub"
fi

echo "🎉 部署完成!"
