#!/bin/bash

# 颜色定义
RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
NC='[0m' # No Color

echo -e "${GREEN}🚀 30天Kaggle项目更新脚本${NC}"
echo "=========================================="

# 检查是否有未提交的更改
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}📝 发现未提交的更改...${NC}"
    git add .
    
    if [[ -n "$1" ]]; then
        commit_msg="$1"
    else
        commit_msg="Update: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    git commit -m "$commit_msg"
    echo -e "${GREEN}✅ 已提交更改: $commit_msg${NC}"
else
    echo -e "${GREEN}📦 没有需要提交的更改${NC}"
fi

# 拉取远程更新
echo -e "${YELLOW}⬇️  拉取远程更新...${NC}"
git pull origin main

# 推送本地更新
echo -e "${YELLOW}⬆️  推送本地更新...${NC}"
git push origin main

# 显示状态
echo -e "${YELLOW}📊 最终状态...${NC}"
echo "------------------------------------------"
git log --oneline -3
echo "------------------------------------------"

echo -e "${GREEN}🎉 更新完成！${NC}"
echo -e "${GREEN}🔗 仓库: https://github.com/BountyJackLee/30day-kaggle-plan${NC}"
