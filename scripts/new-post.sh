#!/bin/bash

# 새 포스트 생성 자동화 스크립트

echo "======================================"
echo "   Hugo 새 포스트 생성 도구"
echo "======================================"
echo ""

# 포스트 제목 입력
read -p "포스트 제목을 입력하세요: " title

if [ -z "$title" ]; then
    echo "제목이 비어있습니다. 종료합니다."
    exit 1
fi

# 슬러그 생성 (공백을 하이픈으로 변경, 한글 제거)
slug=$(echo "$title" | sed 's/ /-/g' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]//g')

# 현재 날짜
date=$(date +"%Y-%m-%d")

# 카테고리 입력
read -p "카테고리를 입력하세요 (쉼표로 구분, 예: Python, AI): " categories_input

# 태그 입력
read -p "태그를 입력하세요 (쉼표로 구분, 예: tutorial, beginner): " tags_input

# 설명 입력
read -p "포스트 설명을 입력하세요: " description

# 카테고리 배열 변환
category_yaml="categories:"
if [ -n "$categories_input" ]; then
    IFS=',' read -ra categories <<< "$categories_input"
    for cat in "${categories[@]}"; do
        cat=$(echo "$cat" | xargs)  # trim whitespace
        category_yaml="$category_yaml\n  - $cat"
    done
else
    category_yaml="categories: []"
fi

# 태그 배열 변환
tag_yaml="tags:"
if [ -n "$tags_input" ]; then
    IFS=',' read -ra tags <<< "$tags_input"
    for tag in "${tags[@]}"; do
        tag=$(echo "$tag" | xargs)  # trim whitespace
        tag_yaml="$tag_yaml\n  - $tag"
    done
else
    tag_yaml="tags: []"
fi

# 파일명 생성
filename="content/posts/${slug}.md"

# Frontmatter 생성
cat > "$filename" << EOF
---
title: "$title"
date: $date
draft: false
description: "$description"
$(echo -e "$category_yaml")
$(echo -e "$tag_yaml")
slug: "$slug"
---

## 개요

포스트 내용을 여기에 작성하세요.

## 본문

### 섹션 1

내용...

### 섹션 2

내용...

## 결론

마무리 내용...
EOF

echo ""
echo "✅ 포스트가 생성되었습니다!"
echo "📁 파일 위치: $filename"
echo "🌐 URL: /$date/$slug/"
echo ""
echo "에디터로 열기:"
echo "  code $filename"
echo "  vi $filename"
echo ""

# VSCode로 자동 열기 (선택사항)
read -p "VSCode로 열까요? (y/n): " open_vscode
if [ "$open_vscode" = "y" ] || [ "$open_vscode" = "Y" ]; then
    if command -v code &> /dev/null; then
        code "$filename"
    else
        echo "VSCode를 찾을 수 없습니다."
    fi
fi
