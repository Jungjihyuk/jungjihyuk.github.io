#!/bin/bash

# Hugo 빌드 및 배포 준비 스크립트

echo "======================================"
echo "   Hugo 빌드 스크립트"
echo "======================================"
echo ""

# 1. 이전 빌드 삭제
echo "[1/4] 이전 빌드 정리..."
rm -rf public/
echo "  ✓ 완료"

# 2. Hugo 빌드
echo "[2/4] Hugo 빌드 시작..."
hugo --minify

if [ $? -ne 0 ]; then
    echo "❌ 빌드 실패!"
    exit 1
fi
echo "  ✓ 완료"

# 3. 빌드 통계
echo "[3/4] 빌드 통계..."
if [ -d "public" ]; then
    file_count=$(find public -type f | wc -l | xargs)
    total_size=$(du -sh public 2>/dev/null | cut -f1)
    echo "  생성된 파일 수: $file_count"
    echo "  총 크기: $total_size"
else
    echo "  ⚠️  public 디렉토리를 찾을 수 없습니다."
fi

# 4. 링크 체크 (선택사항)
echo "[4/4] 검증..."
if [ -d "public" ]; then
    # 간단한 HTML 파일 검증
    html_count=$(find public -name "*.html" | wc -l | xargs)
    echo "  HTML 파일: $html_count"

    # index.html 존재 확인
    if [ -f "public/index.html" ]; then
        echo "  ✓ index.html 존재"
    else
        echo "  ⚠️  index.html 없음"
    fi

    # RSS 피드 확인
    if [ -f "public/feed.xml" ] || [ -f "public/index.xml" ]; then
        echo "  ✓ RSS 피드 존재"
    else
        echo "  ⚠️  RSS 피드 없음"
    fi
fi

echo ""
echo "======================================"
echo "✅ 빌드 완료!"
echo "======================================"
echo "📁 출력 디렉토리: public/"
echo ""
echo "로컬 확인:"
echo "  hugo server"
echo ""
echo "배포 (GitHub Pages):"
echo "  git add ."
echo "  git commit -m 'Deploy'"
echo "  git push origin main"
echo ""
