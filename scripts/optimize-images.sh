#!/bin/bash

# 이미지 최적화 스크립트 (ImageMagick 필요)

echo "======================================"
echo "   이미지 최적화 스크립트"
echo "======================================"
echo ""

# ImageMagick 설치 확인
if ! command -v convert &> /dev/null; then
    echo "❌ ImageMagick이 설치되어 있지 않습니다."
    echo "설치: brew install imagemagick"
    exit 1
fi

# static/images/ 디렉토리 확인
if [ ! -d "static/images" ]; then
    echo "static/images/ 디렉토리가 없습니다. 생성합니다..."
    mkdir -p static/images
fi

# 이미지 파일 찾기
image_count=$(find static/images -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l | xargs)

if [ "$image_count" -eq 0 ]; then
    echo "최적화할 이미지가 없습니다."
    exit 0
fi

echo "총 $image_count 개의 이미지를 찾았습니다."
echo ""

# 이미지 최적화
find static/images -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | while read img; do
    echo "최적화 중: $img"

    # 원본 백업
    cp "$img" "${img}.bak"

    # 이미지 크기 조정 (최대 1920px 폭) 및 품질 85%로 압축
    convert "$img" -resize "1920x1920>" -quality 85 "$img"

    # 파일 크기 비교
    if [ -f "${img}.bak" ]; then
        if command -v stat &> /dev/null; then
            original_size=$(stat -f%z "${img}.bak" 2>/dev/null || stat -c%s "${img}.bak" 2>/dev/null)
            new_size=$(stat -f%z "$img" 2>/dev/null || stat -c%s "$img" 2>/dev/null)

            if [ -n "$original_size" ] && [ -n "$new_size" ]; then
                saved=$((original_size - new_size))
                echo "  원본: $(numfmt --to=iec $original_size 2>/dev/null || echo $original_size bytes)"
                echo "  최적화: $(numfmt --to=iec $new_size 2>/dev/null || echo $new_size bytes)"
                if [ $saved -gt 0 ]; then
                    echo "  절감: $(numfmt --to=iec $saved 2>/dev/null || echo $saved bytes)"
                fi
            fi
        fi

        # 백업 삭제
        rm "${img}.bak"
    fi

    echo ""
done

echo "======================================"
echo "이미지 최적화 완료!"
echo "======================================"
