#!/bin/bash

# 썸네일 자동 변환 스크립트
BASE_DIR="/Users/jihyeokjeong/Documents/jihyeok/blog/jihyeok-blog"
POSTS_DIR="$BASE_DIR/content/posts"
CONTENT_DIR="$BASE_DIR/content"

# 이미지와 포스트 매칭
declare -A IMAGE_MAP
IMAGE_MAP["deque.png"]="deque.md"
IMAGE_MAP["dynamic.png"]="dynamic.md"
IMAGE_MAP["iris.jpeg"]="iris.md"
IMAGE_MAP["queue.png"]="queue.md"
IMAGE_MAP["stack.jpg"]="stack.md"
IMAGE_MAP["tensorflow.png"]="tensorflow.md"

# 각 이미지 처리
for IMAGE in "${!IMAGE_MAP[@]}"; do
    POST="${IMAGE_MAP[$IMAGE]}"
    POST_NAME="${POST%.md}"

    echo "Processing: $POST_NAME ($IMAGE)"

    # 1. 폴더 생성
    mkdir -p "$POSTS_DIR/$POST_NAME"

    # 2. .md 파일을 index.md로 이동
    if [ -f "$POSTS_DIR/$POST" ]; then
        mv "$POSTS_DIR/$POST" "$POSTS_DIR/$POST_NAME/index.md"
        echo "  ✓ Moved $POST to $POST_NAME/index.md"
    else
        echo "  ✗ $POST not found"
        continue
    fi

    # 3. 이미지 파일 이동
    if [ -f "$CONTENT_DIR/$IMAGE" ]; then
        mv "$CONTENT_DIR/$IMAGE" "$POSTS_DIR/$POST_NAME/$IMAGE"
        echo "  ✓ Moved $IMAGE to $POST_NAME/"
    else
        echo "  ✗ $IMAGE not found"
    fi

    echo "  ✓ Completed $POST_NAME"
    echo ""
done

echo "All done! Now adding image tags to front matter..."
