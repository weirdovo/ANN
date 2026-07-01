python main.py \
    --base_url "${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    --api_key "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}" \
    --model "${MODEL:-qwen2.5-14b-instruct}" \
    --dataset "${DATASET:-math500}" \
    --task "${TASK:-scaling}"