#!/bin/bash

# Directory setup
mkdir -p src/tts/piper_bin
mkdir -p src/tts/piper_models

# 1. Pipeline binary is installed via pip (piper-tts)
# We only need to download models now.

echo "✅ Piper is installed via pip."

# 2. Download Voice Models
MODELS_DIR="src/tts/piper_models"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

download_model() {
    LANG=$1
    REGION=$2
    NAME=$3
    QUALITY=$4
    
    FILE="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx"
    JSON="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx.json"
    
    # URL Pattern: lang/lang_REGION/name/quality/filename
    # e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx
    
    echo "⬇️  Downloading Voice: $FILE"
    curl -L -o "$MODELS_DIR/$FILE" "$BASE_URL/$LANG/${LANG}_${REGION}/$NAME/$QUALITY/$FILE"
    curl -L -o "$MODELS_DIR/$JSON" "$BASE_URL/$LANG/${LANG}_${REGION}/$NAME/$QUALITY/$JSON"
}

download_model_optional() {
    LANG=$1
    REGION=$2
    NAME=$3
    QUALITY=$4

    FILE="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx"
    JSON="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx.json"
    URL_BASE="$BASE_URL/$LANG/${LANG}_${REGION}/$NAME/$QUALITY"

    echo "⬇️  Trying optional voice: $FILE"
    if curl -f -L -o "$MODELS_DIR/$FILE" "$URL_BASE/$FILE"; then
        curl -f -L -o "$MODELS_DIR/$JSON" "$URL_BASE/$JSON" || true
        echo "✅ Installed optional voice: $FILE"
        return 0
    fi
    rm -f "$MODELS_DIR/$FILE" "$MODELS_DIR/$JSON"
    echo "⚠️ Optional voice unavailable: $FILE"
    return 1
}

# English
download_model "en" "US" "lessac" "medium"

# Hindi (Rohan, medium)
echo "⬇️  Downloading Hindi Voice (Rohan, medium)..."
download_model "hi" "IN" "rohan" "medium"

# Telugu (Request: Maya)
download_model "te" "IN" "maya" "medium"

# Tamil (requested): prefer custom trained IITM voice if present in repo root.
CUSTOM_TA_MODEL="ta_IN-iitm-female-s1-medium.onnx"
CUSTOM_TA_META="ta_IN-iitm-female-s1-medium.onnx.json"
if [ -f "$CUSTOM_TA_MODEL" ]; then
    echo "📦 Copying custom Tamil Piper model: $CUSTOM_TA_MODEL"
    cp -f "$CUSTOM_TA_MODEL" "$MODELS_DIR/$CUSTOM_TA_MODEL"
    if [ -f "$CUSTOM_TA_META" ]; then
        cp -f "$CUSTOM_TA_META" "$MODELS_DIR/$CUSTOM_TA_META"
    fi
elif [ -f "../$CUSTOM_TA_MODEL" ]; then
    # Support running script from src/tts directory.
    echo "📦 Copying custom Tamil Piper model from parent: ../$CUSTOM_TA_MODEL"
    cp -f "../$CUSTOM_TA_MODEL" "$MODELS_DIR/$CUSTOM_TA_MODEL"
    if [ -f "../$CUSTOM_TA_META" ]; then
        cp -f "../$CUSTOM_TA_META" "$MODELS_DIR/$CUSTOM_TA_META"
    fi
else
    # Fallback attempts from public voices index.
    download_model_optional "ta" "IN" "kani" "medium" || \
    download_model_optional "ta" "IN" "ponni" "medium" || \
    echo "⚠️ Tamil Piper model not found from default source. Add ta_IN-iitm-female-s1-medium.onnx to repo root."
fi

echo "✅ Piper Setup Complete!"
ls -l $MODELS_DIR
