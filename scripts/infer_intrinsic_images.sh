#!/bin/bash

if [ $# -ne 1 ]; then
  echo "One arguments are required." 1>&2
  echo "sh demo_img.sh PATH_TO_YOUR_INPUT_FOLDER"
  exit 1
fi

python3 infer_image.py -i $1 -o ./demo/infer_image -m ./trained_models/model_1st.pth --gpu 0

cat <<__EOT__
FINISH!
__EOT__
exit 0