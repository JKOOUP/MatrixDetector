rm -rf ./data/res/
mkdir ./data/res/
python ./main/predict.py
python ./main/check_minuses.py
python ./main/inference.py
rm -rf ./data/cropped/
mkdir ./data/cropped/
