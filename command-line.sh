echo "===== Browserbite ====="

python3 main.py browserbite dt external
python3 main.py browserbite svm external
python3 main.py browserbite randomforest external
python3 main.py browserbite dt internal
python3 main.py browserbite svm internal
python3 main.py browserbite randomforest internal

echo "===== CrossCheck ====="

python3 main.py crosscheck dt external
python3 main.py crosscheck svm external
python3 main.py crosscheck randomforest external
python3 main.py crosscheck dt internal
python3 main.py crosscheck svm internal
python3 main.py crosscheck randomforest internal

echo "===== BrowserNinja ====="

python3 main.py browserninja dt external
python3 main.py browserninja svm external
python3 main.py browserninja randomforest external
python3 main.py browserninja dt internal
python3 main.py browserninja svm internal
python3 main.py browserninja randomforest internal
