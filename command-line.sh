echo "===== Browserbite External ====="
date
python3 main.py browserbite dt external repeated >> results/browserbite-dt-external.results.txt
python3 main.py browserbite randomforest external repeated >> results/browserbite-randomforest-external.results.txt
python3 main.py browserbite svm external repeated >> results/browserbite-svm-external.results.txt
python3 main.py browserbite nn external repeated >> results/browserbite-nn-external.results.txt
echo "===== Browserbite Internal ====="
python3 main.py browserbite dt internal repeated >> results/browserbite-dt-internal.results.txt
python3 main.py browserbite randomforest internal repeated >> results/browserbite-randomforest-internal.results.txt
python3 main.py browserbite svm internal repeated >> results/browserbite-svm-internal.results.txt
python3 main.py browserbite nn internal repeated >> results/browserbite-nn-internal.results.txt

echo "===== CrossCheck External ====="
date
python3 main.py crosscheck dt external repeated >> results/crosscheck-dt-external.results.txt
python3 main.py crosscheck randomforest external repeated >> results/crosscheck-randomforest-external.results.txt
python3 main.py crosscheck svm external repeated >> results/crosscheck-svm-external.results.txt
python3 main.py crosscheck nn external repeated >> results/crosscheck-nn-external.results.txt
echo "===== CrossCheck Internal ====="
python3 main.py crosscheck dt internal repeated >> results/crosscheck-dt-internal.results.txt
python3 main.py crosscheck randomforest internal repeated >> results/crosscheck-randomforest-internal.results.txt
python3 main.py crosscheck svm internal repeated >> results/crosscheck-svm-internal.results.txt
python3 main.py crosscheck nn internal repeated >> results/crosscheck-nn-internal.results.txt

echo "===== BrowserNinja 1 External ====="
date
python3 main.py browserninja1 dt external repeated >> results/browserninja1-dt-external.results.txt
python3 main.py browserninja1 randomforest external repeated >> results/browserninja1-randomforest-external.results.txt
python3 main.py browserninja1 svm external repeated >> results/browserninja1-svm-external.results.txt
python3 main.py browserninja1 nn external repeated >> results/browserninja1-nn-external.results.txt
echo "===== BrowserNinja 1 Internal ====="
python3 main.py browserninja1 dt internal repeated >> results/browserninja1-dt-internal.results.txt
python3 main.py browserninja1 randomforest internal repeated >> results/browserninja1-randomforest-internal.results.txt
python3 main.py browserninja1 svm internal repeated >> results/browserninja1-svm-internal.results.txt
python3 main.py browserninja1 nn internal repeated >> results/browserninja1-nn-internal.results.txt

echo "===== BrowserNinja 2 External ====="
date
python3 main.py browserninja2 dt external repeated >> results/browserninja2-dt-external.results.txt
python3 main.py browserninja2 randomforest external repeated >> results/browserninja2-randomforest-external.results.txt
python3 main.py browserninja2 svm external repeated >> results/browserninja2-svm-external.results.txt
python3 main.py browserninja2 nn external repeated >> results/browserninja2-nn-external.results.txt
echo "===== BrowserNinja 2 Internal ====="
python3 main.py browserninja2 dt internal repeated >> results/browserninja2-dt-internal.results.txt
python3 main.py browserninja2 randomforest internal repeated >> results/browserninja2-randomforest-internal.results.txt
python3 main.py browserninja2 svm internal repeated >> results/browserninja2-svm-internal.results.txt
python3 main.py browserninja2 nn internal repeated >> results/browserninja2-nn-internal.results.txt
