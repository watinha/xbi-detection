echo "===== Browserbite External ====="
python3 main.py browserbite dt external >> results/browserbite-dt-external.results.txt
python3 main.py browserbite randomforest external >> results/browserbite-randomforest-external.results.txt
python3 main.py browserbite svm external >> results/browserbite-svm-external.results.txt
python3 main.py browserbite nn external >> results/browserbite-nn-external.results.txt
echo "===== Browserbite Internal ====="
python3 main.py browserbite dt internal >> results/browserbite-dt-internal.results.txt
python3 main.py browserbite randomforest internal >> results/browserbite-randomforest-internal.results.txt
python3 main.py browserbite svm internal >> results/browserbite-svm-internal.results.txt
python3 main.py browserbite nn internal >> results/browserbite-nn-internal.results.txt

echo "===== CrossCheck External ====="
python3 main.py crosscheck dt external >> results/crosscheck-dt-external.results.txt
python3 main.py crosscheck randomforest external >> results/crosscheck-randomforest-external.results.txt
python3 main.py crosscheck svm external >> results/crosscheck-svm-external.results.txt
python3 main.py crosscheck nn external >> results/crosscheck-nn-external.results.txt
echo "===== CrossCheck Internal ====="
python3 main.py crosscheck dt internal >> results/crosscheck-dt-internal.results.txt
python3 main.py crosscheck randomforest internal >> results/crosscheck-randomforest-internal.results.txt
python3 main.py crosscheck svm internal >> results/crosscheck-svm-internal.results.txt
python3 main.py crosscheck nn internal >> results/crosscheck-nn-internal.results.txt

echo "===== BrowserNinja 1 External ====="
python3 main.py browserninja1 dt external >> results/browserninja1-dt-external.results.txt
python3 main.py browserninja1 randomforest external >> results/browserninja1-randomforest-external.results.txt
python3 main.py browserninja1 svm external >> results/browserninja1-svm-external.results.txt
python3 main.py browserninja1 nn external >> results/browserninja1-nn-external.results.txt
echo "===== BrowserNinja 1 Internal ====="
python3 main.py browserninja1 dt internal >> results/browserninja1-dt-internal.results.txt
python3 main.py browserninja1 randomforest internal >> results/browserninja1-randomforest-internal.results.txt
python3 main.py browserninja1 svm internal >> results/browserninja1-svm-internal.results.txt
python3 main.py browserninja1 nn internal >> results/browserninja1-nn-internal.results.txt

echo "===== BrowserNinja 2 External ====="
python3 main.py browserninja2 dt external >> results/browserninja2-dt-external.results.txt
python3 main.py browserninja2 randomforest external >> results/browserninja2-randomforest-external.results.txt
python3 main.py browserninja2 svm external >> results/browserninja2-svm-external.results.txt
python3 main.py browserninja2 nn external >> results/browserninja2-nn-external.results.txt
echo "===== BrowserNinja 2 Internal ====="
python3 main.py browserninja2 dt internal >> results/browserninja2-dt-internal.results.txt
python3 main.py browserninja2 randomforest internal >> results/browserninja2-randomforest-internal.results.txt
python3 main.py browserninja2 svm internal >> results/browserninja2-svm-internal.results.txt
python3 main.py browserninja2 nn internal >> results/browserninja2-nn-internal.results.txt
