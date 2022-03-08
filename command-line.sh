echo "===== Browserbite External ====="
date
python3 main.py browserbite external repeated >> results/browserbite-external.results.txt
echo "===== Browserbite Internal ====="
python3 main.py browserbite internal repeated >> results/browserbite-internal.results.txt

echo "===== CrossCheck External ====="
date
python3 main.py crosscheck external repeated >> results/crosscheck-external.results.txt
echo "===== CrossCheck Internal ====="
python3 main.py crosscheck internal repeated >> results/crosscheck-internal.results.txt

echo "===== BrowserNinja 1 External ====="
date
python3 main.py browserninja1 external repeated >> results/browserninja1-external.results.txt
echo "===== BrowserNinja 1 Internal ====="
python3 main.py browserninja1 internal repeated >> results/browserninja1-internal.results.txt

echo "===== BrowserNinja 2 External ====="
date
python3 main.py browserninja2 external repeated >> results/browserninja2-external.results.txt
echo "===== BrowserNinja 2 Internal ====="
python3 main.py browserninja2 internal repeated >> results/browserninja2-internal.results.txt
