PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTHON_BOOTSTRAP ?= $(shell command -v python3.12 2>/dev/null || command -v python3)

.PHONY: setup data validate-data cache-e5 cache-skillrouter experiments analysis tables test paper checklist supplement submission clean

setup:
	$(PYTHON_BOOTSTRAP) -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ is required"'
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install -r requirements-aaai-lock.txt

data:
	$(PYTHON) scripts/download_skillrouter_hard.py

validate-data:
	$(PYTHON) scripts/validate_multilingual_queries.py

cache-e5:
	$(PYTHON) scripts/cache_skillrouter_embeddings.py

cache-skillrouter:
	$(PYTHON) scripts/cache_skillrouter_domain_embeddings.py

experiments:
	$(PYTHON) scripts/evaluate_skillrouter_multilingual.py
	$(PYTHON) scripts/analyze_skillrouter_results.py

analysis:
	$(PYTHON) scripts/analyze_skillrouter_results.py

tables:
	$(PYTHON) scripts/generate_paper_results.py

test:
	$(PYTHON) -m unittest discover -s tests -v

paper: tables
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

checklist:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error reproducibility-checklist.tex

supplement: tables
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

submission: paper checklist supplement
	rm -f build/aaai-submission.zip build/code-data-supplement.zip
	rm -rf build/submission
	mkdir -p build/submission/source
	cp paper/main.pdf build/submission/anonymous-paper.pdf
	cp paper/reproducibility-checklist.pdf build/submission/
	cp paper/supplement.pdf build/submission/technical-supplement.pdf
	cp paper/main.tex paper/supplement.tex paper/reproducibility-checklist.tex \
		paper/references.bib paper/generated-results.tex \
		paper/table-public-results.tex \
		paper/aaai2027.sty paper/aaai2027.bst build/submission/source/
	cp -R paper/figures build/submission/source/
	cd build/submission && zip -qr ../aaai-submission.zip .
	rm -rf build/code-data
	mkdir -p build/code-data/scripts build/code-data/tests build/code-data/config \
		build/code-data/data build/code-data/results
	cp scripts/download_skillrouter_hard.py scripts/validate_multilingual_queries.py \
		scripts/cache_skillrouter_embeddings.py \
		scripts/cache_skillrouter_domain_embeddings.py \
		scripts/evaluate_skillrouter_multilingual.py \
		scripts/analyze_skillrouter_results.py scripts/generate_paper_results.py \
		scripts/audit_submission.py build/code-data/scripts/
	cp tests/test_skillrouter_multilingual.py tests/test_submission_artifacts.py \
		build/code-data/tests/
	cp config/dataset_revisions.json config/experiment_environment.json \
		config/model_revisions.json \
		build/code-data/config/
	cp data/skillrouter-hard.sha256 data/skillrouter_queries_multilingual.jsonl \
		data/skillrouter_queries_multilingual.stats.json data/translation-audit.md \
		build/code-data/data/
	cp results/skillrouter_multilingual.json results/skillrouter_multilingual.md \
		results/skillrouter_rankings.json results/error_analysis.json \
		results/error_analysis.md build/code-data/results/
	mkdir -p build/code-data/paper
	cp paper/main.tex paper/supplement.tex paper/reproducibility-checklist.tex \
		paper/references.bib paper/generated-results.tex \
		paper/table-public-results.tex \
		paper/aaai2027.sty paper/aaai2027.bst paper/AAAI27-SHA256SUMS \
		paper/main.pdf paper/supplement.pdf \
		paper/reproducibility-checklist.pdf build/code-data/paper/
	cp -R paper/figures build/code-data/paper/
	cp README.md ARTIFACTS.md LICENSE LICENSE-APACHE-2.0 NOTICE \
		DATA_LICENSE.md Makefile \
		requirements-aaai.txt requirements-aaai-lock.txt build/code-data/
	rm -rf build/code-data/data/external
	find build/code-data -type f -name '*.pyc' -delete
	find build/code-data -type d -name '__pycache__' -empty -delete
	cd build/code-data && zip -qr ../code-data-supplement.zip .
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/audit_submission.py

clean:
	cd paper && latexmk -C main.tex reproducibility-checklist.tex supplement.tex
	rm -rf build
