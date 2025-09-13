REFERENCE.md: generate_docs.py
	python generate_docs.py matthewplotlib/*.py > REFERENCE.md


copy:
	tail -n +1 pyproject.toml README.md matthewplotlib/*.py examples/*.py | pbcopy


.PHONY: copy
