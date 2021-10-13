black:
	black --line-length=120 src tests scripts

isort:
	isort --profile=black src tests scripts

mypy:
	mypy --config-file mypy.ini src tests scripts

pylint:
	pylint -j 2 src tests scripts

pytest:
	pytest --numprocesses 2 tests

code_check: isort black mypy pylint pytest
