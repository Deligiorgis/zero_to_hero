black:
	black --line-length=120 src tests

isort:
	isort --profile=black src tests

mypy:
	mypy --config-file .\mypy.ini src tests

pylint:
	pylint src tests

#pytest:
