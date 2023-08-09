ci: format test

all: format test lint

format:
	black --quiet --check .
test:
	pytest --doctest-modules -q
lint:
	ruff .
