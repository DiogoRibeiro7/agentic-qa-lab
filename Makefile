.PHONY: install lint typecheck test format precommit docs run

install:
	poetry install --with dev

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest -q

format:
	poetry run ruff format .

precommit:
	poetry run pre-commit run --all-files

docs:
	poetry run mkdocs build --strict

run:
	poetry run python -m agentic_qa_lab.cli
