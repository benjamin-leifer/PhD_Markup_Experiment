.PHONY: test

test:
	pre-commit run --all-files --show-diff-on-failure
	pytest
