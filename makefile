# Use bash for functions and && chaining
SHELL := /bin/bash

# Default remote; change if you use something else
REMOTE ?= origin
BRANCH ?= main

.PHONY: patch minor major show-version tag-version

patch:
	bumpver update --patch --commit --tag-commit
	git push $(REMOTE) $(BRANCH)
	git push $(REMOTE) --tags

minor:
	bumpver update --minor --commit --tag-commit
	git push $(REMOTE) $(BRANCH)
	git push $(REMOTE) --tags

major:
	bumpver update --major --commit --tag-commit
	git push $(REMOTE) $(BRANCH)
	git push $(REMOTE) --tags

# Helpful: see what setuptools-scm thinks the version is
show-version:
	python -m setuptools_scm

# Helpful: show latest tag
tag-version:
	git describe --tags --abbrev=0
