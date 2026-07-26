.DEFAULT_GOAL := help

.PHONY: help check

help:
	@echo "Crypto Agent DP Lab — archived research predecessor"
	@echo "  check  verify archive status, lineage, and repository hygiene"
	@echo "No setup/build/test target is supported for this archived snapshot."

check:
	@test -f repo.toml
	@test -f docs/historical-readme.md
	@test -f docs/historical-research-plan.md
	@grep -Fq 'Archived predecessor' README.md
	@grep -Fq 'hypothesis not validated' README.md
	@grep -Fq 'https://github.com/evanfollis/atlas' README.md
	@test -z "$$(git ls-files | grep -E '(__pycache__/|\.pyc$$)')"
	@echo "archive contract: PASS"
