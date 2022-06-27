.PHONY: test-ddgraph
test-ddgraph:
	$(call run_in_venv,"cmd/test_ddgraph.py")

.PHONY: test-transe
test-transe:
	$(call run_in_venv,"cmd/test_transe.py")


# 1 - script path
define run_in_venv
	@(export DATA_DIR="${PWD}/data" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
