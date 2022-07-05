.PHONY: test-ddgraph
test-ddgraph:
	$(call run_in_venv,"cmd/test_ddgraph.py")

.PHONY: test-trans
test-trans:
	$(call run_in_venv,"cmd/test_trans.py")


# 1 - script path
define run_in_venv
	@(export DATA_DIR="${PWD}/data" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
