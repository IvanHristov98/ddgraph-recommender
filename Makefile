.PHONY: run
run:
	$(call run_in_venv,"cmd/run.py")


# 1 - script path
define run_in_venv
	@(export DATA_DIR="${PWD}/data" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
