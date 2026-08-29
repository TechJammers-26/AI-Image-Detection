# Run `make` with PYTHONPATH=src already exported, or use `pip install -e .`
export PYTHONPATH := src
DATA ?= data/processed
MODEL ?= dummy
OUT ?= outputs

.PHONY: smoke test check train eval bench errors predict gallery all

smoke:   ; python scripts/smoke_test.py
test:    ; python tests/test_augmentations.py
check:   ; python scripts/check_data.py --data $(DATA) --leaks
gallery: ; python scripts/transform_gallery.py --out $(OUT)/gallery.png

train:
	python -m aigcdet.train --data $(DATA) --arch resnet50 --aug-policy heldout \
		--canonicalize --epochs 6 --out $(OUT)/run_heldout

ablation:
	for P in none continuous heldout spec; do \
		python -m aigcdet.train --data $(DATA) --arch resnet50 --aug-policy $$P \
			--canonicalize --epochs 6 --out $(OUT)/run_$$P ; \
	done

eval:
	python -m aigcdet.evaluate --data $(DATA) --split test --model $(MODEL) \
		--canonicalize --out $(OUT)/robustness

bench:
	python -m aigcdet.evaluate --benchmark data/benchmark --model $(MODEL) \
		--canonicalize --out $(OUT)/benchmark_ref

errors:
	python -m aigcdet.error_analysis --scores $(OUT)/robustness/scores.jsonl \
		--out $(OUT)/error_analysis

predict:
	python -m aigcdet.predict --input_dir $(DATA)/test --output $(OUT)/predictions.json \
		--model $(MODEL)

all: test eval errors predict
