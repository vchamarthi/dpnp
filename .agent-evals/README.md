# Agent evaluation tasks for dpnp

Tasks for measuring AI coding agents on dpnp with [repoevals](https://github.com/vchamarthi/repoevals). Grading is
deterministic: hidden tests, mutation analysis and matching against known review findings. No language model is
used in CI.

## Tasks

| task | kind | source |
|---|---|---|
| `fix_arraycreation_mgrid_complex_step` | bug fix, hidden tests | #2971 |
| `fix_histograms_bincount_empty` | bug fix, hidden tests | #3018 |
| `fix_linalg_pinv_tolerance_dtype` | bug fix, hidden tests | #3090 |
| `fix_linalg_svd_hermitian_singular` | bug fix, hidden tests | #2954 |
| `fix_logic_compare_pyint_overflow` | bug fix, hidden tests | #3017 |
| `fix_manipulation_insert_negative_oob` | bug fix, hidden tests | #3041 |
| `fix_manipulation_repeat_nested_list` | bug fix, hidden tests | #3024 |
| `fix_math_interp_empty_x` | bug fix, hidden tests | #2985 |
| `fix_ndarray_view_usm_offset` | bug fix, hidden tests | #3037 |
| `fix_statistics_median_keepdims_shape` | bug fix, hidden tests | #3081 |
| `review_replay_pr3045` | review of a real PR, scored against the human review | #3045 |
| `review_seeded_polyval` | review with planted problems and decoys | written for this suite |
| `review_clean_ptp` | review of a correct change (false-alarm control) | written for this suite |
| `test_pad_normalize_dict_width` | test writing, graded by mutant kill rate | written for this suite |

All tasks are drafts until reviewed (`repoevals author approve <task>`).

## Environment

Every task starts from the 0.20.0 release. The released wheel (`.agent-evals/requirements.txt`) provides the
compiled modules, which `[environment].setup` in `repoevals.toml` copies into each workspace, so no SYCL compiler
is needed and tasks run on a CPU through the OpenCL CPU runtime. Fixes are limited to Python code for that reason.

## Running

```bash
pip install -r .agent-evals/requirements.txt intel-opencl-rt "repoevals @ git+https://github.com/vchamarthi/repoevals"
export OCL_ICD_FILENAMES=$(python -c "import sys; print(sys.prefix)")/lib/libintelocl.so ONEAPI_DEVICE_SELECTOR=opencl:cpu
repoevals validate --source .
repoevals check --repeats 2
repoevals run --include-drafts --agent claude-code --repeats 5     # needs an authenticated agent CLI
```

CI (`.github/workflows/repoevals.yml`) runs `validate` and `check` on every change to the tasks. Dispatch it with
`measure: true` to also run the scripted agents in `probes/` (reference solutions, a do-nothing agent and three
cheating agents), which check the pipeline end to end without a model.
