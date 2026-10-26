# CST-MIL Python package

[Project overview and installation](../README.md)

This package contains the semantic model implementation, portable deployment support, topic-model baseline adapters and research command-line entry points. Use the root README to prepare the environment and run the demonstration. The released model uses joint supervision from five sources; results from the earlier weakly supervised CST-MIL implementation do not describe this model.

After installation, inspect the entry points from the repository root:

```powershell
python -m cst_mil.deployment --help
python -m cst_mil.research_followup.full_joint_comparison --help
python -m cst_mil.research_followup.semantic_patent_followup --help
cst-bench --help
```

Research workflows require source datasets, source audits and frozen split manifests. These prerequisites must be prepared separately. `--help` lists the available commands; it does not prepare the research inputs. See [Research code and reproducibility](../README.md#research-code-and-reproducibility).

The public package includes synthetic tests but no experimental predictions, evaluation metrics, timing records, run logs or research reports. Generated experimental artifacts must remain outside version control. Deployment weights are distributed separately under the [model terms](../MODEL_LICENSE.md).
