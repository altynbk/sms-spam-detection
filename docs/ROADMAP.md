# Research priorities

The current project includes grouped evaluation of eleven classical variants,
validation-only thresholds, three character attacks, a template sensitivity study,
conditional group-bootstrap intervals, an executed notebook, a current manuscript
and an interactive local lab. Reproduction and application checks run in CI.
An [external positive-report evaluation](EXTERNAL_EVALUATION.md) is complete:
the frozen model detects 176 / 372 dated English reports (47.31% recall).
It exposes limited transfer, but the source has no ham and cannot measure FPR.

The next improvements should add evidence where the current study is weakest.

| Priority | Work | Completion criterion |
|---|---|---|
| 1 | Modern legitimate SMS and a new final holdout | Obtain licensed, independently labeled ham and new spam with trustworthy dates; separate related templates, sources and time periods. Measure FP/FN on both classes. Reserve an untouched final set before using the inspected external results to develop improvements. |
| 2 | Variation across training splits | Predefine repeated development splits and repeat fitting/search. Report dispersion while preserving a new independent final holdout. |
| 3 | Stronger attacks and tested defenses | Define perturbation budgets and meaning/readability checks. Keep attack families and examples used to develop a defense separate from final evaluation. |
| Conditional | Russian/Kazakh evaluation | Collect independently labeled data for each target language, including code-switching. Report per-language errors; translating the interface or corpus does not establish accuracy. |

The existing lab supports interactive use. Public hosting, a separate API, mobile
clients and more complex models should follow a concrete user need. They are not
required to complete the present research scope.

See the [model card](MODEL_CARD.md) for measured limits and the
[contribution guide](../CONTRIBUTING.md) for the change workflow.
