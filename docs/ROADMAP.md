# Research priorities

The current project includes grouped evaluation of eleven classical variants,
validation-only thresholds, three character attacks, a template sensitivity study,
conditional group-bootstrap intervals, an executed notebook, a current manuscript
and an interactive local lab. Reproduction and application checks run in CI.

The next improvements should add evidence where the current study is weakest.

| Priority | Work | Completion criterion |
|---|---|---|
| 1 | Independent modern English SMS benchmark | Obtain licensed, labeled ham and spam; separate related templates, sources and time periods. Evaluate the frozen current model and report FP/FN before considering retraining. |
| 2 | Variation across training splits | Predefine repeated development splits and repeat fitting/search. Report dispersion while preserving a new independent final holdout. |
| 3 | Stronger attacks and tested defenses | Define perturbation budgets and meaning/readability checks. Keep attack families and examples used to develop a defense separate from final evaluation. |
| Conditional | Russian/Kazakh evaluation | Collect independently labeled data for each target language, including code-switching. Report per-language errors; translating the interface or corpus does not establish accuracy. |

The existing lab supports interactive use. Public hosting, a separate API, mobile
clients and more complex models should follow a concrete user need. They are not
required to complete the present research scope.

See the [model card](MODEL_CARD.md) for measured limits and the
[contribution guide](../CONTRIBUTING.md) for the change workflow.
