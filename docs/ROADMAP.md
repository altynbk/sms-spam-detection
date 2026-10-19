# Completion plan

Finish and polish the existing educational SMS classification project. Completion
covers the documented local workflow, consistent research materials and repository
cleanup. No new features, datasets or model adaptation are planned. Existing
regression and reproducibility checks remain part of validation.

## Retained scope

Keep the classical classification pipeline, grouped evaluation, existing character
attacks, local demo, notebook and manuscript. The current default model, its
threshold and published experimental results remain frozen.

The [external evaluation](EXTERNAL_EVALUATION.md),
[modern-message adaptation](MODERN_ADAPTATION.md) and
[ordinary-SMS expansion](HAM_EXPANSION.md) remain supporting reports. Both
adaptation candidates failed their replacement criteria. Preserve those findings,
source provenance and limitations when simplifying the main documentation.

## Ordered checklist

- [x] Simplify the README around the existing study, results and local launch;
  replace open-ended research priorities with this completion plan.
- [ ] Verify the documented local workflow from environment setup to opening the
  app, checking a message and viewing saved results. Check behavior with and
  without local model files, and make the required preparation and error messages
  clear. Fix confirmed obstacles within the existing functionality.
- [ ] Align the README, model card, notebook and manuscript: distinguish the
  default model from experimental candidates, check reported metrics against
  saved tables, and keep detailed follow-up studies in their supporting reports.
- [ ] Remove confirmed unused code, duplicate instructions, unnecessary dependencies
  and broken references. Check callers and recorded provenance before removing
  anything. If a broken optional component cannot be repaired within the existing
  scope, remove it with its references and dependencies; retain published evidence.
- [ ] Complete final verification: run relevant tests and dependency checks,
  execute the notebook, review manuscript consistency and layout, check local
  links, and confirm all five GitHub checks pass. Synchronize `main` and remove
  merged temporary branches after confirming their changes are included.

Complete the items in order, using one focused pull request per correction and
merging only after required checks pass. Record completed verification in the
[validation report](VALIDATION.md). See [Contributing](../CONTRIBUTING.md) for
commands and branch rules.

## Definition of done

The documented workflows work, research outputs agree, and no known reproducible
failure blocks the declared use. The repository has a clear entry point, a clean
`main` branch and passing required checks. Classification errors and limitations
remain documented; software completion does not imply perfect SMS detection.

Once the checklist is complete, this cleanup is finished. Additional research or
features require a separate decision and do not block completion.
