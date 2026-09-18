# Contributing

This project is designed for rigorous, publication-oriented research on malaria dynamics in Togo. Contributions should therefore prioritize scientific correctness, methodological transparency, reproducibility, and operational relevance.

All project communication, documentation, comments, and code should be written in English unless a specific exception is formally agreed upon in the project discussion.

## Project scope

Contributions should support one or more of the following goals:

- improving the quality of climate and malaria data processing
- strengthening geospatial and temporal feature engineering
- improving model accuracy, robustness, and interpretability
- improving documentation for scientific review
- supporting operational health decision-making through deployable analysis

## Contribution workflow

1. Create a dedicated branch for each change.
2. Keep modifications focused and aligned with a single objective.
3. Document the scientific rationale for non-trivial updates.
4. Validate the relevant workflow before submitting a pull request.
5. Ensure the final change remains understandable to other researchers and contributors.

## Environment and setup

Before contributing, set up the environment in a clean Python virtual environment:

```bash
pip install uv
uv venv

# activate the virtual environment
# on Linux / macOS :
source .venv/bin/activate
# on Windows (PowerShell):
.venv\Scripts\activate
```

Use the project’s dependency management conventions consistently, and avoid introducing unrelated packages unless they are required for a well-documented scientific or engineering need.

## Code standards

### General expectations

- Write clean, readable, and maintainable Python code.
- Keep functions small and focused on one responsibility.
- Prefer descriptive, domain-aware variable names.
- Use clear English comments and docstrings when context is needed.
- Avoid hidden behavior, undocumented assumptions, or brittle one-off logic.

### Scientific code expectations

- Document data sources and any assumptions used in preprocessing.
- Preserve traceability between raw data, processed features, and analytical results.
- Do not silently drop data points without explanation.
- Label all transformations clearly so results are reproducible by others.
- Use consistent naming conventions for districts, dates, health variables, and climate variables.

## Data and modeling practices

### Data handling

- Store raw input files in the appropriate raw-data directory.
- Keep processed outputs separate from raw data.
- Document the source, date, and version of all external datasets.
- Avoid overwriting important raw files without explicit justification.

### Modeling practices

- State the target variable and prediction objective clearly.
- Report the assumptions of each model and evaluation approach.
- Prefer transparent validation pipelines over opaque training procedures.
- Use cross-validation or comparable validation strategies for model comparison.
- Ensure that performance metrics are interpreted in context, including dataset limitations and possible confounding factors.

## Testing and validation

All changes should be validated in the context of the relevant workflow. Minimum expectations include:

- confirming that relevant scripts still execute without errors
- checking that data pipelines still write valid outputs
- ensuring modified model code does not break evaluation or preprocessing
- documenting any known limitations or caveats in the relevant PR description

If a change affects a data pipeline or model, include a short summary of the validation outcome in the pull request.

## Documentation standards

- Documentation should be written in English.
- Explain the scientific purpose and methodological intent of the change.
- Include practical usage instructions when new functionality is added.
- Update any relevant script comments, README sections, or analysis notes impacted by the work.

## Pull request expectations

A pull request should include:

- a concise summary of the research or technical purpose
- the specific files changed
- a description of validation performed
- any caveats, assumptions, or limitations
- a statement of expected impact on the project workflow or scientific conclusions

## Issue reporting

When reporting a problem, include:

- a clear description of the issue
- the relevant script or module involved
- expected behavior versus observed behavior
- the environment used (Python version, OS, dependency versions if relevant)
- steps to reproduce the problem

## Review expectations

Contributors are expected to maintain a high standard of scientific and technical quality. Review comments may focus on:

- data quality and integrity
- reproducibility and documentation
- model validity and evaluation rigor
- code clarity and maintainability
- consistency with the project’s research objectives

## Final note

The project is intended to support a future scientific article and a practical public health decision-support application. As such, every contribution should reflect the principles of methodological rigor, transparency, and real-world usefulness.
