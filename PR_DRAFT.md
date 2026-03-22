# PR Title: Chore: Professionalize Repository with Community Standards and DevOps

## Description
This Pull Request enhances the repository to meet professional open-source standards. It introduces essential community health files, standardizes the development environment with DevOps tools, and improves documentation for better discoverability and citation.

## Changes

### 1. Community Standards 🤝
- **Added `CONTRIBUTING.md`**: Detailed guidelines for contributors.
- **Added `CODE_OF_CONDUCT.md`**: Contributor Covenant v1.4 to foster a welcoming environment.
- **Added `SECURITY.md`**: Policy for reporting vulnerabilities.
- **Added GitHub Templates**:
  - `bug_report.md` & `feature_request.md` for clearer issues.
  - `PULL_REQUEST_TEMPLATE.md` to standardize PRs.

### 2. DevOps & Automation ⚙️
- **Added `Makefile`**: Standardized commands (`make install`, `make lint`) for ease of use.
- **Added CI/CD**: GitHub Actions workflow (`.github/workflows/ci.yml`) to automatically install dependencies and lint code on push.
- **Added Dependabot**: Configuration (`.github/dependabot.yml`) to keep dependencies up to date.
- **Added Linting**: `.flake8` configuration for consistent code style.
- **Updated `.gitignore`**: Added comprehensive ignore patterns for Python and IDEs.

### 3. Documentation & Research 📄
- **Enhanced `README.md`**: Added status badges (License, Python, Issues) and a Community section.
- **Added `CITATION.cff`**: Standard citation file to allow researchers to easily cite "iGrow".
- **Added `CHANGELOG.md`**: Initialized changelog to track project history.

## Impact
These changes make the repository more accessible to new contributors, easier to maintain through automation, and ensures it follows best practices for research software.

## Type of Change
- [x] Chore (non-breaking change which facilitates development)
- [x] Documentation Update

## Verification
- Verified that all new files are present.
- Verified that `make install` and `make lint` commands work locally.
- Verified that GitHub Actions configuration is valid.
