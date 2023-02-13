#!/bin/bash

set -eu


echo "Copying flake8 in the root of the project..."
cat << 'EOF' >> .flake8
[flake8]
# For more information: https://flake8.pycqa.org/en/latest/user/error-codes.html
# This ignores the following errors/warnings, that black already fixes
# E501: line too long (82 > 79 characters)
# E731: do not assign a lambda expression, use a def
# E203: whitespace before :
# E231: missing whitespace after ','
# E266: too many leading # for block comment
# W503: line break before binary operator
# F403: from module import * used; unable to detect undefined names
# F401: `module` imported but unused
; ignore = E402,E302,E305,E501,E401,E731, W503, E231
ignore = E501, E731, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
max-line-length = 100
max-complexity = 18
[black]
max-line-length = 100
EOF


echo "Copying .pre-commit-config.yaml in the root of the project..."
cat << 'EOF' >> .pre-commit-config.yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
    # -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    -   id: black
-   repo: https://gitlab.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
    -   id: flake8
EOF


echo "Copying ..isort.cfg in the root of the project..."
cat << 'EOF' >> .isort.cfg
[settings]
src_paths = src
EOF

echo "Installing python packages..."
pip install black==22.3.0
pip install flake8==4.0.1
pip install pre-commit==2.18.1


echo "Installing pre-commit hooks.."
pre-commit install
