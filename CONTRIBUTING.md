# Contributing to VoltaSplat

We welcome contributions from the community! VoltaSplat is a high-performance 3D Gaussian Splatting engine built on PyTorch and custom CUDA kernels. Please review the following guidelines before contributing.

## Contribution Policy

1. **Bug Reports & Feature Requests**
   - Please use the GitHub Issue Tracker to report bugs or request features.
   - Provide a clear, detailed description of the issue, including steps to reproduce, expected behavior, and actual behavior.
   - Include your environment details (OS, PyTorch version, CUDA version, GPU model).

2. **Branching Strategy**
   - We follow the standard feature-branch workflow.
   - All feature branches must branch off from `develop`.
   - Never commit directly to `main`. `main` is strictly for production-ready releases.
   - Format branch names as `feature/<short-description>` or `bugfix/<short-description>`.

3. **Code Style & Standards**
   - **C++ / CUDA**: We strictly adhere to C++20 standards. Ensure no deprecated features are used. Comment complex kernel logic (like shared memory synchronization or atomic operations).
   - **Python**: Follow PEP 8 guidelines. Write clean, modular, and well-documented Python code. Avoid AI-sounding language or emojis in documentation.

4. **Testing Requirements**
   - All new features and bug fixes must include corresponding tests (e.g., using `pytest`).
   - Run the existing test suite (`pytest tests/`) to ensure no regressions occur.
   - Memory boundaries and CUDA synchronization logic should be rigorously verified.

5. **Pull Requests**
   - Ensure your PR merges into `develop`.
   - Keep PRs focused on a single issue or feature.
   - Provide a comprehensive description of the changes.
   - Code reviews are required for all PRs before they can be merged into `develop`.

6. **Licensing**
   - By contributing, you agree that your contributions will be licensed under the Elastic License 2.0.

Thank you for helping us improve VoltaSplat!
