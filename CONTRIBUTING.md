# Contributing to yctl 🚀

Thank you for your interest in contributing to `yctl`! We welcome all contributions that help make this the best AI engineering CLI tool for Linux.

## 🛠️ Development Setup

To start developing, follow these steps:

1. **Fork the Repository:**
   Click the **Fork** button at the top of the repository page.

2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/yctl-cli.git
   cd yctl-cli
   ```

3. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install Development Dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Verify Setup:**
   Run the tests to ensure everything is working correctly:
   ```bash
   pytest
   ```

---

## 📏 Code Standards

We use the following tools to maintain code quality:

- **Black**: For automatic code formatting.
- **Flake8**: For linting and ensuring PEP 8 compliance.
- **Mypy**: For static type checking.

Please run these tools before submitting your PR:
```bash
black .
flake8 .
mypy yctl/
```

---

## 🐛 How to Open an Issue

If you find a bug or have a suggestion for a new feature:
1. Go to the [Issues](https://github.com/Youssef-Ai1001/yctl-cli/issues) page.
2. Check if a similar issue already exists.
3. Click **New Issue** and provide a clear description (steps to reproduce, expected behavior, system details).

---

## 🤝 How to Create a Pull Request (PR)

1. **Create a New Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes:**
   - Follow the project standards.
   - Add or update tests in the `tests/` directory.

3. **Commit Your Changes:**
   We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Example:
   ```bash
   git commit -m "feat: add support for HDF5 datasets"
   ```

4. **Run Tests:**
   ```bash
   pytest tests/ -v
   ```

5. **Push and Open a PR:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Submit your pull request on GitHub and explain what your changes do and link the relevant issue.

---

Thanks for helping us build a better tool for the AI community! ❤️
