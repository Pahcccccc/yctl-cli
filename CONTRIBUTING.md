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

## 🐛 How to Open an Issue

If you find a bug or have a suggestion for a new feature:
1. Go to the [Issues](https://github.com/Youssef-Ai1001/yctl-cli/issues) page.
2. Check if a similar issue already exists.
3. Click **New Issue** and provide a clear description of the problem or suggestion.

---

## 🤝 How to Create a Pull Request (PR)

1. **Create a New Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes:**
   - Keep your code clean and followed by project standards.
   - Add tests for new features in the `tests/` directory.

3. **Run Tests:**
   ```bash
   pytest
   ```

4. **Commit and Push:**
   ```bash
   git add .
   git commit -m "feat: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Open a PR:**
   Submit your pull request on GitHub and explain what your changes do.

---

Thanks for helping us build a better tool! ❤️
