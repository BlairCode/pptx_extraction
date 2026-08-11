# 更新现有 GitHub 仓库与发布 Release

目标仓库：<https://github.com/BlairCode/pptx_extraction>

以下命令以 Windows PowerShell 为例。项目已经存在，不要运行 `gh repo create`，也不要修改
`origin`。默认分支当前为 `master`。

## 1. 发布前检查

在项目根目录执行：

```powershell
git remote -v
python scripts/privacy_scan.py
ruff check .
ruff format --check .
mypy src/pptx_extraction
pytest
python -m build
python scripts/build_release.py
git diff --check
```

`git remote -v` 应显示：

```text
origin  https://github.com/BlairCode/pptx_extraction.git (fetch)
origin  https://github.com/BlairCode/pptx_extraction.git (push)
```

生成的发布文件：

```text
dist/pptx_extraction-2.0.0-py3-none-any.whl
dist/pptx_extraction-2.0.0.tar.gz
release/pptx_extraction-v2.0.0.zip
release/pptx_extraction-skill-v2.0.0.zip
```

`dist/`、`release/`、`work/`、`_legacy_local_backup/` 和真实 PPTX/音频均被 Git 忽略。不要使用
`git add -f` 强制加入这些本地内容。

## 2. 在现有仓库创建升级分支

当前工作区包含完整重构，可直接从当前提交点创建升级分支：

```powershell
git switch -c codex/pptx-extraction-v2
git add -A
git status --short
git diff --cached --check
git diff --cached --name-status
```

重点确认：

- 旧原型脚本、个人样例和运行输出显示为删除；
- `src/pptx_extraction/`、`tests/`、`docs/`、`agent-skill/` 显示为新增；
- 没有 `.env`、PPTX、音频、日志、`work/`、`dist/` 或 `release/` 文件被暂存。

确认后提交并推送：

```powershell
git commit -m "refactor: rebuild pptx_extraction for structured workflows"
git push -u origin codex/pptx-extraction-v2
```

## 3. 创建 Pull Request

使用 GitHub CLI：

```powershell
gh auth status
gh pr create `
  --repo BlairCode/pptx_extraction `
  --base master `
  --head codex/pptx-extraction-v2 `
  --title "refactor: pptx_extraction 2.0" `
  --body "Rebuild extraction around structured JSON/Markdown output, safe OOXML validation, batch processing, optional OCR/API, tests, documentation, and an Agent Skill."
```

等待 GitHub Actions 全部通过，再在 GitHub 页面合并 PR。不要在 CI 失败时直接给 `master` 打标签。

## 4. 合并后发布 v2.0.0

回到本地并同步默认分支：

```powershell
git switch master
git pull --ff-only origin master
git tag -a v2.0.0 -m "pptx_extraction 2.0.0"
git push origin v2.0.0
```

创建 Release，并分别上传项目、Python 包和 Skill：

```powershell
gh release create v2.0.0 `
  "release/pptx_extraction-v2.0.0.zip" `
  "release/pptx_extraction-skill-v2.0.0.zip" `
  "dist/pptx_extraction-2.0.0-py3-none-any.whl" `
  "dist/pptx_extraction-2.0.0.tar.gz" `
  --repo BlairCode/pptx_extraction `
  --title "pptx_extraction 2.0.0" `
  --notes-file CHANGELOG.md
```

这样 Skill 和项目仍是两个独立 Release 文件，但共同发布在既有仓库的同一个版本页面中。

## 5. Release 发布后验证

下载 wheel 到新的临时目录并测试：

```powershell
python -m venv release-check
.\release-check\Scripts\Activate.ps1
python -m pip install "PATH_TO_WHEEL\pptx_extraction-2.0.0-py3-none-any.whl"
pptx-extraction --version
```

解压 Skill ZIP，确认根目录为 `pptx-extraction/`，并包含：

```text
SKILL.md
agents/openai.yaml
scripts/extract.py
references/schema.md
references/troubleshooting.md
```

## 关于旧 Git 历史中的隐私内容

现有仓库历史过去曾包含个人路径、样例 PPTX、音频和旧 README 联系信息。普通更新只会确保新提交与
Release 不再包含它们，不会删除旧提交中的对象。如果仓库已经公开，这些内容可能已被克隆或缓存。

若确实需要从全部历史中清除，先备份并评估影响，再使用 `git filter-repo` 重写历史；这会改变所有
提交哈希并要求强制推送和通知协作者，不应与普通 v2.0 更新混在同一次操作中。
