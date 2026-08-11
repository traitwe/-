# MCM 2026 · B题：秦皇岛旅游客流与资源协同

本仓库保留论文复现所需的数据、代码、结果与正式稿。问题一至问题四通过同一条主流程顺序衔接；日级游客规模均为模型估计或情景值，不能表述为官方逐日实测值。

## 快速复现

在项目根目录运行：

```powershell
$py = 'C:\Users\lyy20\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts\run_paper_mainline.py
```

主流程依次重建问题一至问题四的正式输出，并刷新已核验快照。运行后检查 `outputs/delivery_status.json`：仅当 `delivery_mode` 为 `full_model` 时，当前输出可作为完整交付；若中途失败，程序仅降级交付 `outputs/verified_artifacts/` 中上一轮核验快照。

## 目录说明

| 路径 | 用途 |
|---|---|
| `题目/` | B题题面、竞赛格式规范与工具使用规定。 |
| `data/raw/` | 原始资料与来源台账；不由主流程覆盖。 |
| `data/clean/` | 清洗后的中间数据。 |
| `data/model_input/` | 四问建模输入、参数表及字段说明。 |
| `src/` | 可复用的数据处理、特征、模型、分析与交付状态模块。 |
| `scripts/` | 复现入口及各问题正式构建脚本。 |
| `tests/` | 主链模块的回归测试。 |
| `outputs/question1_analysis` 至 `question4_analysis/` | 当前正式模型输出。 |
| `outputs/verified_artifacts/` | 上一轮完整核验后的只读降级快照。 |
| `outputs/figures/` | 论文候选图。 |
| `论文相关模板/` | 国赛 LaTeX 模板与最终 PDF/TeX。 |
| `论文写作/` | 封稿检查、数据边界、复现和交付说明。 |
| `文献资料包/`、`国赛参考/` | 本地参考文献与优秀论文材料。 |

## 正式交付入口

- 论文源文件：`论文相关模板/国赛Latex模板/国赛模板/B题完整论文_国赛模板.tex`
- 论文 PDF：`论文相关模板/国赛Latex模板/国赛模板/B题完整论文_国赛模板.pdf`
- 交付状态：`outputs/delivery_status.json`
- 问题三 VOT 参数：`data/model_input/q3_vot_scenarios.csv`

## 边界与版本管理

- 不提交 Python 缓存、LaTeX 编译中间文件或本地临时目录，规则见 `.gitignore`。
- `outputs/verified_artifacts/` 是自动降级路径依赖的核验快照，不应手动删除。
- `国赛参考/` 和 `文献资料包/` 体积较大；如远程仓库只需复现主链，可在提交前另行决定是否纳入 Git LFS 或排除。
