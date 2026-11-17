# Topic-TACO Signal

面向特朗普推文与资产收益事件研究的 Python 原型：清洗文本、提取主题与情绪特征，生成 TACO 信号，检测事件，并计算累计异常收益（CAR）及统计汇总。

当前代码包含固定研究时段与尚待修复的数据衔接问题，见下方“已知限制”。本仓库不附带原始推文或 WRDS 授权数据。

## 流程与代码导航

| 阶段 | 模块 | 输出 |
| --- | --- | --- |
| 文本特征与信号 | `taco/cli/run_inference.py`、`taco/nlp/`、`taco/signals/` | 推文特征 CSV |
| 峰值检测与事件去重 | `taco/cli/detect_events.py`、`taco/events/` | 原始及去重事件 CSV |
| 市场与资产收益、CAR | `taco/cli/compute_car.py`、`taco/finance/` | CAR CSV |
| 汇总、分行业统计与回归 | `taco/cli/`、`taco/stats/` | 统计表、图与回归结果 |

## 安装

```bash
git clone https://github.com/SkkuFintechLiu/Topic-TACO_Signal.git
cd Topic-TACO_Signal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 可用 `python -m venv .venv` 和 `.venv\Scripts\activate`。依赖尚未锁定版本；复现时请记录 Python、依赖及下载模型的实际版本。

首次推理会下载情绪与嵌入模型，需要网络连接与本地存储空间。Apple GPU 的可选兼容设置：`export PYTORCH_ENABLE_MPS_FALLBACK=1`。

## 输入数据

把推文 CSV 放在 `data/raw/tweets.csv`。`run_inference.py` 读取以下字段：

| 字段 | 内容 |
| --- | --- |
| `id` | 推文标识 |
| `text` | 原始文本 |
| `datetime` | 可解析的时间，程序按 UTC 处理 |
| `is_retweet` | `TRUE` / `FALSE`；转推会被排除 |
| `favorites`、`retweets` | 数值型互动计数 |
| `device` | 设备或来源标记 |

当前推理保留 UTC 时间 `2016-09-01 00:00:00` 至 `2021-03-01 00:00:00` 范围内的非转推记录。先检查字段、时间格式与缺失值，避免过滤后没有有效样本。

## 环境变量

使用 WRDS / CRSP 数据需要相应账户与数据权限，在运行环境中设置 `WRDS_USERNAME`、`WRDS_PASSWORD`。命令行入口未自动加载 `.env`，仅创建该文件不会生效。

`OPENAI_API_KEY` 是可选变量：设置后，政策主题分类器会尝试调用 OpenAI API，将推文文本发送到该服务；未设置或请求失败时回退到词典规则。选择模式会影响可复现性。当前 `policy_topic` 未写入特征输出，详见已知限制。

## 分阶段运行

以下命令在仓库根目录执行，路径均为项目相对路径。

```bash
python -m taco.cli.run_inference --tweets data/raw/tweets.csv --out data/processed/tweets_features.csv
python -m taco.cli.detect_events --features data/processed/tweets_features.csv --out data/processed/events_deduped.csv
```

核对中间文件后，再执行收益分析：

```bash
python -m taco.cli.compute_car --events data/processed/events_deduped.csv --assets NVDA,AAPL,AMZN,TSLA,PFE,UNH,GOOG,META,BTC-USD --out data/results/car_results.csv
python -m taco.cli.stats --car data/results/car_results.csv --out data/results/stats_summary.csv
python -m taco.cli.sector_stats --car data/results/car_results.csv --out data/results/sector_stats.csv --plots_dir data/results/plots
```

在修复下述数据源回退问题前，必须确认 WRDS 连接与各资产实际收益来源。上述命令展示入口与参数，不表示结果已经通过验证。

## 已知限制

- **数据源回退错误**：`compute_car.py` 在 WRDS 连接失败时，对任意资产均请求 `BTC-USD`，再标记为原资产名称。修复前不要使用该回退路径的结果做跨资产比较。
- **回归字段缺失**：`run_inference.py` 计算了 `policy_topic`，但输出列未包含它；`regression.py` 需要此列，直接衔接会失败。因此暂不将回归列为可直接运行的步骤。
- **缓存只比较源文件**：推理输出存在且 `source_hash` 相同时会跳过计算；调整 `alpha`、`beta`、模型或分类模式后，请使用新的输出路径。
- **时间窗口固定**：收益数据请求范围为 `2015-01-01` 至 `2021-04-01`；改变研究时期需要同步调整代码。
- **空结果需逐步检查**：无事件或无行情时可能输出空文件；进入下一阶段前核查记录数、字段、日期与资产覆盖。

本仓库提供探索性研究代码，未在此声明经验证的预测能力、投资收益或因果效应。复现记录应包含输入数据来源、样本筛选、依赖与模型版本、事件参数、收益来源和输出校验结果。
