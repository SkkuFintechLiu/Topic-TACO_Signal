运行前设置环境变量 WRDS_USERNAME 与 WRDS_PASSWORD。
示例命令：
python -m taco.cli.run_inference --tweets /Users/liuhongchuan/Desktop/trump_tweets.csv --out data/processed/tweets_features.csv
python -m taco.cli.detect_events --features data/processed/tweets_features.csv --out data/processed/events_deduped.csv
python -m taco.cli.compute_car --events data/processed/events_deduped.csv --assets SPY --out data/results/car_results.csv
python -m taco.cli.stats --car data/results/car_results.csv --out data/results/stats_summary.csv
python -m taco.cli.compute_car --events data/processed/events_deduped.csv --assets NVDA,AAPL,AMZN,TSLA,PFE,UNH,GOOG,META,BTC-USD --out data/results/car_results.csv
python -m taco.cli.sector_stats --car data/results/car_results.csv --out data/results/sector_stats.csv --plots_dir data/results/plots
python -m taco.cli.regression --car data/results/car_results.csv --events data/processed/events_deduped.csv --features data/processed/tweets_features.csv --out data/results/regression_summary.txt
Apple GPU 加速（可选）：
export PYTORCH_ENABLE_MPS_FALLBACK=1
首次运行会自动下载所需的SentenceTransformer模型（all-MiniLM-L6-v2 / wangrongsheng/bertweet-sentiment-sbert）。
