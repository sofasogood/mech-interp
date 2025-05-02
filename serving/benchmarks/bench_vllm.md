# vLLM Benchmark Results

## Test Configuration
- Threads: 4
- Connections: 32
- Duration: 30 seconds
- Endpoint: http://127.0.0.1:8000

## Performance Metrics
- Average Latency: 814.84ms
- Requests per second: 38.80
- Total requests: 1,165 in 30.03 seconds
- Data transfer rate: 27.70 KB/sec

## Latency Distribution
- 50th percentile: 753.58ms
- 75th percentile: 931.92ms
- 90th percentile: 1.38s
- 99th percentile: 1.69s

## Raw Results
```
Running 30s test @ http://127.0.0.1:8000
  4 threads and 32 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   814.84ms  352.26ms   1.79s    73.09%
    Req/Sec    11.61      7.11    40.00     71.45%
  Latency Distribution
     50%  753.58ms
     75%  931.92ms
     90%    1.38s 
     99%    1.69s 
  1165 requests in 30.03s, 831.85KB read
Requests/sec:     38.80
Transfer/sec:     27.70KB
```

## How to Add New Results
To add new benchmark results to this file, you can use:
```bash
# Run the benchmark and save to a temporary file
wrk -t4 -c32 -d30s --latency -s chat.lua http://127.0.0.1:8000 > temp_results.txt

# Append the results to this markdown file
echo -e "\n## New Benchmark Results\n\`\`\`" >> ~/fs1/mech_interp/serving/bench_vllm.md
cat temp_results.txt >> ~/fs1/mech_interp/serving/bench_vllm.md
echo -e "\`\`\`" >> ~/fs1/mech_interp/serving/bench_vllm.md
```
