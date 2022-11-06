# README
An example of loading and displaying anomaly images
```shell
python main.py
```

Anomaly Isolation Method
```shell
python isolate_anomalies.py --inject_portion 0.1 --trigger_type "gridTrigger"
```

After running, 
- the isolation precision in each iteration will be logged.
- training loss for anomaly data pointss and clean data points will be recorded and saved, respectively. The saved files are 'losses_bad.npy' and 'losses_clean.npy'.

