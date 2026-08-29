# Resume the ACP Lung Cancer Project

Open this file after restarting the computer. The complete command history and
outputs are in `setup_steps_1_to_10.log`.

## Start the environment

Open **Anaconda Prompt**, then run these commands one at a time:

```bat
E:
cd E:\postdoc-work\ist-project
conda activate acp_esm2
python --version
```

Expected Python version:

```text
Python 3.11.x
```

## Confirm the current checkpoint

The completed checkpoint is Step 18. The latest feature file is:

```text
E:\postdoc-work\ist-project\derived\traditional_features_step18.csv
```

Verify it with:

```bat
python -c "import pandas as pd; p=r'derived\traditional_features_step18.csv'; df=pd.read_csv(p); print('Rows:', len(df)); print('Columns:', len(df.columns)); print('RESUME CHECK OK')"
```

Expected output:

```text
Rows: 901
Columns: 31
RESUME CHECK OK
```

## Continue the project

Do not rerun all earlier steps automatically. Read the next instruction, then
run only the next requested script or command. The feature script currently is:

```bat
python scripts\03_traditional_features.py
```

That script rebuilds the Step 16, Step 17, and Step 18 derived files. It does
not modify the original CSV in `data`.

## Useful verification commands

```bat
dir data
dir derived
dir scripts
```

The original dataset is:

```text
E:\postdoc-work\ist-project\data\ACPs_Lung_cancer.csv
```

The shared history log is:

```text
E:\postdoc-work\ist-project\setup_steps_1_to_10.log
```
