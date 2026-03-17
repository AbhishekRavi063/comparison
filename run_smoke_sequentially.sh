#!/bin/zsh

# Sequential Runner for Alljoined Smoke Test (Only GEDAI)
# Usage: ./run_smoke_sequentially.sh

CONFIG="config/config_alljoined_professor_smoke.yml"
PYTHON_BIN="./.venv/bin/python"

# Subject list: 1 and 2
for sub in 1 2
do
    echo "=========================================================="
    echo "🚀 STARTING SMOKE TEST FOR SUBJECT $sub..."
    echo "=========================================================="
    
    # Run subject-by-subject with the virtual environment's python
    $PYTHON_BIN -m src.run_all --config "$CONFIG" --subjects "$sub"
    
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✅ SUBJECT $sub COMPLETED SUCCESSFULLY."
    else
        echo "❌ SUBJECT $sub FAILED or was KILLED."
    fi
    
    echo "Sleeping for 5 seconds to let OS settle..."
    sleep 5
done

echo "🎉 SMOKE TEST FINISHED!"
