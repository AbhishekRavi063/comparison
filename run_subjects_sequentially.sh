#!/bin/zsh

# Sequential Runner for 20-Subject Gold Standard Benchmark
# Usage: ./run_subjects_sequentially.sh

CONFIG="config/config_alljoined_preprint_full.yml"

# Subject list: 1 to 20
for sub in {1..20}
do
    echo "=========================================================="
    echo "🚀 STARTING SUBJECT $sub..."
    echo "=========================================================="
    
    # Run subject-by-subject to ensure 100% RAM flush
    python3 -m src.run_all --config "$CONFIG" --subjects "$sub"
    
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✅ SUBJECT $sub COMPLETED SUCCESSFULLY."
    else
        echo "❌ SUBJECT $sub FAILED or was KILLED."
        # Optional: exit on failure
        # exit 1
    fi
    
    echo "Sleeping for 5 seconds to let OS settle..."
    sleep 5
done

echo "🎉 ALL SUBJECTS FINISHED!"
