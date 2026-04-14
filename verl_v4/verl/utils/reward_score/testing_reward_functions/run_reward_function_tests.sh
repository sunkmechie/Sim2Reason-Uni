file_to_run=$1

case $file_to_run in
    0)
        echo "Computing reward function stats."
        # Redirect all output (stdout and stderr) to the log file
        exec > Log-Files/Reward_Function_Stats/Log-File.log 2>&1

        # Run the Python script
        python3 -B -u test_reward_function.py
        ;;
    1)
        echo "Generating new question and answer sample and generating answers for sample."
        python3 -B -u test_reward_function.py
        ;;
esac