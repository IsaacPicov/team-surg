import itertools
import json

if __name__ == "__main__":

    # Base command parameters
    python_cmd = "python3 main.py train"

    # The five new boolean flags
    flags = [
        "has_distance_traveled",
        "has_avg_stability",
        "has_elbow_avg",
        "has_wrist_avg",
        "has_hand_wrist_distance"
    ]

    # The exclusion-groups arrays to cycle through
    exclude_groups_list = [
        ["spine_joints"],
        ["head_joints", "spine_joints", "thorax_joints"],
        ["head_joints", "leg_joints", "spine_joints"],
        ["leg_joints", "spine_joints", "thorax_joints"],
        ["leg_joints", "spine_joints", "thorax_joints", "head_joints"],
        ["head_joints", "thorax_joints"],
        ["leg_joints", "spine_joints"]
    ]

    # Generate all combinations of True/False for the flags
    bool_combinations = list(itertools.product([True, False], repeat=len(flags)))

    # Write the bash script
    with open("run_all.sh", "w") as sh:
        sh.write("#!/bin/bash\n\n")
        for exclude in exclude_groups_list:
            exclude_json = json.dumps(exclude)
            for combo in bool_combinations:
                # Build exp_name to reflect settings
                suffix = []
                for flag, val in zip(flags, combo):
                    if val:
                        suffix.append(f"{flag} ")
                suffix_str = ",".join(suffix)
                exp_name = f"{suffix_str},exclude={exclude_json}"

                # Build the full command
                parts = [python_cmd,
                        f"--exp_name \"{exp_name}\"",
                        f"--exclude_groups '{exclude_json}'"]
                for flag, val in zip(flags, combo):
                    parts.append(f"--{flag}={val}")

                sh.write(" ".join(parts) + "\n")

    print("Generated run_ablations.sh with all ablation commands.")