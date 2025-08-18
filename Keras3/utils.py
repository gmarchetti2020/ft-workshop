import os
import re
import sys

def load_config(config_path: str = "config.conf"):
    """
    Parses a shell-style config file and loads variables into the environment.
    This allows Python to read variables from the same config file used by shell scripts.
    """
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Match lines like 'KEY=VALUE', 'KEY="VALUE"', or 'export KEY=VALUE'
                match = re.match(r'^(?:export\s+)?([\w_]+)=(.*)', line)
                if match:
                    key, value = match.groups()
                    # Remove surrounding quotes (single or double)
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at '{config_path}'")
        sys.exit(1)