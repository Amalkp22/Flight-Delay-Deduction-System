import os
import time
import difflib
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_DIR, "project_changes.log")
INTERVAL = 60 # seconds

# File extensions to monitor
MONITORED_EXTENSIONS = {'.py', '.html', '.css', '.js', '.json', '.bat', '.sh', '.md', '.txt'}

# Folders to exclude
EXCLUDE_DIRS = {'__pycache__', '.git', '.idea', '.vscode', 'venv', '.gemini'}

# Specific files to exclude
EXCLUDE_FILES = {'project_changes.log', 'change_logger.py'}

def should_monitor(file_path):
    rel_path = os.path.relpath(file_path, PROJECT_DIR)
    parts = rel_path.split(os.sep)
    
    # Exclude files in excluded directories
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
            
    filename = parts[-1]
    if filename in EXCLUDE_FILES:
        return False
        
    _, ext = os.path.splitext(filename)
    if ext.lower() not in MONITORED_EXTENSIONS:
        return False
        
    # Exclude very large files (e.g. datasets)
    try:
        if os.path.getsize(file_path) > 10 * 1024 * 1024: # 10MB limit
            return False
    except OSError:
        return False
        
    return True

def get_file_content(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except Exception:
        return []

def main():
    print(f"Starting Project Change Logger in {PROJECT_DIR}")
    print(f"Logging modifications every {INTERVAL}s to {LOG_FILE}")
    
    # Initialize file states: path -> list of lines
    file_states = {}
    
    # First scan to establish baseline
    for root, dirs, files in os.walk(PROJECT_DIR):
        # In-place modification to skip excluded directories in os.walk
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            path = os.path.join(root, file)
            if should_monitor(path):
                file_states[path] = get_file_content(path)
                
    # Append start message to log
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n==================================================\n")
        log.write(f"LOGGER STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Monitoring {len(file_states)} files.\n")
        log.write(f"==================================================\n")

    while True:
        try:
            time.sleep(INTERVAL)
            current_files = {}
            modified_files = []
            deleted_files = []
            new_files = []
            
            # Scan current files
            for root, dirs, files in os.walk(PROJECT_DIR):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    path = os.path.join(root, file)
                    if should_monitor(path):
                        current_files[path] = True
            
            # Detect deletions
            for path in list(file_states.keys()):
                if path not in current_files:
                    deleted_files.append(path)
            
            # Detect modifications and additions
            for path in current_files:
                if path not in file_states:
                    new_files.append(path)
                else:
                    # We check diff
                    new_content = get_file_content(path)
                    if new_content != file_states[path]:
                        modified_files.append((path, new_content))
            
            # Write to log if changes are found
            if modified_files or deleted_files or new_files:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(LOG_FILE, 'a', encoding='utf-8') as log:
                    log.write(f"\n--- CHANGE DETECTED AT {timestamp} ---\n")
                    
                    for path in deleted_files:
                        rel_path = os.path.relpath(path, PROJECT_DIR)
                        log.write(f"[DELETED] {rel_path}\n")
                        del file_states[path]
                        
                    for path in new_files:
                        rel_path = os.path.relpath(path, PROJECT_DIR)
                        log.write(f"[NEW FILE] {rel_path}\n")
                        content = get_file_content(path)
                        file_states[path] = content
                        log.write(f"Initial lines: {len(content)}\n")
                        
                    for path, new_content in modified_files:
                        rel_path = os.path.relpath(path, PROJECT_DIR)
                        old_content = file_states[path]
                        
                        log.write(f"[MODIFIED] {rel_path}\n")
                        # Generate diff
                        diff = difflib.unified_diff(
                            old_content, new_content,
                            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
                            lineterm=''
                        )
                        diff_text = '\n'.join(diff)
                        if diff_text:
                            log.write(diff_text + '\n')
                        else:
                            log.write("Metadata changes / content unmodified.\n")
                            
                        file_states[path] = new_content
                        
                    log.write("-" * 40 + "\n")
                    log.flush()
                    
        except Exception as e:
            # Prevent crash on errors (e.g. file lock, permission error)
            try:
                with open(LOG_FILE, 'a', encoding='utf-8') as log:
                    log.write(f"\n[ERROR] Logger error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}\n")
            except Exception:
                pass

if __name__ == '__main__':
    main()
