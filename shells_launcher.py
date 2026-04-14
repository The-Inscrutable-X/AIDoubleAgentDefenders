from datetime import datetime
import socket
import shlex
import wandb
import subprocess
import argparse
import os
import multiprocessing
import sys
from dotenv import load_dotenv

# Set multiprocessing method to 'spawn' to prevent CUDA conflicts with vllm
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))

def commit_to_github(commit_message):
    """Commit and push changes to GitHub"""
    try:
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit with message
        try:
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
        except:
            pass
        
        # Push to remote
        subprocess.run(["git", "push"], check=True)
        
        print(f"Successfully committed and pushed: {commit_message}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Git operation failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_name", "-s", type=str, required=True)
    parser.add_argument("--log_directory", type=str, required=False, default=os.environ["LOG_DIRECTORY"])
    parser.add_argument("--log_name", "-l", type=str, required=True)
    parser.add_argument("--enable_git_commit", "-eg", dest="git_commit", action="store_true", help="Commit and push to GitHub before running")
    parser.add_argument("--script_args", "-a", nargs="+", help="Arguments for the script")
    args = parser.parse_args()
    
    # Commit to GitHub if requested
    if args.git_commit:
        commit_message = f"Auto-commit before running {args.log_name}"
        if not commit_to_github(commit_message):
            print("Warning: Git commit failed, but continuing with script execution...")
    
    command = [args.script_name] + (args.script_args if args.script_args else [])

    rerun_command = shlex.join([sys.executable] + sys.argv)
    print(f"Rerun: \n{rerun_command}")

    # Initialize a wandb run
    run = wandb.init(project="da-project-tracker", name=args.log_name)

    # Grab Run Info script content and hostname
    hostname = socket.gethostname()
    os.makedirs(args.log_directory, exist_ok=True)
    log_file_path = os.path.join(args.log_directory, args.log_name)

    # Set environment variables such that downstream processes can access them
    # Also set vllm-specific environment variables to prevent multiprocessing conflicts
    env = os.environ.copy()
    env['SHELLS_LAUNCHER_LOG_NAME'] = args.log_name
    env['SHELLS_LAUNCHER_LOG_PATH'] = log_file_path
    env['WANDB_PARENT_RUN_ID'] = run.id
    env['WANDB_PROJECT'] = run.project
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # Ensure vllm uses spawn method
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '')  # Preserve CUDA settings
    
    with open(args.script_name, "r") as script_file:
        script_content = script_file.read()
    wandb.log({"script_content": script_content})
    wandb.log({"script_name": args.script_name})
    now = datetime.now()
    with open(log_file_path, "w", encoding='utf-8') as log_file:
        log_file.write(f"Executing on server: {hostname}\n")
        log_file.write(f"Start execution time: {now.strftime('%m-%d %I:%M %p [%Y]')}\n")
        log_file.write(f"Rerun: {rerun_command}\n")
        log_file.write(f"Script name: {args.script_name}\n")
        log_file.write("shell_script_content_start:\n")
        log_file.write(script_content)
        log_file.write("\n:shell_script_content_end\n")

    # Setup logging and run using subprocess instead of sh.bash for better vllm compatibility
    try:
        # Use subprocess.Popen for better control over process creation
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True,
            env=env,
            # Use a separate process group to avoid interference with vllm
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                    
                current_time = datetime.now().strftime("%d; %I:%M %p")
                if "Error:" in line or "Traceback (most recent call last)" in line or "Killed" in line or "errored:" in line:
                    wandb.log({"Errors": str(current_time) + str(line)})
                
                # Print to console
                print(line, end="")
                # Write to log file
                log_file.write(line)
                log_file.flush()  # Ensure immediate writing
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = f"Script exited with return code: {return_code}"
            print(error_msg)
            wandb.log({"Script Exit Code": return_code})
            
    except Exception as e:
        error_msg = f"Error: Failed to execute script: {e}"
        print(error_msg)
        wandb.log({"Execution Error": error_msg})
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write(f"\n{error_msg}\n")

    # Finish the run
    run.finish()