"""
Push and run one task's runner on Kaggle from your terminal.

    python kaggle_push.py task1 smoke      Quick test of every step (tasks 1 and 2)
    python kaggle_push.py task1 full       The real run, from the start
    python kaggle_push.py task1 resume     Continue from the previous version's outputs
    python kaggle_push.py task1 status     Show whether the run is going, finished, or failed
    python kaggle_push.py task1 results    Download the results zip into the task's folder

Setup: pip install kaggle, then create an API token at kaggle.com/settings.
Kaggle clones this code from GitHub, so commit and push before running.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
TASKS = {
    "task1": {"notebook": "kaggle_task1.ipynb", "slug": "atml-pa3-task-1", "title": "ATML PA3 Task 1", "flag": "SMOKE_TEST_ONLY", "zip": "task1_results.zip", "folder": "task1_llm_finetuning"},
    "task2": {"notebook": "kaggle_task2.ipynb", "slug": "atml-pa3-task-2", "title": "ATML PA3 Task 2", "flag": "SMOKE_TEST", "zip": "task2_outputs.zip", "folder": "task2_vlm_grounding"},
    "task3": {"notebook": "kaggle_task3.ipynb", "slug": "atml-pa3-task-3", "title": "ATML PA3 Task 3", "flag": None, "zip": "task3_outputs.zip", "folder": "task3_test_time_compute"},
}

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()

def kaggle_username(explicit):
    """Only the username is read from Kaggle's config. Credentials stay with the kaggle tool."""
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            username = json.load(f).get("username")
        if username:
            return username
    sys.exit("Kaggle username not found. Pass it with --user YOUR_KAGGLE_USERNAME.")

def check_code_is_on_github():
    """Kaggle runs what is on GitHub, so local-only changes would silently not run."""
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    if dirty:
        sys.exit(f"You have uncommitted changes, which Kaggle would not see ({len(dirty)} files). Commit and push first:\n" + "\n".join(dirty[:10]))
    remote = git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if not remote or remote[0] != git("rev-parse", "HEAD"):
        sys.exit(f"Branch {branch} is not fully pushed. Run: git push -u origin {branch}")
    return branch

def build_notebook(task, mode, branch):
    """The runner with this run's branch and mode filled in. The file in the repository is not changed."""
    with open(os.path.join(REPO, task["notebook"]), "r", encoding="utf-8") as f:
        notebook = json.load(f)
    replaced_branch = 0
    replaced_flag = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            new_line, count = re.subn(r'^BRANCH = "[^"]*"', f'BRANCH = "{branch}"', line)
            replaced_branch += count
            if task["flag"]:
                new_line, count = re.subn(rf"^{task['flag']} = (True|False)", f"{task['flag']} = {mode == 'smoke'}", new_line)
                replaced_flag += count
            cell["source"][i] = new_line
    if replaced_branch != 1 or (task["flag"] and replaced_flag != 1):
        sys.exit(f"Could not set the branch and mode in {task['notebook']}")
    return notebook

def push(task, mode, user, dry_run):
    if mode == "smoke" and not task["flag"]:
        sys.exit("Task 3 has no smoke mode. Use full.")
    branch = git("rev-parse", "--abbrev-ref", "HEAD") if dry_run else check_code_is_on_github()
    kernel_id = f"{user}/{task['slug']}"
    metadata = {
        "id": kernel_id,
        "title": task["title"],
        "code_file": task["notebook"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [kernel_id] if mode == "resume" else [],
        "model_sources": [],
    }
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, task["notebook"]), "w", encoding="utf-8") as f:
            json.dump(build_notebook(task, mode, branch), f)
        with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=1)
        print(f"Pushing {kernel_id} | mode: {mode} | branch: {branch}")
        if dry_run:
            print(json.dumps(metadata, indent=1))
            return
        subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True)
    print(f"Started. Watch it at https://www.kaggle.com/code/{kernel_id}")

def download_results(task, kernel_id):
    """Fetch only the results zip, not the checkpoints, and unzip it into the task's folder."""
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", folder, "--file-pattern", re.escape(task["zip"])], check=True)
        path = os.path.join(folder, task["zip"])
        if not os.path.exists(path):
            sys.exit(f"{task['zip']} is not in the latest run's output yet.")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(os.path.join(REPO, task["folder"]))
    print(f"Results are in {task['folder']}/. Review them, then commit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push and run one task's runner on Kaggle")
    parser.add_argument("task", choices=list(TASKS))
    parser.add_argument("action", choices=["smoke", "full", "resume", "status", "results"])
    parser.add_argument("--user", default=None, help="Your Kaggle username, if it is not found automatically")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be pushed without pushing")
    args = parser.parse_args()

    task = TASKS[args.task]
    user = kaggle_username(args.user)
    if args.action == "status":
        subprocess.run(["kaggle", "kernels", "status", f"{user}/{task['slug']}"], check=True)
    elif args.action == "results":
        download_results(task, f"{user}/{task['slug']}")
    else:
        push(task, args.action, user, args.dry_run)
