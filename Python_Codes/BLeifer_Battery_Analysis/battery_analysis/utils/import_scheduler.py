"""Schedule recurring imports from configured directories."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Dict, List

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from battery_analysis.utils.import_directory import import_directory
from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path(
    os.environ.get(
        "IMPORT_SCHEDULER_CONFIG",
        Path.home() / ".import_scheduler.json",
    )
)
CONTROL_FILE = Path(
    os.environ.get(
        "IMPORT_SCHEDULER_CONTROL",
        Path.home() / ".import_scheduler.control",
    )
)


def load_jobs(path: Path = CONFIG_PATH) -> List[Dict[str, str]]:
    """Return job definitions from ``path`` if it exists."""
    if not path.exists():
        return []
    if path.suffix.lower() == ".toml":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return list(data.get("jobs", []))
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("jobs", []))


def save_jobs(jobs: List[Dict[str, str]], path: Path = CONFIG_PATH) -> None:
    """Persist ``jobs`` to ``path`` in JSON or TOML format."""
    if path.suffix.lower() == ".toml":
        with open(path, "w", encoding="utf-8") as fh:
            for job in jobs:
                fh.write("[[jobs]]\n")
                directory = job.get("directory", "")
                directory = directory.replace("\\", "\\\\").replace('"', '\\"')
                cron = job.get("cron", "")
                cron = cron.replace("\\", "\\\\").replace('"', '\\"')
                fh.write(f'directory = "{directory}"\n')
                fh.write(f'cron = "{cron}"\n')
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"jobs": jobs}, fh, indent=2)


def _read_control_command() -> str | None:
    try:
        cmd = CONTROL_FILE.read_text(encoding="utf-8").strip().lower()
        return cmd or None
    except FileNotFoundError:
        return None


def _schedule_jobs(
    scheduler: BlockingScheduler,
    jobs: List[Dict[str, str]],
) -> None:
    for job in jobs:
        directory = job.get("directory")
        cron = job.get("cron")
        if not directory or not cron:
            continue
        try:
            trigger = CronTrigger.from_crontab(cron)
        except ValueError:  # pragma: no cover - invalid cron
            logger.warning(
                "Invalid cron expression %s for %s",
                cron,
                directory,
            )
            continue
        scheduler.add_job(
            import_directory,
            trigger,
            args=[directory],
            id=directory,
        )


def start() -> None:
    """Start the scheduler and run until a stop command is issued."""
    CONTROL_FILE.write_text(
        "start",
        encoding="utf-8",
    )
    jobs = load_jobs()
    if not jobs:
        logger.info("No import jobs configured; exiting")
        return

    scheduler = BlockingScheduler()
    _schedule_jobs(scheduler, jobs)

    def _watch_control() -> None:
        if _read_control_command() == "stop":
            scheduler.shutdown()

    scheduler.add_job(
        _watch_control,
        "interval",
        seconds=5,
        id="_control",
    )
    logger.info(
        "Starting import scheduler with %d job(s)",
        len(jobs),
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


def stop() -> None:
    """Signal a running scheduler to stop."""
    CONTROL_FILE.write_text(
        "stop",
        encoding="utf-8",
    )


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manage scheduled directory imports",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="Start running scheduled jobs")
    sub.add_parser("stop", help="Stop a running scheduler")
    args = parser.parse_args(argv)
    if args.command == "start":
        start()
    elif args.command == "stop":
        stop()
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
