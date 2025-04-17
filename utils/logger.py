import os
from datetime import datetime
from typing import Optional, Any
from colorama import init
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

# Initialize colorama for cross-platform color support
init(autoreset=True)

class Logger:
    def __init__(self, log_file: str = "logs/app.log"):
        """
        Initialize the Logger with a log file path.
        
        Args:
            log_file (str): Path to the log file
        """
        self.log_file = log_file
        self.console = Console()
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create or clear log file
        with open(log_file, "a") as f:
            f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def _log_to_file(self, message: str):
        """Write a message to the log file."""

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def print_welcome(self, title: str, subtitle: str = ""):
        """Print a styled welcome message."""
        self.console.print("\n")
        self.console.print(Panel(
            f"[bold blue]{title}[/]\n[cyan]{subtitle}[/]",
            border_style="blue",
            expand=False
        ))
        self.console.print("\n")

    def print_separator(self):
        """Print a separator line."""
        self.console.print("─" * 80, style="dim")

    def info(self, message: str):
        """Print an info message."""
        self.console.print(f"[blue]ℹ[/] {message}")
        self._log_to_file(f"INFO: {message}")

    def success(self, message: str):
        """Print a success message."""
        self.console.print(f"[green]✓[/] {message}")
        self._log_to_file(f"SUCCESS: {message}")

    def warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]⚠[/] {message}")
        self._log_to_file(f"WARNING: {message}")

    def error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]✗[/] {message}")
        self._log_to_file(f"ERROR: {message}")

    def debug(self, message: str):
        """Print a debug message."""
        self.console.print(f"[dim]🔍 {message}[/]")
        self._log_to_file(f"DEBUG: {message}")

    def metrics(self, operation: str, start_time: float, end_time: float, extra_info: Optional[Any] = None):
        """Log execution metrics."""
        execution_time = end_time - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create metrics message
        metrics_msg = f"{operation}: {execution_time:.2f} seconds"
        
        if extra_info:
            if isinstance(extra_info, dict):
                metrics_msg += f"\nDetails: {json.dumps(extra_info, indent=2)}"
                
                # Add extra info to console output
                extra_info_panel = ""
                for key, value in extra_info.items():
                    extra_info_panel += f"[bold]{key}:[/] {value}\n"
            else:
                metrics_msg += f"\nDetails: {extra_info}"

        # Print to console with styling
        panel_content = f"[bold]Operation:[/] {operation}\n[bold]Duration:[/] {execution_time:.2f}s\n[bold]Timestamp:[/] {timestamp}"
        
        # Add extra info to the same panel if available
        if extra_info and isinstance(extra_info, dict):
            panel_content += f"\n\n[blue]Additional Metrics:[/]\n{extra_info_panel.strip()}"
            
        self.console.print(Panel(
            panel_content,
            title="[blue]Metrics[/]",
            border_style="blue"
        ))

        # Log to file
        self._log_to_file(f"METRICS - {metrics_msg}")

    def progress(self, message: str = "Processing"):
        """Create a progress context for long-running operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )

    def print_chat_message(self, role: str, content: str):
        """Print a chat message with appropriate styling."""
        icon = "🧑" if role.lower() == "user" else "🤖"
        color = "cyan" if role.lower() == "user" else "green"
        
        self.console.print(f"\n[{color}]{icon} {role}:[/]")
        self.console.print(Panel(content, border_style=color))
        self._log_to_file(f"CHAT - {role}: {content}")

    def print_json(self, data: Any, title: str = ""):
        """Print JSON data in a formatted and colored way."""
        if title:
            self.console.print(f"\n[bold blue]{title}[/]")
        
        if isinstance(data, (dict, list)):
            self.console.print_json(data=data)
        else:
            self.console.print(data)
        
        self._log_to_file(f"JSON Output - {title}: {json.dumps(data, indent=2)}")

    def create_table(self, title: str, columns: list):
        """Create and return a styled table."""
        table = Table(title=title, show_header=True, header_style="bold blue")
        for column in columns:
            table.add_column(column)
        return table 