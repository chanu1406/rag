from pathlib import Path
import sys
import yaml

# Add root project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import DocumentLoader
from src.config import load_config
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()

def test_extraction():
    """Manual test for metadata extraction pipeline."""
    config = load_config()
    
    # Ensure extraction is enabled
    if not config.ingestion.extraction.enabled:
        console.print("[red]Extraction is disabled in config![/red]")
        return

    loader = DocumentLoader(config)
    
    # Create a dummy test file
    test_file = Path("test_doc.txt")
    test_content = """
    Department of Engineering
    University of California, Merced
    Report Date: 2024-05-15
    
    Project: Quantum Tensor Analysis
    
    Summary:
    This document outlines the results of the Spring 2024 semaphore logic gates testing.
    The primary focus was on minimizing VRAM usage while maximizing throughput.
    """
    
    with open(test_file, "w") as f:
        f.write(test_content)
        
    try:
        console.print(Panel(f"[bold]Testing Extraction on {test_file}[/bold]", border_style="cyan"))
        
        docs = loader.load_file(test_file)
        
        if docs:
            metadata = docs[0].metadata
            console.print("\n[bold green]Success! Extracted Metadata:[/bold green]")
            console.print(JSON.from_data(metadata))
            
            # Assertions
            if "Engineering" in metadata.get("department", ""):
                console.print("\n[bold blue]Department Check: PASS[/bold blue]")
            else:
                console.print(f"\n[bold red]Department Check: FAIL (Got {metadata.get('department')})[/bold red]")
                
        else:
            console.print("[red]No documents loaded.[/red]")
            
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    test_extraction()
