import sys
from pathlib import Path
import click
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.logger import setup_logging, logger
from src.ingest import DocumentLoader
from src.vectorstore import VectorManager
from src.rag import RAGEngine
from src.utils import check_cuda_available

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Local Brain RAG - Privacy-focused document chat."""
    pass

@cli.command()
@click.argument('source', type=click.Path(exists=True))
@click.option('--reset', is_flag=True, help='Clear existing vector store')
def ingest(source: str, reset: bool):
    """
    Ingest documents from SOURCE path.
    
    Example: python main.py ingest ./data/documents
    """
    try:
        # Load config
        config = load_config()
        setup_logging(config)
        
        # Check CUDA availability
        cuda_available = check_cuda_available()
        if not cuda_available and config.embedding.device == "cuda":
            console.print("[yellow]Notice: CUDA not available but config requires 'cuda' device[/yellow]")
            console.print("[yellow]   Embeddings will run on CPU (slower performance)[/yellow]")
        
        console.print(Panel("[bold]Document Ingestion[/bold]"))
        
        # Initialize components
        vector_manager = VectorManager(config)
        loader = DocumentLoader(config)
        
        # Reset if requested
        if reset:
            console.print("[yellow]Clearing existing embeddings...[/yellow]")
            vector_manager.clear()
        
        # Load documents
        source_path = Path(source)
        console.print(f"[cyan]Loading from:[/cyan] {source_path}")
        
        if source_path.is_dir():
            files = [f for f in source_path.rglob("*") if f.is_file() and f.suffix in config.ingestion.valid_extensions]
            documents = []
            for file in files:
                try:
                    docs = loader.load_file(file)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipped {file.name}: {e}")
        else:
            documents = loader.load_file(source_path)
        
        if not documents:
            console.print("[red]No valid documents found.[/red]")
            return
        
        # Chunk
        chunks = loader.chunk_documents(documents)
        console.print(f"[green]Created {len(chunks)} chunks from {len(documents)} documents[/green]")
        
        # Embed and store
        with console.status("[bold green]Generating embeddings (GPU)...[/bold green]"):
            ids = vector_manager.add_documents(chunks)
        
        console.print(f"[bold green]Successfully ingested {len(ids)} chunks[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Ingestion failed")
        sys.exit(1)

@cli.command()
def chat():
    """
    Start interactive chat session.
    
    Example: python main.py chat
    """
    try:
        # Load config
        config = load_config()
        setup_logging(config)
        
        console.print(Panel(
            "[bold]Local Brain RAG[/bold]\n"
            "Commands: /exit, /help"
        ))
        
        # Initialize
        vector_manager = VectorManager(config)
        rag_engine = RAGEngine(config, vector_manager)
        
        logger.info("Chat session started")
        
        # Chat loop
        while True:
            try:
                question = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                if question.lower() == "/exit":
                    console.print("[yellow]Exiting application.[/yellow]")
                    break
                elif question.lower() == "/help":
                    console.print("Available commands:\n  /exit - Quit\n  /help - This message")
                    continue
                
                # Query
                with console.status("[bold green]Processing...[/bold green]"):
                    result = rag_engine.query(question)
                
                # Display answer
                console.print("[bold green]Assistant:[/bold green]")
                console.print(Markdown(result["answer"]))
                
                # Show sources
                if result["sources"]:
                    console.print("\n[dim]Sources:[/dim]")
                    for src in result["sources"]:
                        console.print(f"  - {src['filename']}")
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting application.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Query failed")
    
    except Exception as e:
        console.print(f"[bold red]Startup Error:[/bold red] {e}")
        logger.exception("Chat initialization failed")
        sys.exit(1)

if __name__ == "__main__":
    cli()
