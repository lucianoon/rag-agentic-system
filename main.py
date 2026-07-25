"""Main CLI entry point for the RAG Agentic System."""
import argparse
import logging
import sys
from pathlib import Path

from src.rag_agent import AgenticRAG, create_context, load_config

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; env vars still work without it
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_interactive(agent: AgenticRAG) -> None:
    """Run the interactive CLI mode."""
    print("\n🤖 RAG Agentic System - Modo interativo")
    print("Digite 'help' para comandos, 'quit' para sair\n")

    while True:
        try:
            user_input = input("RAG> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  <question>  - Ask a question")
                print("  stats       - Show system statistics")
                print("  history     - Show recent task history")
                print("  clear       - Clear vector store")
                print("  quit/exit/q - Exit the system\n")
                continue

            if user_input.lower() == "stats":
                stats = agent.get_stats()
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue

            if user_input.lower() == "history":
                recent = agent.context.memory.recent(limit=5)
                print(f"\n📝 Recent Tasks ({len(recent)}):")
                for i, log in enumerate(recent, 1):
                    print(f"  {i}. {log.query} ({len(log.steps)} steps)")
                print()
                continue

            if user_input.lower() == "clear":
                agent.clear_memory()
                print("\n✅ Vector store cleared\n")
                continue

            # Process query
            print(f"\n🔍 Processing: {user_input}")
            response = agent.query(user_input)

            print(f"\n📄 Response:\n{response.answer}\n")

            if response.references:
                print(f"📚 References ({len(response.references)}):")
                for ref in response.references[:3]:
                    print(f"  - {ref}")
                if len(response.references) > 3:
                    print(f"  ... and {len(response.references) - 3} more")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error("Error processing request: %s", e, exc_info=True)
            print(f"\n❌ Error: {e}\n")


def run_single_task(agent: AgenticRAG, task: str) -> None:
    """Run a single task and exit."""
    print(f"\n🔍 Processing: {task}\n")
    response = agent.query(task)
    print(f"📄 Response:\n{response.answer}\n")

    if response.references:
        print(f"📚 References: {', '.join(response.references[:3])}")
        if len(response.references) > 3:
            print(f"  ... and {len(response.references) - 3} more")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Agentic System")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Single task to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--add-docs",
        nargs="+",
        help="Add documents to the system",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = load_config(args.config)

        # Create context and agent
        context = create_context(config)
        agent = AgenticRAG(context)
        agent.initialize()

        # Add documents if specified
        if args.add_docs:
            print(f"\n📥 Adding {len(args.add_docs)} documents...")
            agent.add_documents(args.add_docs)
            print("✅ Documents added\n")

        # Run in appropriate mode
        if args.task:
            run_single_task(agent, args.task)
        else:
            run_interactive(agent)

        return 0

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())