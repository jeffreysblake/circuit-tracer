import argparse
import logging
import os
import time
import warnings


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="CLI for attribution, graph file creation, and server hosting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Attribution subcommand
    attr_parser = subparsers.add_parser("attribute", help="Run attribution analysis on a prompt")

    # Arguments from attribute_batch.py
    attr_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=("Model architecture to use for attribution. Can be inferred from transcoder config."),
    )
    attr_parser.add_argument(
        "-t",
        "--transcoder_set",
        required=True,
        help=(
            "HuggingFace repository ID containing transcoders "
            "(e.g. username/repo-name, username/repo-name@revision)."
        ),
    )
    attr_parser.add_argument("-p", "--prompt", required=True, help="Input prompt text to analyze.")
    attr_parser.add_argument(
        "-o",
        "--graph_output_path",
        help=(
            "Path where to save the attribution graph (.pt file). Required if not "
            "creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights (default: float32).",
    )
    attr_parser.add_argument(
        "--max_n_logits", type=int, default=10, help="Maximum number of logit nodes."
    )
    attr_parser.add_argument(
        "--desired_logit_prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for top logits.",
    )
    attr_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for backward passes."
    )
    attr_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory.",
    )
    attr_parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=7500,
        help="Maximum number of feature nodes.",
    )
    attr_parser.add_argument("--verbose", action="store_true", help="Display progress information.")
    attr_parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help="Enable lazy loading for encoder weights to save memory.",
    )
    attr_parser.add_argument(
        "--lazy-decoder",
        action="store_true",
        default=True,
        help="Enable lazy loading for decoder weights to save memory (default: True).",
    )

    # Arguments for graph creation
    attr_parser.add_argument(
        "--slug",
        type=str,
        help=(
            "Slug for the model metadata (used for graph files). Required if creating "
            "graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--graph_file_dir",
        type=str,
        help=(
            "Path to save the output JSON graph files, and also used as data dir for "
            "server. Required if creating graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Node threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Edge threshold for pruning graph files.",
    )

    # Server arguments
    attr_parser.add_argument(
        "--server",
        action="store_true",
        help="Start a local server to visualize graphs after processing.",
    )
    attr_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")

    # Start-server subcommand
    server_parser = subparsers.add_parser(
        "start-server", help="Start a local server to visualize existing graphs"
    )
    server_parser.add_argument(
        "--graph_file_dir",
        type=str,
        required=True,
        help="Path to the directory containing graph JSON files.",
    )
    server_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Batch evaluate model on task-specific datasets"
    )
    eval_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model architecture to use for evaluation. Can be inferred from transcoder config.",
    )
    eval_parser.add_argument(
        "-t",
        "--transcoder_set",
        required=True,
        help="HuggingFace repository ID containing transcoders",
    )
    eval_parser.add_argument(
        "--task_config",
        required=True,
        help="Path to task configuration YAML file (e.g., task_configs/tool_use.yaml)",
    )
    eval_parser.add_argument(
        "--dataset",
        help="Path to dataset JSON file. If not provided, uses datasets from task config.",
    )
    eval_parser.add_argument(
        "--prompts",
        nargs="+",
        help="List of prompts to evaluate (alternative to --dataset)",
    )
    eval_parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory to save evaluation results",
    )
    eval_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights",
    )
    eval_parser.add_argument(
        "--max_n_logits", type=int, default=10, help="Maximum number of logit nodes"
    )
    eval_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for attribution"
    )
    eval_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory",
    )
    eval_parser.add_argument(
        "--save_graphs",
        action="store_true",
        help="Save full attribution graphs (uses more memory)",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display progress information",
    )

    # Discover SAEs subcommand
    discover_parser = subparsers.add_parser(
        "discover-saes", help="Discover available SAEs/transcoders for different models"
    )
    discover_parser.add_argument(
        "--model",
        help="Filter by model name (partial match)",
    )
    discover_parser.add_argument(
        "--hook_type",
        help="Filter by hook type (e.g., mlp, resid, attn)",
    )

    # Compare checkpoints subcommand
    compare_parser = subparsers.add_parser(
        "compare-checkpoints", help="Compare fine-tuned checkpoints against base model"
    )
    compare_parser.add_argument(
        "--base_model",
        required=True,
        help="Base model name (HuggingFace)",
    )
    compare_parser.add_argument(
        "--transcoder_set",
        required=True,
        help="Transcoder set (trained on base model)",
    )
    compare_parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoint directories",
    )
    compare_parser.add_argument(
        "--task_config",
        required=True,
        help="Task configuration YAML file",
    )
    compare_parser.add_argument(
        "--eval_prompts",
        nargs="+",
        help="Prompts to evaluate (alternative to --dataset)",
    )
    compare_parser.add_argument(
        "--dataset",
        help="Path to dataset JSON file",
    )
    compare_parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory to save comparison results",
    )
    compare_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights",
    )

    args = parser.parse_args()

    if args.command == "attribute":
        run_attribution(args, attr_parser)
    elif args.command == "evaluate":
        run_evaluation(args, eval_parser)
    elif args.command == "discover-saes":
        run_discover_saes(args)
    elif args.command == "compare-checkpoints":
        run_compare_checkpoints(args, compare_parser)
    elif args.command == "start-server" or args.server:
        run_server(args)


def run_attribution(args, parser):
    # Check if one of slug/graph_file_dir is provided but not the other
    if bool(args.slug) != bool(args.graph_file_dir):
        which_one = "slug" if args.slug else "graph_file_dir"
        missing_one = "graph_file_dir" if args.slug else "slug"
        warnings.warn(
            (
                f"You provided --{which_one} but not --{missing_one}. Both are required "
                "for creating graph files."
            ),
            UserWarning,
        )

    # Determine if we're creating graph files
    create_graph_files_enabled = args.slug is not None and args.graph_file_dir is not None

    # Validate arguments
    if args.server and (not args.slug or not args.graph_file_dir):
        parser.error("Both --slug and --graph_file_dir are required when using --server")

    if not create_graph_files_enabled and not args.graph_output_path:
        parser.error(
            "--graph_output_path is required when not creating graph files "
            "(--slug and --graph_file_dir)"
        )

    # Ensure graph output directory exists if needed
    if create_graph_files_enabled:
        os.makedirs(args.graph_file_dir, exist_ok=True)

    import torch

    dtype = args.dtype
    # Convert short dtype string to long dtype string
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Run attribution
    logging.info(f"Generating attribution graph for model: {args.model}")
    logging.info(f"Loading model with dtype: {dtype}")
    logging.info(f'Input prompt: "{args.prompt}"')
    if args.graph_output_path:
        logging.info(f"Output will be saved to: {args.graph_output_path}")
    logging.info(
        f"Including logits with cumulative probability >= {args.desired_logit_prob} "
        f"(max {args.max_n_logits})"
    )
    logging.info(f"Using batch size of {args.batch_size} for backward passes")

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=args.lazy_encoder,
        lazy_decoder=args.lazy_decoder,
    )
    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    model_instance = ReplacementModel.from_pretrained_and_transcoders(
        args.model, transcoder, dtype=dtype
    )

    logging.info("Running attribution...")
    graph = attribute(
        prompt=args.prompt,
        model=model_instance,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        verbose=args.verbose,
        offload=args.offload,
        max_feature_nodes=args.max_feature_nodes,
    )

    # Save to file if output path specified
    if args.graph_output_path:
        logging.info(f"Saving graph to {args.graph_output_path}")
        graph.to_pt(args.graph_output_path)

    # Create graph files if both slug and graph_file_dir are provided
    if create_graph_files_enabled:
        logging.info(f"Creating graph files with slug: {args.slug}")
        create_graph_files(
            graph_or_path=graph,  # Use the graph object directly
            slug=args.slug,
            scan=None,  # No scan argument needed
            output_path=args.graph_file_dir,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        logging.info(f"Graph JSON files written to {args.graph_file_dir}")


def run_evaluation(args, parser):
    """Run batch evaluation on a task dataset."""
    import json
    from pathlib import Path

    import torch

    from circuit_tracer import ReplacementModel
    from circuit_tracer.evaluation import TaskEvaluator, load_task_config
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    # Load task configuration
    logging.info(f"Loading task configuration from {args.task_config}")
    task_config = load_task_config(args.task_config)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert dtype
    dtype = args.dtype
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Load model and transcoders
    logging.info(f"Loading transcoders from {args.transcoder_set}")
    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
    )

    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    logging.info(f"Loading model {args.model}")
    model = ReplacementModel.from_pretrained_and_transcoders(
        args.model, transcoder, dtype=dtype
    )

    # Create evaluator
    attribution_kwargs = {
        "max_n_logits": args.max_n_logits,
        "batch_size": args.batch_size,
        "offload": args.offload,
    }
    evaluator = TaskEvaluator(
        model=model,
        task_config=task_config,
        attribution_kwargs=attribution_kwargs,
    )

    # Get prompts to evaluate
    prompts = []
    if args.prompts:
        prompts = args.prompts
        logging.info(f"Evaluating {len(prompts)} prompts from command line")
    elif args.dataset:
        # Load from JSON file
        with open(args.dataset) as f:
            dataset = json.load(f)
        prompts = [item["prompt"] for item in dataset]
        logging.info(f"Loaded {len(prompts)} prompts from {args.dataset}")
    elif task_config.datasets:
        # Use first dataset from task config
        dataset_info = task_config.datasets[0]
        dataset_path = Path(dataset_info.path)
        if not dataset_path.exists():
            # Try relative to task_configs directory
            dataset_path = Path(args.task_config).parent.parent / dataset_info.path
        if dataset_path.exists():
            with open(dataset_path) as f:
                dataset = json.load(f)
            prompts = [item["prompt"] for item in dataset]
            logging.info(f"Loaded {len(prompts)} prompts from {dataset_path}")
        else:
            parser.error(
                f"Dataset file not found: {dataset_info.path}. "
                "Please provide --dataset or --prompts"
            )
    else:
        parser.error("No prompts provided. Use --prompts or --dataset or configure datasets in task config")

    # Run evaluation
    logging.info(f"Starting evaluation on {len(prompts)} prompts...")
    results = evaluator.evaluate_batch(
        prompts=prompts,
        verbose=args.verbose,
        save_graphs=args.save_graphs,
    )

    # Save results
    results_file = output_dir / f"results_{task_config.task_name}.json"
    with open(results_file, "w") as f:
        json.dump(results.summary(), f, indent=2)

    logging.info(f"Results saved to {results_file}")

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info(f"EVALUATION SUMMARY: {task_config.task_name}")
    logging.info("=" * 60)
    logging.info(f"Model: {args.model}")
    logging.info(f"Prompts evaluated: {len(prompts)}")
    logging.info(f"\nAggregate Metrics:")
    for metric_name, value in results.aggregate_metrics.items():
        logging.info(f"  {metric_name}: {value:.4f}")
    logging.info("=" * 60)


def run_discover_saes(args):
    """Discover available SAEs/transcoders."""
    from circuit_tracer.utils.sae_discovery import SAERegistry

    registry = SAERegistry()

    if args.model:
        results = registry.search_saes(
            model_name=args.model,
            hook_type=args.hook_type,
        )

        if results:
            print(f"\nFound {len(results)} SAE(s) for model: {args.model}\n")
            for sae in results:
                print(f"Name: {sae.name}")
                print(f"Repo: {sae.repo_id}")
                print(f"Hook Type: {sae.hook_point_type}")
                if sae.d_sae:
                    print(f"Features: {sae.d_sae:,}")
                print(f"Description: {sae.description}")
                print()
        else:
            print(f"No SAEs found for model: {args.model}")
    else:
        registry.print_summary()


def run_compare_checkpoints(args, parser):
    """Compare fine-tuned checkpoints against base model."""
    import json
    from pathlib import Path

    import torch

    from circuit_tracer.evaluation import FineTuneComparator, load_task_config

    # Convert dtype
    dtype = args.dtype
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Load task config
    task_config = load_task_config(args.task_config)

    # Get evaluation prompts
    if args.eval_prompts:
        prompts = args.eval_prompts
    elif args.dataset:
        with open(args.dataset) as f:
            dataset = json.load(f)
        prompts = [item["prompt"] for item in dataset]
    else:
        parser.error("Must provide --eval_prompts or --dataset")

    logging.info(f"Comparing {len(args.checkpoints)} checkpoints against base model...")
    logging.info(f"Base model: {args.base_model}")
    logging.info(f"Transcoders: {args.transcoder_set}")

    # Create comparator
    comparator = FineTuneComparator(
        base_model=args.base_model,
        transcoder_set=args.transcoder_set,
        task_config=task_config,
        dtype=dtype,
    )

    # Run comparison
    analysis = comparator.compare_checkpoints(
        checkpoint_paths=args.checkpoints,
        eval_prompts=prompts,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"checkpoint_comparison_{task_config.task_name}.json"

    comparator.save_analysis(analysis, output_file)

    # Print summary
    print("\n" + "=" * 70)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Task: {task_config.task_name}")
    print(f"Checkpoints analyzed: {analysis.summary['total_checkpoints']}")
    print(f"\nBest coverage: {analysis.summary['best_coverage_checkpoint']}")
    print(f"Best coherence: {analysis.summary['best_coherence_checkpoint']}")
    print(f"Final transcoder validity: {analysis.summary['final_validity']:.2f}")
    print(f"Validity trend: {analysis.summary['validity_trend']}")
    print("=" * 70)

    print("\nDetailed results:")
    for i, ckpt in enumerate(analysis.checkpoints):
        print(f"\n{i+1}. {ckpt.checkpoint_name}")
        print(f"   {ckpt.interpretation}")
        print(f"   Metric changes:")
        for metric, delta in ckpt.metric_deltas.items():
            sign = "+" if delta > 0 else ""
            print(f"     {metric}: {sign}{delta:.3f}")

    print(f"\n✅ Full analysis saved to: {output_file}")


def run_server(args):
    from circuit_tracer.frontend.local_server import serve

    logging.info(f"Starting server on port {args.port}...")
    logging.info(f"Serving data from: {os.path.abspath(args.graph_file_dir)}")
    server = serve(data_dir=args.graph_file_dir, port=args.port)
    try:
        logging.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        server.stop()


if __name__ == "__main__":
    main()
