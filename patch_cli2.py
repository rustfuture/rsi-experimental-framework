with open("rsi_framework/cli.py", "r") as f:
    content = f.read()

import_block = """
    generator = None
    if args.provider == "json":
        if args.proposals is None:
            raise SystemExit("--provider json requires --proposals PATH")
        generator = JsonFileProposalProvider(args.proposals)
    elif args.provider == "llm":
        try:
            provider = LocalTransformersProvider()
            generator = LLMMutationGenerator(provider, "Improve the policy based on these keywords.")
        except RuntimeError as e:
            if "EXPERIMENT_BLOCKED_BY_RUNTIME" in str(e):
                print("EXPERIMENT_BLOCKED_BY_RUNTIME", file=sys.stderr)
                sys.exit(0)  # Just exit for CI to pass when runtime isn't available
            raise
"""

replacement = """
    generator = None
    if args.provider == "json":
        if args.proposals is None:
            raise SystemExit("--provider json requires --proposals PATH")
        generator = JsonFileProposalProvider(args.proposals)
    elif args.provider == "llm":
        try:
            provider = LocalTransformersProvider()
            generator = LLMMutationGenerator(provider, "Improve the policy based on these keywords.")
            config["llm_provider"] = {
                "model_name": provider.model_name,
                "backend": "transformers",
                "seed": config.get("seed", 42),
                "decoding": "greedy (do_sample=False, num_return_sequences=1)"
            }
        except RuntimeError as e:
            if "EXPERIMENT_BLOCKED_BY_RUNTIME" in str(e):
                print("EXPERIMENT_BLOCKED_BY_RUNTIME", file=sys.stderr)
                sys.exit(0)  # Just exit for CI to pass when runtime isn't available
            raise
"""

content = content.replace(import_block, replacement)

with open("rsi_framework/cli.py", "w") as f:
    f.write(content)
