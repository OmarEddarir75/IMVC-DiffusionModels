import yaml
import pprint

# -------------------------------
# Load the YAML file
# -------------------------------
yaml_path = "agent_guide.yaml"

try:
    with open(yaml_path, "r") as f:
        guide = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: {yaml_path} not found. Make sure it's in your workspace.")
    exit(1)

# -------------------------------
# Function to print tasks nicely
# -------------------------------
def print_section(title, content, indent=0):
    prefix = " " * indent
    print(f"{prefix}{title}:")
    if isinstance(content, dict):
        for k, v in content.items():
            print_section(k, v, indent + 2)
    elif isinstance(content, list):
        for i, item in enumerate(content, 1):
            if isinstance(item, (dict, list)):
                print_section(f"- Item {i}", item, indent + 2)
            else:
                print(f"{prefix}  - {item}")
    else:
        print(f"{prefix}  {content}")

# -------------------------------
# Print top-level guide
# -------------------------------
print("\n===== AGENT REFACTORING PLAN =====\n")
for key, value in guide.items():
    print_section(key.upper(), value)
    
# -------------------------------
# Generate a simple step-by-step checklist
# -------------------------------
print("\n===== STEP-BY-STEP ACTIONABLE CHECKLIST =====\n")

checklist = [
    "1. Identify all 2-view-specific code in target files.",
    "2. Refactor view encoding to handle V views (replace x1/x2 with loops/lists).",
    "3. Update attention/fusion layers to multi-view, masked operations, avoid O(V^2).",
    "4. Refactor diffusion: one UNet per view, condition on latent presence & view index.",
    "5. Update high-confidence, contrastive, and MMI losses for V views.",
    "6. Update dataset & masking utilities to produce [N, V] masks and ensure at least one view per sample.",
    "7. Refactor training loops using ModuleList and normalize branch losses.",
    "8. Save source_mode/view structure for consistent evaluation.",
    "9. Run smoke test with V > 2, check NaNs, finite losses, stable gradients.",
    "10. Log per-view statistics and verify recovered latents for missing views."
]

for item in checklist:
    print(f"- {item}")

print("\n✅ Use this plan as a reference for Copilot Pro suggestions in your VS Code workspace.\n")