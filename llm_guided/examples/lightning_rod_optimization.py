#!/usr/bin/env python3
"""
⚡ Lightning Rod Placement Optimization

A non-ML geometric optimization problem where we need to optimally place
lightning rods around a facility to minimize the maximum unprotected distance
while respecting various real-world constraints.

Problem: Given a rectangular facility (100m x 80m) with several buildings,
place 3-5 lightning rods optimally to minimize the worst-case distance
from any point to the nearest rod's protection zone.

Constraints:
- Lightning rods have 45m protection radius (cone of protection)
- Cannot place rods within 10m of buildings
- Cannot place rods within 15m of each other (electrical interference)
- Each rod costs $5000, total budget $25000 (max 5 rods)
- Rods must be at least 5m from facility edges

This has no analytical solution due to:
- Non-convex optimization landscape
- Multiple local optima
- Complex geometric constraints
- Discrete-continuous hybrid problem (number of rods + positions)

Usage:
  OPENAI_API_KEY=your-key python examples/lightning_rod_optimization.py
  python examples/lightning_rod_optimization.py --show-full-llm   # Show full LLM responses  
  python examples/lightning_rod_optimization.py --model gpt-4o-mini --trials 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import optuna
from sampler import LLMGuidedSampler
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - visualizations disabled")
import logging
import time
import argparse

# Configure minimal logging - turn off debug spam
logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress debug logging from LLM components
logging.getLogger('llm_client').setLevel(logging.WARNING)
logging.getLogger('sampler').setLevel(logging.WARNING)
logging.getLogger('context_builder').setLevel(logging.WARNING)
logging.getLogger('parameter_validator').setLevel(logging.WARNING)

class LightningRodOptimizer:
    def __init__(self):
        # Facility dimensions
        self.facility_width = 100.0  # meters
        self.facility_height = 80.0  # meters
        
        # Lightning rod specifications
        self.rod_protection_radius = 45.0  # meters
        self.rod_cost = 5000  # dollars
        self.max_budget = 25000  # dollars
        self.min_rod_spacing = 15.0  # meters (interference)
        self.edge_clearance = 5.0  # meters from facility edge
        
        # Buildings (obstacles where rods cannot be placed)
        self.buildings = [
            {"x": 20, "y": 15, "width": 25, "height": 20, "name": "Main Building"},
            {"x": 60, "y": 40, "width": 15, "height": 25, "name": "Storage"},
            {"x": 10, "y": 55, "width": 30, "height": 15, "name": "Workshop"},
            {"x": 75, "y": 10, "width": 20, "height": 18, "name": "Office"}
        ]
        self.building_clearance = 10.0  # meters from buildings
        
        # Create high-resolution grid for coverage calculation
        self.grid_resolution = 2.0  # meters
        self.x_points = np.arange(0, self.facility_width + self.grid_resolution, self.grid_resolution)
        self.y_points = np.arange(0, self.facility_height + self.grid_resolution, self.grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(self.x_points, self.y_points)
        
    def is_valid_rod_position(self, x, y):
        """Check if rod position satisfies all constraints."""
        # Edge clearance
        if (x < self.edge_clearance or x > self.facility_width - self.edge_clearance or
            y < self.edge_clearance or y > self.facility_height - self.edge_clearance):
            return False
            
        # Building clearance
        for building in self.buildings:
            bx, by = building["x"], building["y"]
            bw, bh = building["width"], building["height"]
            
            # Distance to building rectangle
            dx = max(0, max(bx - x, x - (bx + bw)))
            dy = max(0, max(by - y, y - (by + bh)))
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < self.building_clearance:
                return False
                
        return True
    
    def check_rod_spacing(self, positions):
        """Check minimum spacing between rods."""
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:  # Avoid double checking
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if distance < self.min_rod_spacing:
                        return False
        return True
    
    def calculate_coverage_quality(self, rod_positions):
        """Calculate coverage quality - lower is better (minimizing max unprotected distance)."""
        if not rod_positions:
            return 1000.0  # No rods = very bad
            
        # Validate all rod positions
        valid_positions = []
        for x, y in rod_positions:
            if self.is_valid_rod_position(x, y):
                valid_positions.append((x, y))
        
        if not valid_positions:
            return 1000.0  # No valid rods
            
        if not self.check_rod_spacing(valid_positions):
            return 1000.0  # Spacing violation
        
        # Calculate protection coverage for each grid point
        max_unprotected_distance = 0.0
        total_unprotected_area = 0.0
        
        for i, x in enumerate(self.x_points):
            for j, y in enumerate(self.y_points):
                # Find distance to nearest rod
                min_distance_to_rod = float('inf')
                
                for rod_x, rod_y in valid_positions:
                    distance = np.sqrt((x - rod_x)**2 + (y - rod_y)**2)
                    min_distance_to_rod = min(min_distance_to_rod, distance)
                
                # Check if point is protected (within rod radius)
                if min_distance_to_rod <= self.rod_protection_radius:
                    unprotected_distance = 0.0
                else:
                    # Distance beyond protection radius
                    unprotected_distance = min_distance_to_rod - self.rod_protection_radius
                
                max_unprotected_distance = max(max_unprotected_distance, unprotected_distance)
                total_unprotected_area += unprotected_distance * (self.grid_resolution ** 2)
        
        # Combined objective: minimize maximum unprotected distance + total unprotected area
        # Weight max distance heavily since it's critical safety metric
        objective = max_unprotected_distance * 10.0 + total_unprotected_area * 0.001
        
        return objective
    
    def visualize_solution(self, rod_positions, objective_value, trial_number):
        """Create visualization of the rod placement solution."""
        if not HAS_MATPLOTLIB:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw facility boundary
        facility_rect = plt.Rectangle((0, 0), self.facility_width, self.facility_height, 
                                    fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(facility_rect)
        
        # Draw buildings
        for building in self.buildings:
            building_rect = plt.Rectangle(
                (building["x"], building["y"]), building["width"], building["height"],
                fill=True, facecolor='gray', alpha=0.7, edgecolor='black'
            )
            ax.add_patch(building_rect)
            # Label buildings
            ax.text(building["x"] + building["width"]/2, building["y"] + building["height"]/2,
                   building["name"], ha='center', va='center', fontsize=8)
        
        # Draw protection zones and rods
        valid_positions = [(x, y) for x, y in rod_positions if self.is_valid_rod_position(x, y)]
        
        for i, (rod_x, rod_y) in enumerate(valid_positions):
            # Protection circle
            protection_circle = plt.Circle((rod_x, rod_y), self.rod_protection_radius,
                                         fill=False, color='blue', alpha=0.3, linestyle='--')
            ax.add_patch(protection_circle)
            
            # Rod position
            ax.plot(rod_x, rod_y, 'ro', markersize=10, label=f'Rod {i+1}' if i == 0 else "")
            ax.text(rod_x + 2, rod_y + 2, f'R{i+1}', fontsize=10, fontweight='bold')
        
        # Create coverage heatmap
        coverage_grid = np.zeros_like(self.grid_x)
        for i in range(len(self.x_points)):
            for j in range(len(self.y_points)):
                x, y = self.x_points[i], self.y_points[j]
                min_distance = float('inf')
                for rod_x, rod_y in valid_positions:
                    distance = np.sqrt((x - rod_x)**2 + (y - rod_y)**2)
                    min_distance = min(min_distance, distance)
                
                if min_distance <= self.rod_protection_radius:
                    coverage_grid[j, i] = 0  # Protected
                else:
                    coverage_grid[j, i] = min_distance - self.rod_protection_radius  # Unprotected distance
        
        # Plot coverage heatmap
        im = ax.contourf(self.grid_x, self.grid_y, coverage_grid, 
                        levels=20, cmap='Reds', alpha=0.6)
        plt.colorbar(im, ax=ax, label='Unprotected Distance (m)')
        
        ax.set_xlim(-5, self.facility_width + 5)
        ax.set_ylim(-5, self.facility_height + 5)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Trial {trial_number}: Lightning Rod Placement\n'
                    f'Objective: {objective_value:.2f}, Rods: {len(valid_positions)}, '
                    f'Cost: ${len(valid_positions) * self.rod_cost:,}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'lightning_rods_trial_{trial_number}.png', dpi=150, bbox_inches='tight')
        plt.close()

def lightning_rod_objective(trial):
    """Objective function for lightning rod optimization."""
    optimizer = LightningRodOptimizer()
    
    # Number of rods (discrete choice within budget)
    num_rods = trial.suggest_int("num_rods", 3, 5)
    
    # Rod positions - each rod has (x, y) coordinates
    # ALWAYS suggest all 5 possible rod positions to ensure consistent search space
    rod_positions = []
    for i in range(5):  # Always suggest 5 rods for consistent search space
        x = trial.suggest_float(f"rod_{i}_x", 
                               optimizer.edge_clearance, 
                               optimizer.facility_width - optimizer.edge_clearance)
        y = trial.suggest_float(f"rod_{i}_y", 
                               optimizer.edge_clearance, 
                               optimizer.facility_height - optimizer.edge_clearance)
        if i < num_rods:  # Only use the first num_rods positions
            rod_positions.append((x, y))
    
    # Calculate objective
    objective_value = optimizer.calculate_coverage_quality(rod_positions)
    
    # Print concise trial info
    global trial_display_count
    if trial.number < 2:  # Baseline trials  
        print(f"📍 Baseline {trial.number + 1}: {num_rods} rods → Score: {objective_value:.1f}")
    else:  # LLM trials
        trial_display_count += 1
        print(f"📍 Trial {trial_display_count}/{args.trials if 'args' in globals() else 'N'}: {num_rods} rods → Score: {objective_value:.1f}")
    
    # Create visualization for good solutions
    if objective_value < 100:
        try:
            optimizer.visualize_solution(rod_positions, objective_value, trial.number)
        except Exception:
            pass
    
    return objective_value

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lightning Rod LLM Optimization")
    parser.add_argument("--show-full-llm", action="store_true", help="Show full LLM responses for examination")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="LLM model to use")
    parser.add_argument("--trials", type=int, default=6, help="Number of LLM trials to run")
    return parser.parse_args()

def main():
    """Run lightning rod optimization with LLM guidance."""
    
    # Check API key FIRST before doing anything
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: No API key found.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        print("Example: OPENAI_API_KEY=your-key python examples/lightning_rod_optimization.py")
        sys.exit(1)
    
    args = parse_args()
    
    print("⚡ Lightning Rod LLM Optimization")
    print(f"🤖 Model: {args.model} | Trials: {args.trials}")
    if args.show_full_llm:
        print("📋 Full LLM responses will be displayed")
    
    print("\n🔧 Running 2 baseline trials...")
    
    # First, run a couple baseline trials with default sampler to establish search space
    baseline_study = optuna.create_study(direction="minimize")
    baseline_study.optimize(lightning_rod_objective, n_trials=2, show_progress_bar=False)
    
    print("\n🚀 Starting LLM-guided trials...")
    
    # Create LLM sampler 
    sampler = LLMGuidedSampler(
        model=args.model,
        temperature=0.4,  # Moderate creativity for spatial reasoning
        timeout=45,       # Longer timeout for complex geometric reasoning
        max_retries=3,
        max_context_trials=8
    )
    
    # Create new study with comprehensive context and use baseline trials
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Copy baseline trials to new study to provide search space context
    for trial in baseline_study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            study.add_trial(trial)
    
    # Rich context for LLM reasoning
    study.set_user_attr("problem_type", "geometric_optimization")
    study.set_user_attr(
        "problem_description", 
        "Lightning rod placement optimization for electrical safety. Must place 3-5 rods "
        "optimally in 100m×80m facility to minimize maximum unprotected distance. "
        "Each rod provides 45m radius protection zone."
    )
    
    study.set_user_attr("constraints", [
        "facility_size: 100m × 80m rectangular area",
        "rod_protection: 45m radius protection zone per rod",
        "rod_cost: $5,000 each, max budget $25,000 (3-5 rods)",
        "edge_clearance: 5m minimum from facility boundaries", 
        "building_clearance: 10m minimum from buildings",
        "rod_spacing: 15m minimum between rods (electrical interference)",
        "buildings: 4 rectangular obstacles in facility"
    ])
    
    study.set_user_attr("domain_knowledge", {
        "objective": "minimize maximum distance from any point to nearest protection zone",
        "facility_layout": {
            "dimensions": "100m × 80m",
            "buildings": [
                "Main Building: 25×20m at (20,15)",
                "Storage: 15×25m at (60,40)", 
                "Workshop: 30×15m at (10,55)",
                "Office: 20×18m at (75,10)"
            ]
        },
        "protection_physics": "Lightning rods create circular protection zones with 45m radius",
        "optimization_insights": [
            "Corner placement often inefficient due to wasted coverage outside facility",
            "Central placement maximizes coverage but may conflict with buildings", 
            "Rod clustering reduces total coverage due to overlap",
            "Building shadows create coverage gaps requiring strategic placement",
            "Edge effects: coverage near facility boundaries is critical"
        ],
        "geometric_strategies": [
            "Triangle/pentagon arrangements often optimal for multiple rods",
            "Balance coverage overlap vs gap elimination",
            "Consider building positions as exclusion zones",
            "Facility aspect ratio (5:4) suggests non-square arrangements"
        ],
        "typical_good_solutions": "3 rods: ~40-60 objective, 4 rods: ~25-40, 5 rods: ~15-30"
    })
    
    study.set_user_attr("strategy_guidance", {
        "early_trials": "Explore different rod counts and basic geometric patterns",
        "middle_trials": "Refine promising arrangements, balance coverage vs cost",
        "later_trials": "Fine-tune positions for optimal coverage with constraint satisfaction"
    })
    
    # Global trial counter for display
    global trial_display_count
    trial_display_count = 0
    
    try:
        # Patch to show LLM recommendations 
        from unittest.mock import patch
        original_method = sampler.llm_client.generate_trial_configuration
        
        def show_llm_rec(context, temperature=None):
            response = original_method(context, temperature)
            num_rods = response.parameters.get('num_rods', 'Unknown')
            print(f"🧠 LLM rec: {num_rods} rods")
            return response

        if args.show_full_llm:
            
            def show_full_llm_response(context, temperature=None):
                print(f"\n🤖 FULL LLM CALL for Trial {len(study.trials)}:")
                print("=" * 60)
                
                # Show the prompt being sent
                prompt = sampler.llm_client._build_configuration_prompt(context)
                print("📤 PROMPT SENT TO LLM:")
                print("-" * 30)
                print(prompt)
                print("-" * 30)
                
                # Call the real LLM
                response = original_method(context, temperature)
                
                print("📥 LLM RESPONSE:")
                print("-" * 30)
                print(f"Parameters: {response.parameters}")
                print(f"Reasoning: {response.reasoning}")
                print(f"Confidence: {response.confidence}")
                print(f"Strategy: {response.strategy}")
                if response.expected_performance:
                    print(f"Expected Performance: {response.expected_performance}")
                print("=" * 60)
                
                return response
            
            with patch.object(sampler.llm_client, 'generate_trial_configuration', show_full_llm_response):
                study.optimize(
                    lightning_rod_objective, 
                    n_trials=args.trials
                )
        else:
            # Show concise LLM recommendations
            with patch.object(sampler.llm_client, 'generate_trial_configuration', show_llm_rec):
                study.optimize(
                    lightning_rod_objective, 
                    n_trials=args.trials
                )
        
        print(f"\n🏆 OPTIMIZATION RESULTS")
        print("=" * 60)
        
        best_trial = study.best_trial
        print(f"🥇 Best Trial: #{best_trial.number}")
        print(f"📊 Best Objective: {study.best_value:.3f}")
        print(f"💰 Best Cost: ${best_trial.params['num_rods'] * 5000:,}")
        
        print(f"\n🎯 Best Rod Configuration:")
        num_rods = best_trial.params['num_rods']
        for i in range(num_rods):
            x = best_trial.params[f'rod_{i}_x']
            y = best_trial.params[f'rod_{i}_y']
            print(f"   Rod {i+1}: ({x:.1f}m, {y:.1f}m)")
        
        # Show trial progression
        print(f"\n📈 Trial Progression:")
        best_so_far = float('inf')
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if trial.value < best_so_far:
                    best_so_far = trial.value
                    marker = "🥇" if trial.number == best_trial.number else "📈"
                else:
                    marker = "  "
                print(f"   {marker} Trial {trial.number}: {trial.value:.3f} "
                      f"({trial.params['num_rods']} rods)")
        
        # Sampler performance statistics
        stats = sampler.get_statistics()
        print(f"\n🤖 LLM Sampler Performance:")
        print(f"   Success Rate: {stats['sampler_stats']['success_rate']:.1%}")
        print(f"   Avg LLM Time: {stats['sampler_stats']['average_llm_time']:.1f}s")
        print(f"   Total Generations: {stats['sampler_stats']['successful_generations']}")
        print(f"   Fallback Uses: {stats['sampler_stats']['fallback_uses']}")
        
        # Create final best solution visualization
        optimizer = LightningRodOptimizer()
        best_positions = []
        for i in range(num_rods):
            x = best_trial.params[f'rod_{i}_x']
            y = best_trial.params[f'rod_{i}_y']
            best_positions.append((x, y))
        
        optimizer.visualize_solution(best_positions, study.best_value, "BEST")
        print(f"\n📊 Best solution visualization: lightning_rods_trial_BEST.png")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        
        # Show any completed trials
        if study.trials:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_completed = min(completed_trials, key=lambda t: t.value)
                print(f"\n📊 Best from {len(completed_trials)} completed trials:")
                print(f"   Value: {best_completed.value:.3f}")
                print(f"   Rods: {best_completed.params['num_rods']}")

if __name__ == "__main__":
    main()