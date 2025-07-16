import os
from inference import SimpleInference

def run_all_demos():
    """
    Runs a series of inference demos for RoboBrain 2.0 to showcase its capabilities.
    """
    print("Initializing RoboBrain 2.0 (7B model)...")
    # Using the 7B model for a quick start, as it's faster and requires less VRAM.
    model = SimpleInference("BAAI/RoboBrain2.0-7B")
    print("Model initialized successfully.")

    # --- Task Definitions ---
    tasks = [
        {
            "task_name": "Visual Grounding (VG)",
            "prompt": "the person wearing a red hat",
            "image": "./assets/demo/grounding.jpg",
            "task_type": "grounding",
        },
        {
            "task_name": "Affordance Prediction",
            "prompt": "hold the cup",
            "image": "./assets/demo/affordance.jpg",
            "task_type": "affordance",
        },
        {
            "task_name": "Trajectory Prediction",
            "prompt": "reach for the banana on the plate",
            "image": "./assets/demo/trajectory.jpg",
            "task_type": "trajectory",
        },
        {
            "task_name": "Pointing Prediction",
            "prompt": "Identify several spots within the vacant space that's between the two mugs",
            "image": "./assets/demo/pointing.jpg",
            "task_type": "pointing",
        },
        {
            "task_name": "Navigation Task (Pointing)",
            "prompt": "Identify several spots within toilet in the house",
            "image": "./assets/demo/navigation.jpg",
            "task_type": "pointing",
        },
    ]

    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved in '{output_dir}/'")

    # --- Run Inference for Each Task ---
    for task in tasks:
        print("\n" + "="*50)
        print(f"🚀 Running Task: {task['task_name']}")
        print(f"   Prompt: '{task['prompt']}'")
        print(f"   Image: '{task['image']}'")
        print("="*50)

        output_filename = os.path.join(output_dir, f"output_{task['task_type']}_{os.path.basename(task['image'])}")
        
        pred = model.inference(
            prompt=task["prompt"],
            image_path=task["image"],
            task=task["task_type"],
            plot=True,
            save_path=output_filename,
            enable_thinking=True,
            do_sample=True
        )

        print("\n✅ Prediction Result:")
        print(pred)

    print("\n\n🎉 All demos completed!")

if __name__ == "__main__":
    run_all_demos() 