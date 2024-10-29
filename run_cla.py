from ultralytics import YOLO
import csv

def run_classification(args):
    # 加载自定义模型
    model = YOLO(args.cla_model_path)

    # 输入文件夹路径
    results = model(args.cla_dir)

    # Output CSV file path
    output_csv = "./cla_pre.csv"

    # Write results to the CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])  # Write header

        # Iterate over results and write each image's id and predicted label
        for idx, result in enumerate(results, start=1):
            writer.writerow([idx, result.probs.top1 + 1])

    print(f"Results saved to: {output_csv}")