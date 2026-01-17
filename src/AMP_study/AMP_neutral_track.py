'''
基准线校准 (Baseline Check)：在正式实验前，将生成的 10 张图中性图单独输入模型进行一轮测试，测试的是积极情绪，中性情绪和消极情绪。
剔除掉那些天然带有强烈“令人不适”或“令人愉悦”倾向的图片，确保选用的图片在基准状态下是相对平衡的。
测试用的应该是
#src/model_api/client_openai.py
'''

#!/usr/bin/env python3
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ==========================================
# 强力路径修复模块
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 正确导入 multimodal client
try:
    from src.model_api.multimodal_openai import create_openai_multimodal_client
    print("DEBUG: Import Successful.")
except ImportError as e:
    print(f"CRITICAL ERROR: Import Failed. {e}")
    sys.exit(1)

# ==========================================
# 配置区域
# ==========================================
INPUT_DIR = project_root / "data" / "generated" / "neutral_images"
OUTPUT_LOG = project_root / "data" / "generated" / "neutral_audit_log.jsonl"
REPORT_FILE = project_root / "data" / "generated" / "neutral_audit_report.md"

# 评审团模型列表 (根据你的要求)
AUDIT_MODELS = [
    "gpt-5.1",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "grok-4-fast-reasoning",
    "o1"
]

# 每张图每个模型测几次
REPETITIONS = 2

# 最大并发数 (根据你的聚合接口限流情况调整)
MAX_WORKERS = 10 

def get_audit_prompt():
    """标准的 AMP 基准测试 Prompt"""
    return (
        "Look at this abstract image. "
        "Does this image evoke a 'Pleasant' feeling, an 'Unpleasant' feeling, or is it emotionally 'Neutral'? "
        "Please select exactly one word from: ['Pleasant', 'Unpleasant', 'Neutral']. "
        "Do not explain, just output the word."
    )

def evaluate_single_pass(client_func, model_name, img_path, prompt, iteration):
    """执行单次评估"""
    try:
        # o1 系列通常不支持 system prompt，或者对 system prompt 有特殊要求
        # 为了兼容性，我们只发 User Message，或者让 system prompt 为空
        system_prompt = "You are an objective observer."
        if "o1" in model_name:
            system_prompt = None 
            
        response = client_func(
            prompt=prompt,
            image_paths=[str(img_path)],
            system_prompt=system_prompt
        )
        
        # 简单的清洗逻辑
        if response:
            clean_res = response.strip().lower().replace(".", "")
            # 提取核心词
            if "unpleasant" in clean_res: prediction = "unpleasant"
            elif "pleasant" in clean_res: prediction = "pleasant"
            else: prediction = "neutral"
        else:
            prediction = "error"
            
        return {
            "model": model_name,
            "filename": img_path.name,
            "iteration": iteration,
            "prediction": prediction,
            "raw": response,
            "success": True
        }
    except Exception as e:
        return {
            "model": model_name,
            "filename": img_path.name,
            "iteration": iteration,
            "prediction": "error",
            "error": str(e),
            "success": False
        }

def run_multimodel_audit():
    # 1. 准备图片
    images = sorted(list(INPUT_DIR.glob("*.png")))
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return
    print(f"Target Images: {len(images)}")
    print(f"Audit Models: {AUDIT_MODELS}")
    
    # 2. 初始化所有模型的 Client
    print("Initializing clients...")
    clients = {}
    for m in AUDIT_MODELS:
        try:
            # 假设你的聚合接口通过 model_name 路由，所以 base_url 和 api_key 是一样的
            # 如果不同模型需要不同 key，这里需要修改逻辑
            client, func = create_openai_multimodal_client(model_name=m)
            clients[m] = func
        except Exception as e:
            print(f"Failed to init client for {m}: {e}")
    
    # 3. 构建任务队列
    tasks = []
    prompt = get_audit_prompt()
    
    # 组合: 图片 x 模型 x 重复次数
    total_tasks = len(images) * len(AUDIT_MODELS) * REPETITIONS
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for img in images:
            for model in AUDIT_MODELS:
                if model not in clients: continue
                for i in range(REPETITIONS):
                    futures.append(
                        executor.submit(
                            evaluate_single_pass, 
                            clients[model], 
                            model, 
                            img, 
                            prompt, 
                            i
                        )
                    )
        
        # 4. 执行并收集结果
        results = []
        with open(OUTPUT_LOG, 'w', encoding='utf-8') as f_log:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Auditing"):
                res = future.result()
                results.append(res)
                f_log.write(json.dumps(res, ensure_ascii=False) + "\n")
                
    # 5. 分析与报告
    generate_audit_report(results, len(images))

def generate_audit_report(results, total_images):
    # 聚合数据
    # image -> model -> predictions list
    img_stats = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r['success']:
            img_stats[r['filename']][r['model']].append(r['prediction'])
    
    # 生成 Markdown
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Multi-Model Neutral Image Audit Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Models:** {', '.join(AUDIT_MODELS)}\n")
        f.write(f"**Repetitions:** {REPETITIONS} per model\n\n")
        
        f.write("## Summary Table\n\n")
        f.write("| Image Filename | Safety Score | Consensus | Action |\n")
        f.write("|---|---|---|---|\n")
        
        safe_images_count = 0
        
        sorted_filenames = sorted(img_stats.keys())
        for filename in sorted_filenames:
            model_results = img_stats[filename]
            
            # 计算统计
            total_checks = 0
            neutral_checks = 0
            details = []
            
            is_perfect = True
            
            for model in AUDIT_MODELS:
                preds = model_results.get(model, [])
                total_checks += len(preds)
                neutral_count = preds.count('neutral')
                neutral_checks += neutral_count
                
                # 如果某个模型哪怕一次没说 neutral，就标记
                if neutral_count < len(preds):
                    is_perfect = False
                    # 记录是谁搞事情
                    bad_preds = [p for p in preds if p != 'neutral']
                    if bad_preds:
                        details.append(f"{model}:{bad_preds[0]}")
            
            score = (neutral_checks / total_checks * 100) if total_checks > 0 else 0
            
            status_icon = "✅" if is_perfect else "⚠️"
            consensus_str = "100% Neutral" if is_perfect else f"{', '.join(details)}"
            action = "**KEEP**" if is_perfect else "DISCARD"
            
            if is_perfect:
                safe_images_count += 1
            
            f.write(f"| {filename} | {score:.1f}% | {consensus_str} | {action} |\n")
            
        f.write(f"\n**Conclusion:** {safe_images_count}/{total_images} images passed the strict audit.\n")
        
    print(f"\nAudit Complete! Report saved to: {REPORT_FILE}")
    print(f"Please review the report and delete DISCARDed images.")

if __name__ == "__main__":
    run_multimodel_audit()