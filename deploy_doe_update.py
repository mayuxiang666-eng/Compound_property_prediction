# -*- coding: utf-8 -*-
"""
Deploy updated DOE Interface with All-Batches Distribution Chart and Detailed Batch Table
to remote production server \\10.246.97.159\D\HefeiEnergyDashboard_Release\run_server\
"""
import shutil
import os
import re

REMOTE_DIR = r"\\10.246.97.159\D\HefeiEnergyDashboard_Release\run_server"

def deploy():
    print("1. Copying energy_optimizer_module.js...")
    shutil.copy2("energy_optimizer_module.js", os.path.join(REMOTE_DIR, "energy_optimizer_module.js"))
    print("   energy_optimizer_module.js copied.")

    print("2. Copying doe_orders_database.json...")
    shutil.copy2("doe_orders_database.json", os.path.join(REMOTE_DIR, "doe_orders_database.json"))
    print("   doe_orders_database.json copied.")

    print("3. Injecting energy_optimizer_section.html into remote index.html...")
    with open("energy_optimizer_section.html", "r", encoding="utf-8") as f:
        section_html = f.read().strip()

    remote_index_path = os.path.join(REMOTE_DIR, "index.html")
    with open(remote_index_path, "r", encoding="utf-8") as f:
        index_content = f.read()

    start_marker = "<!-- MODULE 3: 胶料节能潜力排行与实测 DOE 优化工作台 (POTENTIAL RANKING & DOE WORKBENCH) -->"
    end_marker = "</div> <!-- /main-dashboard-wrapper -->"

    idx_start = index_content.find(start_marker)
    idx_end = index_content.find(end_marker)

    if idx_start == -1 or idx_end == -1:
        raise ValueError(f"Could not locate markers: idx_start={idx_start}, idx_end={idx_end}")

    new_index_content = index_content[:idx_start] + section_html + "\n\n\n        " + index_content[idx_end:]

    # Bump cache-busting version for energy_optimizer_module.js
    new_index_content = re.sub(
        r'src="energy_optimizer_module\.js\?v=[^"]*"',
        'src="energy_optimizer_module.js?v=20261019-doe-v3"',
        new_index_content
    )

    with open(remote_index_path, "w", encoding="utf-8") as f:
        f.write(new_index_content)
    print("   index.html successfully updated and saved.")

    print("\n[OK] Deployment completed successfully!")

if __name__ == "__main__":
    deploy()
