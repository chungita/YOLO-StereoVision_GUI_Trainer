from pathlib import Path
import shutil
import sys


def main() -> int:
    # 定位專案根目錄 / Locate project root
    project_root = Path(__file__).resolve().parent.parent
    stereo_dir = project_root / "Model_file" / "Stereo_Vision"

    if not stereo_dir.exists():
        print(f"[ERROR] 找不到資料夾 / Folder not found: {stereo_dir}")
        return 1

    pth_files = sorted(stereo_dir.glob("*.pth"))
    if not pth_files:
        print("[WARN] 未找到 .pth 檔案 / No .pth files found.")
        return 0

    print("準備開始轉換 / Prepare to start conversion...")
    print(f"來源資料夾 / Source folder: {stereo_dir}")

    converted_count = 0
    skipped_count = 0

    for src_path in pth_files:
        dst_path = src_path.with_suffix(".pt")

        # 若目標已存在，跳過以避免覆蓋 / Skip existing targets to avoid overwrite
        if dst_path.exists():
            print(f"[SKIP] 已存在，跳過 / Exists, skip: {dst_path.name}")
            skipped_count += 1
            continue

        # 直接複製（內容相同，只是副檔名不同）/ Direct copy (same bytes, different extension)
        shutil.copy2(src_path, dst_path)
        print(f"[OK] 複製 / Copied: {src_path.name} -> {dst_path.name}")
        converted_count += 1

    print("----")
    print(f"完成 / Done. 新增 / Created: {converted_count}, 跳過 / Skipped: {skipped_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


