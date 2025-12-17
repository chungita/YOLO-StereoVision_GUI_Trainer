"""
預定義類別模組
從 predefined_classes.txt 文件讀取類別定義
"""

import os
from typing import List

def load_predefined_classes() -> List[str]:
    """
    從 predefined_classes.txt 文件加載預定義類別
    
    Returns:
        List[str]: 預定義類別列表
    """
    try:
        # 獲取當前文件所在目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        txt_file_path = os.path.join(current_dir, 'predefined_classes.txt')
        
        # 檢查文件是否存在
        if not os.path.exists(txt_file_path):
            print(f"Warning: predefined_classes.txt not found at {txt_file_path}")
            return ['drone', 'fixed wing', 'tree', 'ground']  # 默認類別
        
        # 讀取文件內容
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 解析類別（支持多行，每行一個類別）
        if content:
            classes = [line.strip() for line in content.split('\n') if line.strip()]
            return classes
        else:
            # 如果文件為空，返回默認類別
            return ['drone', 'fixed wing', 'tree', 'ground']
            
    except Exception as e:
        print(f"Error loading predefined classes: {e}")
        # 發生錯誤時返回默認類別
        return ['drone', 'fixed wing', 'tree', 'ground']

def get_predefined_classes_count() -> int:
    """
    獲取預定義類別數量
    
    Returns:
        int: 類別數量
    """
    return len(load_predefined_classes())

def print_predefined_classes():
    """
    打印預定義類別信息
    """
    classes = load_predefined_classes()
    print("預定義類別 (Predefined Classes):")
    print("=" * 40)
    for i, class_name in enumerate(classes):
        print(f"{i:2d}. {class_name}")
    print(f"\n總共 {len(classes)} 個類別")

if __name__ == '__main__':
    print_predefined_classes()
