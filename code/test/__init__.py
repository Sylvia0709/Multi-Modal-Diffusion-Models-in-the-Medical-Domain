# 测试包初始化文件
# 使Python将test目录识别为一个包，便于导入和运行测试

from pathlib import Path
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))
