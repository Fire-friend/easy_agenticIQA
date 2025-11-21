#!/usr/bin/env python
"""
验证IQA工具的权重下载和功能可用性

测试所有已注册的IQA工具，确保：
1. IQA-PyTorch可以创建工具实例
2. 工具权重已正确下载
3. 工具可以在测试图像上正常运行
4. 得分归一化正常工作

使用方法:
    python scripts/verify_iqa_tools.py                  # 测试所有工具
    python scripts/verify_iqa_tools.py --tools QAlign BRISQUE  # 测试指定工具
    python scripts/verify_iqa_tools.py --type NR        # 只测试NR工具
    python scripts/verify_iqa_tools.py --skip-slow      # 跳过慢速工具
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.tool_registry import ToolRegistry, ToolExecutionError

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToolVerifier:
    """IQA工具验证器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.results: Dict[str, Dict] = {}

        # 慢速工具列表（需要较长时间执行）
        self.slow_tools = {'HyperIQA', 'MUSIQ', 'MANIQA', 'TReS'}

    def create_test_images(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        创建测试图像对（原始图像和失真图像）

        Args:
            output_dir: 输出目录

        Returns:
            (reference_path, distorted_path) 元组
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建简单的测试图像 (256x256, RGB)
        size = (256, 256)

        # 参考图像：渐变色彩
        reference = Image.new('RGB', size)
        pixels = reference.load()
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = (
                    int(255 * i / size[0]),  # R
                    int(255 * j / size[1]),  # G
                    128                       # B
                )

        reference_path = output_dir / 'reference.png'
        reference.save(reference_path)

        # 失真图像：添加高斯噪声
        distorted_array = np.array(reference)
        noise = np.random.normal(0, 25, distorted_array.shape)
        distorted_array = np.clip(distorted_array + noise, 0, 255).astype(np.uint8)
        distorted = Image.fromarray(distorted_array)

        distorted_path = output_dir / 'distorted.png'
        distorted.save(distorted_path)

        logger.info(f"创建测试图像: {reference_path}, {distorted_path}")
        return reference_path, distorted_path

    def verify_tool(
        self,
        tool_name: str,
        test_image: Path,
        reference_image: Optional[Path] = None
    ) -> Dict:
        """
        验证单个IQA工具

        Args:
            tool_name: 工具名称
            test_image: 测试图像路径
            reference_image: 参考图像路径（FR工具需要）

        Returns:
            验证结果字典
        """
        result = {
            'tool': tool_name,
            'status': 'unknown',
            'error': None,
            'raw_score': None,
            'normalized_score': None,
            'execution_time': None
        }

        try:
            # 检查工具是否在注册表中
            if not self.registry.is_tool_available(tool_name):
                result['status'] = 'not_registered'
                result['error'] = f"Tool {tool_name} not found in registry"
                return result

            # 获取工具元数据
            metadata = self.registry.tools[tool_name]
            tool_type = metadata['type']

            logger.info(f"测试 {tool_name} ({tool_type})...")

            # FR工具需要参考图像
            if tool_type == 'FR' and reference_image is None:
                result['status'] = 'skipped'
                result['error'] = 'FR tool requires reference image'
                return result

            # 执行工具
            start_time = time.time()

            try:
                if tool_type == 'FR':
                    raw_score, normalized_score = self.registry.execute_tool(
                        tool_name,
                        str(test_image),
                        reference_path=str(reference_image)
                    )
                else:  # NR
                    raw_score, normalized_score = self.registry.execute_tool(
                        tool_name,
                        str(test_image)
                    )

                execution_time = time.time() - start_time

                # 验证得分
                if not np.isfinite(raw_score):
                    result['status'] = 'invalid_score'
                    result['error'] = f'Non-finite raw score: {raw_score}'
                    return result

                if not (1.0 <= normalized_score <= 5.0):
                    result['status'] = 'invalid_normalized'
                    result['error'] = f'Normalized score {normalized_score} out of range [1, 5]'
                    return result

                # 成功
                result['status'] = 'success'
                result['raw_score'] = float(raw_score)
                result['normalized_score'] = float(normalized_score)
                result['execution_time'] = execution_time

                logger.info(
                    f"  ✓ {tool_name}: raw={raw_score:.4f}, "
                    f"normalized={normalized_score:.2f}, time={execution_time:.2f}s"
                )

            except ImportError as e:
                result['status'] = 'missing_dependency'
                result['error'] = f'Missing dependency: {str(e)}'
                logger.error(f"  ✗ {tool_name}: {result['error']}")

            except ToolExecutionError as e:
                result['status'] = 'execution_error'
                result['error'] = str(e)
                logger.error(f"  ✗ {tool_name}: {result['error']}")

            except Exception as e:
                result['status'] = 'unknown_error'
                result['error'] = f'{type(e).__name__}: {str(e)}'
                logger.error(f"  ✗ {tool_name}: {result['error']}")

        except Exception as e:
            result['status'] = 'verification_error'
            result['error'] = f'Verification failed: {str(e)}'
            logger.error(f"  ✗ {tool_name}: {result['error']}")

        return result

    def verify_all_tools(
        self,
        tool_names: Optional[List[str]] = None,
        tool_type: Optional[str] = None,
        skip_slow: bool = False
    ) -> Dict[str, Dict]:
        """
        验证多个工具

        Args:
            tool_names: 要测试的工具列表（None表示测试所有）
            tool_type: 工具类型过滤 ("FR" 或 "NR")
            skip_slow: 是否跳过慢速工具

        Returns:
            验证结果字典 {tool_name: result}
        """
        # 创建测试图像
        test_dir = Path('/tmp/iqa_tool_test')
        reference_path, distorted_path = self.create_test_images(test_dir)

        # 确定要测试的工具
        if tool_names is None:
            # 获取所有工具（排除_comment等）
            all_tools = [
                name for name in self.registry.tools.keys()
                if not name.startswith('_')
            ]
        else:
            all_tools = tool_names

        # 应用类型过滤
        if tool_type:
            all_tools = [
                name for name in all_tools
                if self.registry.tools.get(name, {}).get('type') == tool_type
            ]

        # 应用慢速工具过滤
        if skip_slow:
            all_tools = [
                name for name in all_tools
                if name not in self.slow_tools
            ]

        logger.info(f"\n开始验证 {len(all_tools)} 个工具...\n")

        # 验证每个工具
        results = {}
        for i, tool_name in enumerate(all_tools, 1):
            logger.info(f"[{i}/{len(all_tools)}] {tool_name}")

            result = self.verify_tool(
                tool_name,
                test_image=distorted_path,
                reference_image=reference_path
            )

            results[tool_name] = result

        # 清理测试图像
        try:
            reference_path.unlink()
            distorted_path.unlink()
            test_dir.rmdir()
        except Exception:
            pass

        return results

    def print_summary(self, results: Dict[str, Dict]) -> None:
        """打印验证摘要"""
        total = len(results)
        success = sum(1 for r in results.values() if r['status'] == 'success')
        failed = total - success

        # 按状态分组
        by_status = {}
        for tool_name, result in results.items():
            status = result['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(tool_name)

        # 打印摘要
        print("\n" + "=" * 70)
        print("验证摘要")
        print("=" * 70)
        print(f"总计: {total} 个工具")
        print(f"成功: {success} 个工具 ({success/total*100:.1f}%)")
        print(f"失败: {failed} 个工具 ({failed/total*100:.1f}%)")
        print()

        # 打印各状态详情
        for status, tools in sorted(by_status.items()):
            if status == 'success':
                print(f"✓ 成功 ({len(tools)}):")
                for tool in sorted(tools):
                    r = results[tool]
                    print(f"  - {tool}: normalized={r['normalized_score']:.2f}, time={r['execution_time']:.2f}s")
            else:
                print(f"✗ {status.replace('_', ' ').title()} ({len(tools)}):")
                for tool in sorted(tools):
                    error_msg = results[tool].get('error', 'Unknown error')
                    # 截断过长的错误消息
                    if len(error_msg) > 100:
                        error_msg = error_msg[:97] + "..."
                    print(f"  - {tool}: {error_msg}")
            print()

        # 执行时间统计（仅成功的工具）
        if 'success' in by_status:
            execution_times = [
                (tool, results[tool]['execution_time'])
                for tool in by_status['success']
            ]
            execution_times.sort(key=lambda x: x[1], reverse=True)

            print("执行时间排名 (Top 5 慢速工具):")
            for tool, exec_time in execution_times[:5]:
                print(f"  - {tool}: {exec_time:.2f}s")
            print()

        print("=" * 70)

        # 返回退出码
        return 0 if failed == 0 else 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证IQA工具的权重下载和功能可用性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有工具
  python scripts/verify_iqa_tools.py

  # 测试指定工具
  python scripts/verify_iqa_tools.py --tools QAlign BRISQUE TOPIQ_FR

  # 只测试NR工具
  python scripts/verify_iqa_tools.py --type NR

  # 跳过慢速工具
  python scripts/verify_iqa_tools.py --skip-slow

  # 组合使用
  python scripts/verify_iqa_tools.py --type FR --skip-slow
        """
    )

    parser.add_argument(
        '--tools',
        nargs='+',
        help='指定要测试的工具名称（默认测试所有）'
    )

    parser.add_argument(
        '--type',
        choices=['FR', 'NR'],
        help='只测试指定类型的工具 (FR: Full-Reference, NR: No-Reference)'
    )

    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='跳过已知的慢速工具 (HyperIQA, MUSIQ, MANIQA, TReS)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # 加载工具注册表
    try:
        logger.info("加载工具注册表...")
        registry = ToolRegistry()
        total_tools = len([k for k in registry.tools.keys() if not k.startswith('_')])
        logger.info(f"已加载 {total_tools} 个工具")
    except Exception as e:
        logger.error(f"无法加载工具注册表: {e}")
        return 1

    # 验证工具
    verifier = ToolVerifier(registry)

    try:
        results = verifier.verify_all_tools(
            tool_names=args.tools,
            tool_type=args.type,
            skip_slow=args.skip_slow
        )

        # 打印摘要
        exit_code = verifier.print_summary(results)

        return exit_code

    except KeyboardInterrupt:
        logger.warning("\n用户中断测试")
        return 130

    except Exception as e:
        logger.error(f"验证过程发生错误: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
