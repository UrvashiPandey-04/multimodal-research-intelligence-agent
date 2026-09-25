"""
Safe Mathematical Calculator Tool for numerical calculations without unrestricted Python eval().
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CalculationResult:
    """Dataclass representing the result of a safe mathematical calculation."""
    expression: str
    result: float
    formatted_result: str
    tool_type: str = "calculator"
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert calculation result to dictionary format."""
        return {
            "expression": self.expression,
            "result": self.result,
            "formatted_result": self.formatted_result,
            "tool_type": self.tool_type,
            "success": self.success,
            "error_message": self.error_message,
        }


class SafeCalculator:
    """Safe mathematical expression evaluator using AST whitelist parsing."""

    def __init__(self):
        """Initialize calculator tool."""
        pass

    def evaluate(self, expression: str) -> CalculationResult:
        """
        Evaluate a mathematical expression string safely.

        Args:
            expression: Expression string (e.g. "1.42 - 0.98", "(1.42 - 0.98) / 1.42 * 100", "25% of 1.42").

        Returns:
            CalculationResult object containing expression and evaluated float result.
        """
        clean_expr = expression.strip()
        if not clean_expr:
            return CalculationResult(
                expression=expression,
                result=0.0,
                formatted_result="0.0",
                success=False,
                error_message="Empty expression provided.",
            )

        # 1. Preprocess percentage phrases e.g. "25% of 100" -> "(25 / 100.0) * 100"
        normalized = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
            r"(\1 / 100.0) * \2",
            clean_expr,
            flags=re.IGNORECASE,
        )
        # 2. Replace lone percentages e.g. "25%" -> "(25 / 100.0)"
        normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", normalized)

        try:
            tree = ast.parse(normalized, mode="eval")
            val = self._eval_ast(tree.body)
            val_float = float(val)

            # Format result nicely (round float representation if needed)
            if val_float.is_integer():
                formatted = str(int(val_float))
            else:
                formatted = f"{val_float:.4f}".rstrip("0").rstrip(".")

            return CalculationResult(
                expression=clean_expr,
                result=val_float,
                formatted_result=formatted,
                success=True,
            )
        except ZeroDivisionError:
            return CalculationResult(
                expression=clean_expr,
                result=0.0,
                formatted_result="Error: Division by zero",
                success=False,
                error_message="Division by zero.",
            )
        except Exception as e:
            return CalculationResult(
                expression=clean_expr,
                result=0.0,
                formatted_result=f"Error: {e}",
                success=False,
                error_message=f"Failed to evaluate expression: {e}",
            )

    def _eval_ast(self, node: ast.AST) -> float:
        """Recursively evaluate AST nodes safely."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            elif isinstance(node.op, ast.Mod):
                return left % right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        else:
            raise ValueError(f"Unsupported expression construct: {type(node).__name__}")
