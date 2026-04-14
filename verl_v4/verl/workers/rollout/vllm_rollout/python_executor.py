# From Tool-Star repo

import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import time

import multiprocess

# Set the start method to 'spawn' to avoid forking the large model process.
# 'force=True' is needed if the method has already been set implicitly.
try:
    multiprocess.set_start_method('spawn', force=True)
except RuntimeError:
    print("Multiprocess start method already set. This is okay if it's already 'spawn'.")


from multiprocess import Pool
from typing import Any, Dict, Optional, List, Tuple
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
from scipy import optimize
from functools import wraps

import resource

# Check if in Ray environment
def _is_in_ray():
    try:
        import ray
        return ray.is_initialized()
    except ImportError:
        return False

def set_memory_limit(max_mem_mb):
    max_bytes = max_mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

def memory_limited_executor(code, executor_fn, memory_limit_mb):
    set_memory_limit(memory_limit_mb)
    return executor_fn(code)

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        
        # Do not perform any imports during initialization
        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        # Preprocessing: add necessary imports before executing user code
        imports = """
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
x, y, z = sympy.symbols('x y z')
"""
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
            
        # Execute import statements first
        exec(imports, self._global_vars)
        # Then execute user code
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 2,
        enable_multiple: bool = True,  # Default to enable parallel execution
        use_ray_tasks: bool = None,  # Automatically detect Ray environment, use Tasks in Ray, otherwise use ProcessPool
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length
        self.enable_multiple = enable_multiple
        
        # Automatically detect if using Ray Tasks
        if use_ray_tasks is None:
            self.use_ray_tasks = _is_in_ray()
        else:
            self.use_ray_tasks = use_ray_tasks
        
        # If in Ray environment and Ray Tasks is enabled, do not use ProcessPool
        # if self.use_ray_tasks:
        #     print("🚀 PythonExecutor: Using Ray Tasks for distributed execution")
        # else:
        #     print("🔧 PythonExecutor: Using ProcessPool for local execution")

    def __del__(self):
        """Clean up resources (if any)"""
        # Ray Tasks will be automatically cleaned up, no need to handle manually
        pass

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = True,
        timeout_length = 1,
    ):
        try:
            # Ensure code is a string rather than a list
            if isinstance(code, list):
                code = '\n'.join(code)
            
            # Remove all leading whitespace
            code = code.strip()
            
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)(code)
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                # Separate the last line as an expression
                code_lines = code.split('\n')
                if len(code_lines) > 1:
                    exec_code = '\n'.join(code_lines[:-1])
                    eval_code = code_lines[-1]
                    timeout(timeout_length)(runtime.exec_code)(exec_code)
                    result = timeout(timeout_length)(runtime.eval_code)(eval_code)
                else:
                    result = timeout(timeout_length)(runtime.eval_code)(code)
                    
            report = "Done"
            
            # Safely handle the result
            try:
                # Try to serialize
                pickle.dumps(result)
            except (pickle.PicklingError, TypeError):
                # If cannot serialize, convert to string
                try:
                    result = str(result)
                except:
                    # If even string conversion fails, return type information
                    result = f"<unprintable object of type {type(result).__name__}>"
            
        except Exception as e:
            result = ''
            report = str(e)
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        start_time = time.time()
        all_code_snippets = self.process_generation_to_code(batch_code)
        timeout_cnt = 0
        all_exec_results = []

        # Optimized for H100 nodes
        if self.use_ray_tasks:
            # Use Ray Tasks - optimized for H100
            MAX_RAY_TASKS = 50  # H100 nodes have enough memory, can support more parallel execution
            max_workers = min(len(all_code_snippets), MAX_RAY_TASKS)
            execution_mode = "ray_tasks"
        else:
            # Traditional ProcessPool mode
            MAX_PARALLEL_WORKERS = 3
            max_workers = min(len(all_code_snippets), MAX_PARALLEL_WORKERS)
            
            # Determine execution mode
            if not self.enable_multiple:
                execution_mode = "sequential"
                max_workers = 1
            else:
                execution_mode = "sequential" if max_workers <= 1 or len(all_code_snippets) == 1 else "parallel"
        
        # print(f"🔧 Python Executor: {len(all_code_snippets)} tasks, mode={execution_mode}, max_workers={max_workers}")
        
        # Execution strategy selection
        if execution_mode == "ray_tasks":
            # Ray Tasks execution mode
            all_exec_results = self._execute_with_ray_tasks(all_code_snippets, max_workers)
            
        elif execution_mode == "sequential":
            # Sequential execution: single task or not using multiple processes
            for code in all_code_snippets:
                result, report = self.execute(
                    code,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime=self.runtime,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )
                all_exec_results.append((result, report))
        else:
            # Traditional ProcessPool parallel execution
            all_exec_results = self._execute_with_process_pool(all_code_snippets, max_workers)

        total_time = time.time() - start_time
        # print(f"✅ Python Executor completed: {total_time:.3f}s, {len(all_exec_results)} results")

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results
    
    def _execute_with_ray_tasks(self, all_code_snippets: List[str], max_workers: int) -> List[Tuple[str, str]]:
        """Use Ray Tasks to execute code snippets"""
        import ray
        
        # Prepare Ray Task parameters
        runtime_type = "generic"
        if isinstance(self.runtime, DateRuntime):
            runtime_type = "date"
        elif isinstance(self.runtime, ColorObjectRuntime):
            runtime_type = "color"
        
        try:
            # Create Ray Tasks
            futures = []
            for code in all_code_snippets:
                future = execute_code_ray_task.remote(
                    code=code,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime_type=runtime_type,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )
                futures.append(future)
            
            # Show progress bar (only when there are many tasks)
            if len(all_code_snippets) > 10:
                # print(f"🚀 Submitting {len(all_code_snippets)} Ray Tasks...")
                progress_bar = tqdm(total=len(all_code_snippets), desc="Ray Tasks")
            else:
                progress_bar = None
            
            # Get results - keep original order
            all_exec_results = []
            for i, future in enumerate(futures):
                try:
                    result = ray.get(future)
                    all_exec_results.append(result)
                    
                    if progress_bar is not None:
                        progress_bar.update(1)
                        
                except Exception as e:
                    # print(f"❌ Ray Task {i} error: {e}")
                    all_exec_results.append(("", f"Ray Task Error: {str(e)}"))
                    
                    if progress_bar is not None:
                        progress_bar.update(1)
            
            if progress_bar is not None:
                progress_bar.close()
            
            return all_exec_results
            
        except Exception as e:
            # print(f"🚨 Ray Tasks failed, falling back to sequential execution: {e}")
            # Fallback to sequential execution
            all_exec_results = []
            for code in all_code_snippets:
                result, report = self.execute(
                    code,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime=self.runtime,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )
                all_exec_results.append((result, report))
            return all_exec_results
    
    def _execute_with_process_pool(self, all_code_snippets: List[str], max_workers: int) -> List[Tuple[str, str]]:
        """Use traditional ProcessPool to execute code snippets"""
        MEMORY_LIMIT_MB = 200
        
        executor_fn = partial(
            self.execute,
            get_answer_from_stdout=self.get_answer_from_stdout,
            runtime=self.runtime,
            answer_symbol=self.answer_symbol,
            answer_expr=self.answer_expr,
            timeout_length=self.timeout_length,
        )

        try:
            with ProcessPool(max_workers=max_workers) as pool:
                wrapped_executor = partial(
                    memory_limited_executor, 
                    executor_fn=executor_fn, 
                    memory_limit_mb=MEMORY_LIMIT_MB
                )
                future = pool.map(wrapped_executor, all_code_snippets, timeout=self.timeout_length)
                iterator = future.result()

                # Progress bar: only show when there are many tasks
                if len(all_code_snippets) > 10:
                    progress_bar = tqdm(total=len(all_code_snippets), desc="ProcessPool")
                else:
                    progress_bar = None

                all_exec_results = []
                while True:
                    try:
                        result = next(iterator)
                        all_exec_results.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        # print(f"⏰ Timeout error: {error}")
                        all_exec_results.append(("", "Timeout Error"))
                    except Exception as error:
                        # print(f"❌ Execution error: {error}")
                        all_exec_results.append(("", f"Error: {str(error)}"))
                    if progress_bar is not None:
                        progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.close()
                    
                return all_exec_results
                        
        except Exception as e:
            # print(f"🚨 ProcessPool failed, falling back to sequential execution: {e}")
            # Fallback to sequential execution
            all_exec_results = []
            for code in all_code_snippets:
                result, report = self.execute(
                    code,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime=self.runtime,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )
                all_exec_results.append((result, report))
            return all_exec_results


def _test():
    batch_code = [
        """
# Create symbolic variables
x = sympy.symbols('x')
y = sympy.symbols('y')

# Create an expression
expr = x**2 + 2*x*y + y**2

print(f"Expression: {expr}")

# Calculate derivative
derivative = sympy.diff(expr, x)
print(f"Derivative with respect to x: {derivative}")

# Substitute specific values
result = expr.subs([(x, 1), (y, 2)])
print(f"Value at x=1, y=2: {result}")
        """,
        """
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        print(np.array([1, 2, 3]))
        """

    ]

    executor = PythonExecutor(get_answer_from_stdout=True)
    predictions = executor.apply(batch_code[0])
    print("Test Results:", predictions)


# Ray Task function - self-contained implementation to avoid module import issues
try:
    import ray
    
    @ray.remote
    def execute_code_ray_task(
        code: str,
        get_answer_from_stdout: bool = None,
        runtime_type: str = "generic",
        answer_symbol: str = None,
        answer_expr: str = None,
        timeout_length: int = 2,
    ) -> Tuple[str, str]:
        """Ray Task for executing a single code snippet - fully self-contained implementation"""
        
        # Import required modules
        import io
        import regex
        import pickle
        import copy
        import datetime
        import dateutil.relativedelta
        from timeout_decorator import timeout
        from contextlib import redirect_stdout
        import numpy as np
        import sympy
        import math
        from sympy import symbols, Eq, solve
        
        # Redefine Runtime class in Ray Task - self-contained implementation
        class TaskGenericRuntime:
            GLOBAL_DICT = {}
            LOCAL_DICT = None
            HEADERS = []

            def __init__(self):
                self._global_vars = copy.copy(self.GLOBAL_DICT)
                self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
                
                for c in self.HEADERS:
                    self.exec_code(c)

            def exec_code(self, code_piece: str) -> None:
                imports = """
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
x, y, z = sympy.symbols('x y z')
"""
                if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
                    raise RuntimeError()
                    
                exec(imports, self._global_vars)
                exec(code_piece, self._global_vars)
                
            def eval_code(self, expr: str):
                return eval(expr, self._global_vars)
            
            def inject(self, var_dict):
                for k, v in var_dict.items():
                    self._global_vars[k] = v
            
            @property
            def answer(self):
                return self._global_vars['answer']

        class TaskDateRuntime(TaskGenericRuntime):
            GLOBAL_DICT = {
                'datetime': datetime.datetime, 
                'timedelta': dateutil.relativedelta.relativedelta,
                'relativedelta': dateutil.relativedelta.relativedelta
            }

        class TaskCustomDict(dict):
            def __iter__(self):
                return list(super().__iter__()).__iter__()

        class TaskColorObjectRuntime(TaskGenericRuntime):
            GLOBAL_DICT = {'dict': TaskCustomDict}
        
        # Redefine execute method in Ray Task - self-contained implementation
        def task_execute(
            code,
            get_answer_from_stdout=None,
            runtime=None,
            answer_symbol=None,
            answer_expr=True,
            timeout_length=1,
        ):
            try:
                if isinstance(code, list):
                    code = '\n'.join(code)
                
                code = code.strip()
                
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)(code)
                    program_io.seek(0)
                    result = program_io.read()
                elif answer_symbol:
                    timeout(timeout_length)(runtime.exec_code)(code)
                    result = runtime._global_vars[answer_symbol]
                elif answer_expr:
                    timeout(timeout_length)(runtime.exec_code)(code)
                    result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
                else:
                    code_lines = code.split('\n')
                    if len(code_lines) > 1:
                        exec_code = '\n'.join(code_lines[:-1])
                        eval_code = code_lines[-1]
                        timeout(timeout_length)(runtime.exec_code)(exec_code)
                        result = timeout(timeout_length)(runtime.eval_code)(eval_code)
                    else:
                        result = timeout(timeout_length)(runtime.eval_code)(code)
                        
                report = "Done"
                
                try:
                    pickle.dumps(result)
                except (pickle.PicklingError, TypeError):
                    try:
                        result = str(result)
                    except:
                        result = f"<unprintable object of type {type(result).__name__}>"
                
            except Exception as e:
                result = ''
                report = str(e)
            return result, report
        
        # Create appropriate runtime
        if runtime_type == "date":
            runtime = TaskDateRuntime()
        elif runtime_type == "color":
            runtime = TaskColorObjectRuntime()
        else:
            runtime = TaskGenericRuntime()
        
        # Execute code
        return task_execute(
            code=code,
            get_answer_from_stdout=get_answer_from_stdout,
            runtime=runtime,
            answer_symbol=answer_symbol,
            answer_expr=answer_expr,
            timeout_length=timeout_length,
        )
        
except ImportError:
    # If Ray is not available, define a placeholder function
    def execute_code_ray_task(*args, **kwargs):
        raise RuntimeError("Ray is not available")


if __name__ == '__main__':
    _test()